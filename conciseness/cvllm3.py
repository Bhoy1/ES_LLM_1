#conciseness/cvllm3.2.py
#cleaned up version of working


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import os
import argparse
import time
import gc
import tempfile
import shutil
import json

from peft import LoraConfig, get_peft_model, PeftModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description='ES fine-tuning with LoRA and vLLM')

# Model configuration
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='huggingface_cache')
parser.add_argument('--precision', type=str, default='bfloat16', choices=['bfloat16', 'float16'])

# ES Hyperparameters
parser.add_argument('--num_iterations', type=int, default=1000)
parser.add_argument('--population_size', type=int, default=30)
parser.add_argument('--sigma', type=float, default=0.001, help='Perturbation noise scale')
parser.add_argument('--alpha', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--initial_seed', type=int, default=33)
parser.add_argument('--checkpoint_interval', type=int, default=100)

# Generation parameters
parser.add_argument('--max_new_tokens', type=int, default=100)
parser.add_argument('--do_sample', action='store_true', help='Use sampling instead of greedy')

# LoRA configuration
parser.add_argument('--lora_r', type=int, default=8)
parser.add_argument('--lora_alpha', type=int, default=32)
parser.add_argument('--lora_dropout', type=float, default=0.1)
parser.add_argument('--lora_target_modules', type=str, 
    default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')

# vLLM configuration
parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9)
parser.add_argument('--vllm_max_model_len', type=int, default=2048)
parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1)

args = parser.parse_args()

# Training dataset
dataset = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]

# Test dataset for generalization evaluation
test_dataset = [
    ("What is the capital of France?", "Paris"),
    ("Calculate: 12×7=", "84"),
    ("Is the statement \"All cats are mammals\" true or false?", "True"),
    ("What comes next in the sequence: 2,4,6,8, ?", "10"),
    ("Translate \"Hello\" to Spanish:", "Hola"),
    ("What is 15% of 200?", "30"),
    ("Name one primary color:", "Red"),
    ("How many days are in a week?", "7"),
]


def compute_reward(generated_text, target_text):
    """Reward function: negative absolute difference in length"""
    return -abs(len(generated_text) - len(target_text))


def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def create_base_lora_adapter(model_name, args):
    """Create the initial LoRA adapter configuration"""
    print(f"Creating base LoRA adapter with r={args.lora_r}, alpha={args.lora_alpha}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=args.hf_cache_dir,
        torch_dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16,
        device_map='cuda',
    )
    
    target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
    
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    
    peft_model = get_peft_model(base_model, lora_config)
    
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total params: {total_params:,}")
    
    return peft_model, lora_config


def get_lora_parameters(peft_model):
    """Get only the LoRA parameters (A and B matrices)"""
    lora_params = {}
    for name, param in peft_model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_params[name] = param
    
    if len(lora_params) == 0:
        raise ValueError("No LoRA parameters found in model!")
    
    return lora_params


def perturb_and_save_lora(base_lora_model, seed, sigma, temp_dir, adapter_id):
    """Perturb LoRA weights and save to disk. Returns path to saved adapter."""
    adapter_path = os.path.join(temp_dir, f"adapter_{adapter_id}")
    os.makedirs(adapter_path, exist_ok=True)
    
    lora_params = get_lora_parameters(base_lora_model)
    
    # Perturb weights
    seed_shift = 0
    for name, param in lora_params.items():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed + seed_shift))
        seed_shift += 1
        
        noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
        param.data.add_(sigma * noise)
    
    # Save perturbed adapter
    base_lora_model.save_pretrained(adapter_path)
    
    # Restore original weights
    seed_shift = 0
    for name, param in lora_params.items():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed + seed_shift))
        seed_shift += 1
        
        noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
        param.data.add_(-sigma * noise)
    
    return adapter_path


def evaluate_population_vllm(llm, adapter_paths, dataset, seeds_info, args, global_id_start):
    """Evaluate all perturbed adapters using vLLM with globally unique IDs"""
    print(f"Evaluating {len(adapter_paths)} adapters with vLLM...")
    
    # Prepare all prompts (each adapter evaluates on all dataset items)
    all_prompts = []
    prompt_metadata = []
    
    for adapter_idx, adapter_path in enumerate(adapter_paths):
        for data_idx, (input_text, target_text) in enumerate(dataset):
            all_prompts.append(input_text)
            prompt_metadata.append({
                'adapter_idx': adapter_idx,
                'target_text': target_text
            })
    
    # Create LoRARequest objects with globally unique IDs
    lora_requests = []
    for adapter_idx, adapter_path in enumerate(adapter_paths):
        global_lora_id = global_id_start + adapter_idx
        lora_request = LoRARequest(
            lora_name=f"adapter_{global_lora_id}",
            lora_int_id=global_lora_id,
            lora_path=adapter_path
        )
        lora_requests.append(lora_request)
    
    # Map each prompt to its LoRA request
    prompt_lora_requests = [lora_requests[meta['adapter_idx']] for meta in prompt_metadata]
    
    # Set up sampling params
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.0 if not args.do_sample else 1.0,
        top_p=1.0 if not args.do_sample else 0.9,
    )
    
    # Generate with vLLM
    outputs = llm.generate(
        prompts=all_prompts,
        sampling_params=sampling_params,
        lora_request=prompt_lora_requests,
    )
    
    # Organize results by adapter
    adapter_rewards = {}
    for output, meta in zip(outputs, prompt_metadata):
        adapter_idx = meta['adapter_idx']
        target_text = meta['target_text']
        generated_text = output.outputs[0].text
        reward = compute_reward(generated_text, target_text)
        
        if adapter_idx not in adapter_rewards:
            adapter_rewards[adapter_idx] = []
        adapter_rewards[adapter_idx].append(reward)
    
    # Compute average reward per adapter
    results = []
    for seed_idx, seed in seeds_info:
        rewards = adapter_rewards[seed_idx]
        average_reward = sum(rewards) / len(rewards)
        results.append((seed_idx, average_reward))
    
    return results


def evaluate_on_test_set_vllm(llm, adapter_path, test_dataset, args, global_lora_id):
    """Evaluate a single adapter on the test dataset using globally unique ID"""
    sampling_params = SamplingParams(
        temperature=0.0 if not args.do_sample else 1.0,
        max_tokens=args.max_new_tokens
    )
    
    lora_request = LoRARequest(f"test_eval_{global_lora_id}", global_lora_id, adapter_path)
    
    test_results = []
    for prompt, target in test_dataset:
        outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
        generated_text = outputs[0].outputs[0].text
        reward = compute_reward(generated_text, target)
        
        test_results.append({
            'prompt': prompt,
            'target': target,
            'generated': generated_text,
            'reward': reward
        })
    
    return test_results


def update_lora_weights_es(base_peft_model, seeds, rewards_normalized, sigma, alpha):
    """Update LoRA weights using ES gradient estimate"""
    lora_params = get_lora_parameters(base_peft_model)
    
    seed_shift = 0
    for name, param in lora_params.items():
        gen = torch.Generator(device=param.device)
        update = torch.zeros_like(param)
        
        for seed_idx in range(len(seeds)):
            r_norm = rewards_normalized[seed_idx]
            seed = seeds[seed_idx]
            gen.manual_seed(int(seed + seed_shift))
            
            noise = torch.randn(param.shape, generator=gen, device=param.device, dtype=param.dtype)
            noise.mul_(float(r_norm))
            update.add_(noise)
            del noise
        
        update.div_(len(seeds))
        param.data.add_(alpha * update)
        seed_shift += 1
    
    force_memory_cleanup()


def main():
    print(f"\n{'='*60}")
    print(f"ES Fine-tuning with LoRA + vLLM")
    print(f"{'='*60}\n")
    print(f"Model: {args.model_name}")
    print(f"Population size: {args.population_size}, Iterations: {args.num_iterations}")
    print(f"Sigma: {args.sigma}, Alpha: {args.alpha}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    
    # Create temporary directory for adapters
    temp_dir = tempfile.mkdtemp(prefix="es_lora_adapters_")
    base_adapter_dir = os.path.join(temp_dir, "base_adapter")
    print(f"Using temp directory: {temp_dir}\n")
    
    try:
        # Step 1: Create base LoRA adapter
        print(f"Loading model and creating LoRA adapter...")
        base_peft_model, lora_config = create_base_lora_adapter(args.model_name, args)
        base_peft_model.save_pretrained(base_adapter_dir)
        print("Base LoRA adapter created and saved\n")
        
        # Delete model before vLLM initialization to free GPU memory
        del base_peft_model
        force_memory_cleanup()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, 
                                                  cache_dir=args.hf_cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        
        # Step 2: Initialize vLLM with LoRA support
        print("Initializing vLLM with LoRA support...")
        llm = LLM(
            model=args.model_name,
            download_dir=args.hf_cache_dir,
            enable_lora=True,
            max_loras=args.population_size + 1,  # Population + 1 for test eval
            max_lora_rank=args.lora_r,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            dtype=args.precision,
            enforce_eager=True,
        )
        print("vLLM initialized\n")
        
        # Reload base model for perturbations
        print("Reloading base LoRA model for training...")
        base_model_for_lora = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            cache_dir=args.hf_cache_dir,
            torch_dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16,
            device_map='cuda',
        )
        base_peft_model = PeftModel.from_pretrained(base_model_for_lora, base_adapter_dir, 
                                                     is_trainable=True)
        print("Base LoRA model reloaded\n")
        
        force_memory_cleanup()
        
        # Initialize global LoRA ID counter for unique adapter IDs
        global_lora_id = 0
        
        # Training setup
        training_start_time = time.time()
        np.random.seed(args.initial_seed)
        
        metrics_log = []
        checkpoint_history = []
        
        # Prepare save directory name
        save_dir = (f"finetuned_{args.model_name}_es_lora_"
                   f"seed{args.initial_seed}_pop{args.population_size}_"
                   f"iter{args.num_iterations}_sigma{args.sigma}_alpha{args.alpha}_"
                   f"r{args.lora_r}_q{len(dataset)}")
        save_dir = save_dir.replace('/', '_')
        
        # Main ES training loop
        print("="*60)
        print("Starting ES Training")
        print("="*60 + "\n")
        
        for iteration in range(args.num_iterations):
            iter_start_time = time.time()
            force_memory_cleanup()
            
            print(f"Iteration {iteration + 1}/{args.num_iterations}")
            
            # Generate seeds for this iteration
            seeds = np.random.randint(0, 2**30, size=args.population_size, dtype=np.int64).tolist()
            seeds_info = [(idx, seed) for idx, seed in enumerate(seeds)]
            
            # Create perturbed adapters
            adapter_paths = []
            for seed_idx, seed in seeds_info:
                adapter_path = perturb_and_save_lora(
                    base_peft_model, seed, args.sigma, temp_dir, seed_idx
                )
                adapter_paths.append(adapter_path)
            
            # Evaluate all adapters with globally unique IDs
            results = evaluate_population_vllm(
                llm, adapter_paths, dataset, seeds_info, args,
                global_id_start=global_lora_id + 1
            )
            
            # Increment global ID counter
            global_lora_id += args.population_size
            
            rewards = [r for _, r in results]
            
            # Normalize rewards
            rewards_tensor = np.array(rewards, dtype=np.float32)
            rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            
            # Update base LoRA weights using ES
            update_lora_weights_es(base_peft_model, seeds, rewards_normalized, args.sigma, args.alpha)
            
            # Save updated base adapter
            base_peft_model.save_pretrained(base_adapter_dir)
            
            # Clean up perturbed adapters
            for adapter_path in adapter_paths:
                if os.path.exists(adapter_path):
                    shutil.rmtree(adapter_path)
            
            force_memory_cleanup()
            
            # Log metrics
            iter_time = time.time() - iter_start_time
            mean_reward = rewards_tensor.mean().item()
            min_reward = rewards_tensor.min().item()
            max_reward = rewards_tensor.max().item()
            
            metrics_log.append({
                'iteration': iteration + 1,
                'time': iter_time,
                'mean_reward': mean_reward,
                'min_reward': min_reward,
                'max_reward': max_reward,
                'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                'global_lora_id': global_lora_id
            })
            
            print(f"  Time: {iter_time:.2f}s | Mean: {mean_reward:.2f} | "
                  f"Min: {min_reward:.2f} | Max: {max_reward:.2f}")
            
            # Checkpoint evaluation
            if (iteration + 1) % args.checkpoint_interval == 0:
                print(f"\n{'='*60}")
                print(f"CHECKPOINT - Iteration {iteration + 1}")
                print(f"{'='*60}")
                
                # Calculate training mean over checkpoint window
                window_start = max(0, iteration + 1 - args.checkpoint_interval)
                window_end = iteration + 1
                window_rewards = [m['mean_reward'] for m in metrics_log[window_start:window_end]]
                training_mean = np.mean(window_rewards)
                
                # Run test evaluation with unique global ID
                global_lora_id += 1
                test_results = evaluate_on_test_set_vllm(
                    llm, base_adapter_dir, test_dataset, args, global_lora_id=global_lora_id
                )
                test_mean_reward = np.mean([r['reward'] for r in test_results])
                
                # Save test results
                test_results_file = f"{save_dir}_eval_iter{iteration + 1}.json"
                with open(test_results_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                
                elapsed_minutes = (time.time() - training_start_time) / 60
                
                print(f"Training (iters {window_start + 1}-{window_end}): {training_mean:.2f}")
                print(f"Test: {test_mean_reward:.2f}")
                print(f"Elapsed: {elapsed_minutes:.1f} min")
                
                # Show trend
                if len(checkpoint_history) > 0:
                    prev_train = checkpoint_history[-1]['training_mean']
                    prev_test = checkpoint_history[-1]['test_mean']
                    train_diff = training_mean - prev_train
                    test_diff = test_mean_reward - prev_test
                    train_arrow = '↑' if train_diff > 0 else ('↓' if train_diff < 0 else '→')
                    test_arrow = '↑' if test_diff > 0 else ('↓' if test_diff < 0 else '→')
                    print(f"Trend: Train {train_arrow} {train_diff:+.2f} | Test {test_arrow} {test_diff:+.2f}")
                
                print(f"{'='*60}\n")
                
                checkpoint_history.append({
                    'iteration': iteration + 1,
                    'training_mean': training_mean,
                    'test_mean': test_mean_reward,
                    'window_start': window_start + 1,
                    'window_end': window_end
                })
            
            del rewards_tensor, rewards_normalized
            force_memory_cleanup()
        
        total_time = time.time() - training_start_time
        
        # Save final model
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"\nSaving final LoRA adapter to {save_dir}...")
        base_peft_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        # Save training metrics
        metrics_file = f"{save_dir}_training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_log, f, indent=2)
        print(f"Training metrics saved to {metrics_file}")
        
        # Print final summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print("Checkpoint History:")
        for cp in checkpoint_history:
            print(f"  Iter {cp['iteration']:4d} | "
                  f"Train ({cp['window_start']}-{cp['window_end']}): {cp['training_mean']:8.2f} | "
                  f"Test: {cp['test_mean']:8.2f}")
        
        if checkpoint_history:
            best_checkpoint = max(checkpoint_history, key=lambda x: x['test_mean'])
            print(f"\nBest test reward: {best_checkpoint['test_mean']:.2f} "
                  f"(at iteration {best_checkpoint['iteration']})")
        
        print(f"\nTotal time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print("="*60)
        
    finally:
        # Cleanup temp directory
        print("\nCleaning up temporary files...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("Cleanup complete.")


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    main()