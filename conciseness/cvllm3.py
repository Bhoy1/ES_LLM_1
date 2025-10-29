#permutations on gpu, not cpu in 2
#added iterations evaluation to see convergence


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import numpy as np
import os
import argparse
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import gc
import tempfile
import shutil
import json


try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("Warning: vLLM or PEFT not available.")


logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser()
# Model configuration
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_cache_dir', type=str, default='huggingface_cache')
parser.add_argument('--precision', type=str, default='bf16')
parser.add_argument('--gpu_threads', type=int, default=4, help='Number of parallel threads per GPU (only for full-param mode)')
parser.add_argument('--verbose', action='store_true', help='Print verbose logs')

# ES Hyperparameters
parser.add_argument('--num_iterations', type=int, default=1000, help='Number of ES iterations')
parser.add_argument('--population_size', type=int, default=30, help='Population size')
parser.add_argument('--sigma', type=float, default=0.001, help='Perturbation noise scale')
parser.add_argument('--alpha', type=float, default=0.0005, help='Learning rate')
parser.add_argument('--max_new_tokens', type=int, default=100, help='Max tokens to generate')
parser.add_argument('--do_sample', action='store_true', help='Use sampling instead of greedy')
parser.add_argument('--initial_seed', type=int, default=33, help='Random seed')
parser.add_argument('--checkpoint_interval', type=int, default=100, help='Evaluate on test set every N iterations')

# LoRA/vLLM configuration
parser.add_argument('--use_lora', action='store_true', help='Use LoRA mode (requires vLLM)')
parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha scaling')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
parser.add_argument('--lora_target_modules', type=str, 
    default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj',
    help='Comma-separated LoRA target modules')

# vLLM specific
parser.add_argument('--vllm_gpu_memory_utilization', type=float, default=0.9,
                   help='GPU memory utilization for vLLM')
parser.add_argument('--vllm_max_model_len', type=int, default=2048,
                   help='Maximum model length for vLLM')
parser.add_argument('--vllm_tensor_parallel_size', type=int, default=1,
                   help='Tensor parallel size for vLLM')

args = parser.parse_args()

# Set hyperparameters from args
NUM_ITERATIONS = args.num_iterations
POPULATION_SIZE = args.population_size
SIGMA = args.sigma
ALPHA = args.alpha
max_new_tokens = args.max_new_tokens
do_sample = args.do_sample
initial_seed = args.initial_seed


# --- Dummy Dataset and Reward Function ---
dataset = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]
# --- Test Dataset for Generalization Evaluation ---
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
    """Negative absolute difference in length"""
    return -abs(len(generated_text) - len(target_text))

def force_memory_cleanup():
    """Force aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


# ============================================================================
# LoRA/vLLM MODE FUNCTIONS
# ============================================================================

def create_base_lora_adapter(model_name, args):
    """Create the initial LoRA adapter configuration"""
    print(f"Creating base LoRA adapter with r={args.lora_r}, alpha={args.lora_alpha}")
    
    # Load base model temporarily to create LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=args.hf_cache_dir,
        torch_dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16,
        device_map='cuda',  # Load on CPU to save GPU memory
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
    return lora_params


def perturb_and_save_lora(base_lora_model, seed, sigma, temp_dir, adapter_id):
    """
    Perturb LoRA weights and save to disk
    Returns path to saved adapter
    """
    adapter_path = os.path.join(temp_dir, f"adapter_{adapter_id}")
    os.makedirs(adapter_path, exist_ok=True)
    
    # Get LoRA parameters
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


def evaluate_population_vllm(llm, adapter_paths, dataset, tokenizer, seeds_info, args):
    """
    Evaluate all perturbed adapters using vLLM with dynamic adapter swapping
    """
    print(f"Evaluating {len(adapter_paths)} adapters with vLLM...")
    
    # Prepare all prompts (each adapter evaluates on all dataset items)
    all_prompts = []
    prompt_metadata = []  # Track which adapter and dataset item each prompt belongs to
    
    for adapter_idx, adapter_path in enumerate(adapter_paths):
        for data_idx, (input_text, target_text) in enumerate(dataset):
            all_prompts.append(input_text)
            prompt_metadata.append({
                'adapter_idx': adapter_idx,
                'adapter_path': adapter_path,
                'data_idx': data_idx,
                'target_text': target_text
            })
    
    # Create LoRARequest objects for each unique adapter
    lora_requests = []
    for adapter_idx, adapter_path in enumerate(adapter_paths):
        lora_request = LoRARequest(
            lora_name=f"adapter_{adapter_idx}",
            lora_int_id=adapter_idx + 1,  # Must be > 0
            lora_local_path=adapter_path
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
    
    # Generate with vLLM (all adapters in parallel)
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
        adapter_idx = seed_idx
        rewards = adapter_rewards[adapter_idx]
        average_reward = sum(rewards) / len(rewards)
        results.append((seed_idx, average_reward))
    
    return results


def evaluate_on_test_set_vllm(llm, adapter_path, test_dataset, args):
    """
    Evaluate a single adapter on the test dataset using vLLM
    Returns list of results with prompt, target, generated, reward
    """
    sampling_params = SamplingParams(
        temperature=0.0 if not args.do_sample else 1.0,
        max_tokens=args.max_new_tokens
    )
    
    lora_request = LoRARequest("test_eval", 1, adapter_path)
    
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
    """
    Update LoRA weights using ES gradient estimate
    """
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


# ============================================================================
# FULL-PARAM MODE FUNCTIONS (Original Transformers-based)
# ============================================================================

def evaluate_model(model, tokenizer, input_text, target_text, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False):
    """
    Generate a response from the model given an input (single or batch) and compute rewards.
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    # Handle both single input and batch input
    is_batch = isinstance(input_text, list)
    input_texts = input_text if is_batch else [input_text]
    target_texts = target_text if is_batch else [target_text]

    # Batch tokenization
    tokenized_inputs = tokenizer(input_texts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
    with torch.inference_mode():
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)

    del input_ids, outputs
    torch.cuda.empty_cache()

    # Compute rewards for batch texts
    rewards = [compute_reward(gen_text, tgt_text) for gen_text, tgt_text in zip(generated_texts, target_texts)]

    if return_text:
        return rewards, generated_texts
    else:
        return rewards


def evaluate_on_test_set_fullparam(model, tokenizer, test_dataset, accelerator):
    """
    Evaluate model on test dataset for full-param mode
    Returns list of results with prompt, target, generated, reward
    """
    test_results = []
    
    for prompt, target in test_dataset:
        rewards, generated_texts = evaluate_model(
            model, tokenizer, [prompt], [target], 
            accelerator, return_text=True
        )
        
        test_results.append({
            'prompt': prompt,
            'target': target,
            'generated': generated_texts[0],
            'reward': rewards[0]
        })
    
    return test_results


def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Weight perturbation
    seed_shift = 0
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed+seed_shift))
        seed_shift += 1

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)

    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # Evaluate all prompts with perturbed weights in batch
    input_texts = [input_text for input_text, _ in dataset]
    target_texts = [target_text for _, target_text in dataset]
    rewards = evaluate_model(model, tokenizer, input_texts, target_texts, accelerator,
                           seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=False)
    total_reward = sum(rewards)

    # Restore original weights (direct inplace modification)
    seed_shift = 0
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)
        gen.manual_seed(int(seed+seed_shift))
        seed_shift += 1

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    average_reward = total_reward / len(dataset)

    force_memory_cleanup()

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    return seed_idx, average_reward


# ============================================================================
# MAIN FUNCTION WITH MODE SWITCHING
# ============================================================================

def main():
    # Check if LoRA mode is requested but not available
    if args.use_lora and not VLLM_AVAILABLE:
        raise RuntimeError("LoRA mode requested but vLLM/PEFT not available. Install with: pip install vllm peft")
    
    # Set up mode
    mode = "LoRA+vLLM" if args.use_lora else "Full-Param+Transformers"
    print(f"\n{'='*60}")
    print(f"Running in {mode} mode")
    print(f"{'='*60}\n")
    
    if args.use_lora:
        main_lora_mode()
    else:
        main_fullparam_mode()


def main_lora_mode():
    """Main function for LoRA+vLLM mode"""
    print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
    print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    
    model_name = args.model_name
    
    # Create temporary directory for adapters
    temp_dir = tempfile.mkdtemp(prefix="es_lora_adapters_")
    base_adapter_dir = os.path.join(temp_dir, "base_adapter")
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Step 1: Create base LoRA adapter
        print(f"\nLoading model {model_name} and creating LoRA adapter...")
        base_peft_model, lora_config = create_base_lora_adapter(model_name, args)
        
        # Save base adapter
        base_peft_model.save_pretrained(base_adapter_dir)
        print("Base LoRA adapter created and saved successfully")
        
        # CRITICAL FIX: Delete model before vLLM initialization
        print("\n[DEBUG] Deleting base_peft_model before vLLM init...")
        del base_peft_model
        force_memory_cleanup()
        print(f"[DEBUG] GPU memory after deletion: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        # Load tokenizer
        print("[DEBUG] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=args.hf_cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        print("[DEBUG] Tokenizer loaded successfully")
        
        # Step 2: Initialize vLLM with LoRA support (clean GPU state)
        print("\n[DEBUG] Initializing vLLM with LoRA support...")
        print(f"[DEBUG] Config: max_loras=128, lora_rank={args.lora_r}")
        
        import sys
        sys.stdout.flush()
        
        llm = LLM(
            model=model_name,
            download_dir=args.hf_cache_dir,
            enable_lora=True,
            max_loras=128,
            max_lora_rank=args.lora_r,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=args.vllm_max_model_len,
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            dtype=args.precision,
            enforce_eager=True,
        )
        print("\n[DEBUG] vLLM initialized successfully!")
        
        # Reload base model for perturbations
        print("[DEBUG] Reloading base LoRA model from saved adapter...")
        base_model_for_lora = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=args.hf_cache_dir,
            torch_dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16,
            device_map='cuda',
        )
        base_peft_model = PeftModel.from_pretrained(base_model_for_lora, base_adapter_dir)
        print("[DEBUG] Base LoRA model reloaded successfully")
        
        force_memory_cleanup()
        
        # Record training start time
        training_start_time = time.time()
        np.random.seed(initial_seed)
        
        # Initialize metrics logging
        metrics_log = []
        checkpoint_history = []  # Track checkpoint evaluations

        # Prepare save directory name
        question_num = len(dataset)
        save_dir = f"finetuned_{model_name}_es_lora_vllm_seed{initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_r{args.lora_r}_questions{question_num}"
        save_dir = save_dir.replace('/', '_')

        # Step 3: Main ES loop
        for iteration in range(NUM_ITERATIONS):
            iter_start_time = time.time()
            force_memory_cleanup()
            
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
            print(f"{'='*60}")
            
            # Generate seeds
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_info = [(idx, seed) for idx, seed in enumerate(seeds)]
            
            # Perturb and save all adapters
            print(f"Creating {POPULATION_SIZE} perturbed LoRA adapters...")
            adapter_paths = []
            perturb_start = time.time()
            
            for seed_idx, seed in seeds_info:
                adapter_path = perturb_and_save_lora(
                    base_peft_model,
                    seed,
                    SIGMA,
                    temp_dir,
                    seed_idx
                )
                adapter_paths.append(adapter_path)
            
            perturb_time = time.time() - perturb_start
            print(f"Perturbation time: {perturb_time:.2f}s")
            
            # Evaluate all adapters in one batched vLLM call
            eval_start = time.time()
            results = evaluate_population_vllm(
                llm,
                adapter_paths,
                dataset,
                tokenizer,
                seeds_info,
                args
            )
            eval_time = time.time() - eval_start
            print(f"Evaluation time: {eval_time:.2f}s")
            
            rewards = [r for _, r in results]
            
            # Normalize rewards
            rewards_tensor = np.array(rewards, dtype=np.float32)
            rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            
            # Update base LoRA weights using ES
            print("Updating base LoRA weights...")
            update_lora_weights_es(base_peft_model, seeds, rewards_normalized, SIGMA, ALPHA)
            
            # Save updated base adapter
            base_peft_model.save_pretrained(base_adapter_dir)
            
            # Clean up perturbed adapters
            for adapter_path in adapter_paths:
                if os.path.exists(adapter_path):
                    shutil.rmtree(adapter_path)
            
            force_memory_cleanup()
            
            # Log results
            iter_time = time.time() - iter_start_time
            mean_reward = rewards_tensor.mean().item()
            min_reward = rewards_tensor.min().item()
            max_reward = rewards_tensor.max().item()

            # Log metrics for this iteration
            metrics_log.append({
                'iteration': iteration + 1,
                'time': iter_time,
                'mean_reward': mean_reward,
                'min_reward': min_reward,
                'max_reward': max_reward,
                'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            })
            
            print(f"\nIteration {iteration + 1}/{NUM_ITERATIONS} completed in {iter_time:.2f}s")
            print(f"  Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
            print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            
            # Check if this is a checkpoint iteration
            if (iteration + 1) % args.checkpoint_interval == 0:
                checkpoint_start = time.time()
                
                print(f"\n{'='*80}")
                print(f"CHECKPOINT EVALUATION - Iteration {iteration + 1}")
                print(f"{'='*80}")
                
                # Calculate rolling window training mean
                window_start = max(0, iteration + 1 - args.checkpoint_interval)
                window_end = iteration + 1
                window_rewards = [m['mean_reward'] for m in metrics_log[window_start:window_end]]
                training_mean = np.mean(window_rewards)
                
                # Run test evaluation
                test_results = evaluate_on_test_set_vllm(llm, base_adapter_dir, test_dataset, args)
                test_mean_reward = np.mean([r['reward'] for r in test_results])
                
                # Save test results to JSON
                test_results_file = f"{save_dir}_eval_iter{iteration + 1}.json"
                with open(test_results_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                
                # Calculate elapsed time
                elapsed_minutes = (time.time() - training_start_time) / 60
                
                # Print checkpoint summary
                print(f"Training reward (mean of iters {window_start + 1}-{window_end}): {training_mean:.2f}")
                print(f"Test mean reward: {test_mean_reward:.2f}")
                print(f"Elapsed time: {elapsed_minutes:.1f} minutes")
                
                # Calculate and print trend if not first checkpoint
                if len(checkpoint_history) > 0:
                    prev_train = checkpoint_history[-1]['training_mean']
                    prev_test = checkpoint_history[-1]['test_mean']
                    train_diff = training_mean - prev_train
                    test_diff = test_mean_reward - prev_test
                    train_arrow = '↑' if train_diff > 0 else ('↓' if train_diff < 0 else '→')
                    test_arrow = '↑' if test_diff > 0 else ('↓' if test_diff < 0 else '→')
                    print(f"Trend: Train {train_arrow} {train_diff:+.2f} | Test {test_arrow} {test_diff:+.2f}")
                
                print(f"{'='*80}\n")
                
                # Save checkpoint info
                checkpoint_history.append({
                    'iteration': iteration + 1,
                    'training_mean': training_mean,
                    'test_mean': test_mean_reward,
                    'window_start': window_start + 1,
                    'window_end': window_end
                })
                
                checkpoint_time = time.time() - checkpoint_start
                print(f"Checkpoint evaluation completed in {checkpoint_time:.2f}s\n")
            
            del rewards_tensor, rewards_normalized
            force_memory_cleanup()
        
        total_time = time.time() - training_start_time
        
        # Save final model
        print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        print(f"\nSaving final LoRA adapter to {save_dir}...")
        base_peft_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Model saved successfully.")
        
        # Save training metrics log
        metrics_file = f"{save_dir}_training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_log, f, indent=2)
        print(f"Training metrics saved to {metrics_file}")
        
        # Print final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print("Checkpoint History:")
        for cp in checkpoint_history:
            print(f"  Iter {cp['iteration']:4d} | Train ({cp['window_start']}-{cp['window_end']}): {cp['training_mean']:8.2f} | Test: {cp['test_mean']:8.2f}")
        
        # Find best test reward
        if checkpoint_history:
            best_checkpoint = max(checkpoint_history, key=lambda x: x['test_mean'])
            print(f"\nBest test reward: {best_checkpoint['test_mean']:.2f} (at iteration {best_checkpoint['iteration']})")
        
        print(f"Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print("Model saved successfully.")
        print("="*80)
        
    finally:
        # Cleanup temp directory
        print("\nCleaning up temporary files...")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        print("Cleanup complete.")


def main_fullparam_mode():
    """Main function for full-param+Transformers mode (original)"""
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"Total processes: {accelerator.num_processes}, GPU threads per process: {args.gpu_threads}")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")
        print(f"Checkpoint interval: {args.checkpoint_interval}")

    # Load model
    model_name = args.model_name
    hf_cache_dir = args.hf_cache_dir

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")

    # Load multiple model copies for threading
    model_list = []
    for model_index in range(args.gpu_threads):
        model_list.append(AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=hf_cache_dir,
            device_map={"": accelerator.process_index},
            torch_dtype=torch.float16 if args.precision == 'fp16' else (torch.bfloat16 if args.precision == 'bf16' else torch.float32),
        ))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, cache_dir=hf_cache_dir)

    if accelerator.is_main_process:
        print("Model loaded successfully")

    # Prepare models
    for model in model_list:
        model.eval()

    force_memory_cleanup()

    # Record training start time
    training_start_time = time.time()
    np.random.seed(initial_seed)

    # Initialize metrics logging
    metrics_log = []
    checkpoint_history = []  # Track checkpoint evaluations

    # Prepare save directory name
    question_num = len(dataset)
    save_dir = f"finetuned_{model_name}_es_fullparam_seed{initial_seed}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_{args.precision}_threads{args.gpu_threads}_questions{question_num}"
    save_dir = save_dir.replace('/', '_')

    for iteration in range(NUM_ITERATIONS):
        iter_start_time = time.time()
        force_memory_cleanup()

        if args.verbose:
            print(f"Process {accelerator.process_index} starting iteration {iteration + 1}/{NUM_ITERATIONS}")

        # Generate seeds on main process only
        if accelerator.is_main_process:
            if args.verbose:
                print(f"Main process {accelerator.process_index} generating seeds")
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            if args.verbose:
                print(f"Worker process {accelerator.process_index} waiting for seeds")
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        # Broadcast seeds from main process to all processes
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()

        if args.verbose:
            print(f"Process {accelerator.process_index} received seeds")

        # Assign seeds to each process
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        if args.verbose:
            print(f"Process {accelerator.process_index} assigned {len(local_seeds)} seeds: {[idx for idx, _ in local_seeds]}")

        # Process seeds in batches
        local_rewards = []
        batch_size = max(1, min(args.gpu_threads, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, args.verbose))

                results = list(executor.map(process_seed, thread_args))
                local_rewards.extend(results)

            force_memory_cleanup()

        # Collect rewards from all processes
        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)

        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        if accelerator.num_processes > 1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)

        rewards = all_rewards.cpu().tolist()
        del all_rewards
        force_memory_cleanup()

        # Normalize rewards
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Update model weights
        if args.verbose:
            print(f"Process {accelerator.process_index} updating model weights")
        
        original_model = model_list[0]
        seed_shift = 0
        for name, param in original_model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed+seed_shift))

                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            torch.cuda.empty_cache()
            seed_shift += 1

        # Sync weights across model copies
        for model_idx in range(1, len(model_list)):
            original_model_tmp = model_list[model_idx]
            for name, param in original_model_tmp.named_parameters():
                param.data.copy_(original_model.get_parameter(name).data.clone())

        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time
        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        # Log metrics for this iteration
        if accelerator.is_main_process:
            metrics_log.append({
                'iteration': iteration + 1,
                'time': iter_time,
                'mean_reward': mean_reward,
                'min_reward': min_reward,
                'max_reward': max_reward,
                'gpu_memory_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            })

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

        if accelerator.is_main_process:
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")
            
            # Check if this is a checkpoint iteration
            if (iteration + 1) % args.checkpoint_interval == 0:
                checkpoint_start = time.time()
                
                print(f"\n{'='*80}")
                print(f"CHECKPOINT EVALUATION - Iteration {iteration + 1}")
                print(f"{'='*80}")
                
                # Calculate rolling window training mean
                window_start = max(0, iteration + 1 - args.checkpoint_interval)
                window_end = iteration + 1
                window_rewards = [m['mean_reward'] for m in metrics_log[window_start:window_end]]
                training_mean = np.mean(window_rewards)
                
                # Run test evaluation
                test_results = evaluate_on_test_set_fullparam(original_model, tokenizer, test_dataset, accelerator)
                test_mean_reward = np.mean([r['reward'] for r in test_results])
                
                # Save test results to JSON
                test_results_file = f"{save_dir}_eval_iter{iteration + 1}.json"
                with open(test_results_file, 'w') as f:
                    json.dump(test_results, f, indent=2)
                
                # Calculate elapsed time
                elapsed_minutes = (time.time() - training_start_time) / 60
                
                # Print checkpoint summary
                print(f"Training reward (mean of iters {window_start + 1}-{window_end}): {training_mean:.2f}")
                print(f"Test mean reward: {test_mean_reward:.2f}")
                print(f"Elapsed time: {elapsed_minutes:.1f} minutes")
                
                # Calculate and print trend if not first checkpoint
                if len(checkpoint_history) > 0:
                    prev_train = checkpoint_history[-1]['training_mean']
                    prev_test = checkpoint_history[-1]['test_mean']
                    train_diff = training_mean - prev_train
                    test_diff = test_mean_reward - prev_test
                    train_arrow = '↑' if train_diff > 0 else ('↓' if train_diff < 0 else '→')
                    test_arrow = '↑' if test_diff > 0 else ('↓' if test_diff < 0 else '→')
                    print(f"Trend: Train {train_arrow} {train_diff:+.2f} | Test {test_arrow} {test_diff:+.2f}")
                
                print(f"{'='*80}\n")
                
                # Save checkpoint info
                checkpoint_history.append({
                    'iteration': iteration + 1,
                    'training_mean': training_mean,
                    'test_mean': test_mean_reward,
                    'window_start': window_start + 1,
                    'window_end': window_end
                })
                
                checkpoint_time = time.time() - checkpoint_start
                print(f"Checkpoint evaluation completed in {checkpoint_time:.2f}s\n")

    total_time = time.time() - training_start_time

    # Save the fine-tuned model and run evaluations
    if accelerator.is_main_process:
        print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        
        print(f"\nSaving model to {save_dir}...")
        original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print("Model saved successfully.")
        
        # Save training metrics log
        metrics_file = f"{save_dir}_training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_log, f, indent=2)
        print(f"Training metrics saved to {metrics_file}")
        
        # Print final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print("Checkpoint History:")
        for cp in checkpoint_history:
            print(f"  Iter {cp['iteration']:4d} | Train ({cp['window_start']}-{cp['window_end']}): {cp['training_mean']:8.2f} | Test: {cp['test_mean']:8.2f}")
        
        # Find best test reward
        if checkpoint_history:
            best_checkpoint = max(checkpoint_history, key=lambda x: x['test_mean'])
            print(f"\nBest test reward: {best_checkpoint['test_mean']:.2f} (at iteration {best_checkpoint['iteration']})")
        
        print(f"Total training time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print("Model saved successfully.")
        print("="*80)


if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()
