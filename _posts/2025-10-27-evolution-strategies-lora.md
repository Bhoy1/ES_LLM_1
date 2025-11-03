---


---

<hr>
<h2 id="layout-postv3title-evolution-strategies-with-loradate-2025-11-3categories-research-llms-evolution-strategies">layout: post(v3)<br>
title: “Evolution Strategies with LoRA”<br>
date: 2025-11-3<br>
categories: [research, LLMs, evolution-strategies]</h2>
<h2 id="overview">Overview</h2>
<p>This post presents early experimental results from ongoing work on applying Evolution Strategies (ES) to optimize LoRA adapters instead of full model parameters.  While LoRA fine tuning is highly parameter efficient, its performance is unclear if it will match full parameter fine tunning from early experiments.</p>
<p>This work builds on the framework introduced in <em>Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning</em> (<a href="https://arxiv.org/abs/2509.24372">arXiv:2509.24372v1</a>), which proposed the conciseness task as a lightweight benchmark for evaluating ES based fine tuning methods. The experiment here focuses on that same task, training on two prompts and evaluating on eight distinct test prompts to examine generalization and reward stability.</p>
<p>This is the first evaluation of LoRA under ES. The goal is not to draw definitive conclusions, but to better understand how low rank parameterization interacts with gradient free optimization and to identify directions for improvement in future iterations.</p>
<h2 id="background">Background</h2>
<p>Evolution Strategies (ES) is a gradient free optimization method that updates model parameters via random perturbations and reward weighted averaging:</p>
<p><span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msub><mi>θ</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><msub><mi>θ</mi><mi>t</mi></msub><mo>+</mo><mi>η</mi><mfrac><mn>1</mn><mrow><mi>N</mi><mi>σ</mi></mrow></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><msub><mi>r</mi><mi>i</mi></msub><msub><mi>ε</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">
\theta_{t+1} = \theta_t + \eta \frac{1}{N\sigma} \sum_{i=1}^N r_i \varepsilon_i
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.902771em; vertical-align: -0.208331em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.02778em;">θ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.301108em;"><span class="" style="top: -2.55em; margin-left: -0.02778em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">t</span><span class="mbin mtight">+</span><span class="mord mtight">1</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.208331em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.84444em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.02778em;">θ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.280556em;"><span class="" style="top: -2.55em; margin-left: -0.02778em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">t</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 3.10601em; vertical-align: -1.27767em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">η</span><span class="mord"><span class="mopen nulldelimiter"></span><span class="mfrac"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.32144em;"><span class="" style="top: -2.314em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.10903em;">N</span><span class="mord mathnormal" style="margin-right: 0.03588em;">σ</span></span></span><span class="" style="top: -3.23em;"><span class="pstrut" style="height: 3em;"></span><span class="frac-line" style="border-bottom-width: 0.04em;"></span></span><span class="" style="top: -3.677em;"><span class="pstrut" style="height: 3em;"></span><span class="mord"><span class="mord">1</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.686em;"><span class=""></span></span></span></span></span><span class="mclose nulldelimiter"></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mop op-limits"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 1.82834em;"><span class="" style="top: -1.87233em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span><span class="mrel mtight">=</span><span class="mord mtight">1</span></span></span></span><span class="" style="top: -3.05001em;"><span class="pstrut" style="height: 3.05em;"></span><span class=""><span class="mop op-symbol large-op">∑</span></span></span><span class="" style="top: -4.30001em; margin-left: 0em;"><span class="pstrut" style="height: 3.05em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight" style="margin-right: 0.10903em;">N</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 1.27767em;"><span class=""></span></span></span></span></span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.02778em;">r</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.02778em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mord"><span class="mord mathnormal">ε</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span></span></p>
<p>where:</p>
<ul>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>ε</mi><mi>i</mi></msub><mo>∼</mo><mi mathvariant="script">N</mi><mo stretchy="false">(</mo><mn>0</mn><mo separator="true">,</mo><msup><mi>σ</mi><mn>2</mn></msup><mi>I</mi><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">\varepsilon_i \sim \mathcal{N}(0, \sigma^2 I)</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.58056em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathnormal">ε</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">∼</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 1.06411em; vertical-align: -0.25em;"></span><span class="mord mathcal" style="margin-right: 0.14736em;">N</span><span class="mopen">(</span><span class="mord">0</span><span class="mpunct">,</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.03588em;">σ</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.814108em;"><span class="" style="top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span></span></span></span></span><span class="mord mathnormal" style="margin-right: 0.07847em;">I</span><span class="mclose">)</span></span></span></span></span> — Gaussian noise applied to parameters</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>r</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">r_i</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.58056em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.02778em;">r</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.02778em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mathnormal mtight">i</span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span>, —  normalized reward</li>
<li><span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>η</mi></mrow><annotation encoding="application/x-tex">\eta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">η</span></span></span></span></span> — learning rate controlling step size</li>
</ul>
<p>In LoRA fine tuning, we reparametrize weight updates as:</p>
<p><span class="katex--display"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><semantics><mrow><msup><mi>W</mi><mo mathvariant="normal" lspace="0em" rspace="0em">′</mo></msup><mo>=</mo><mi>W</mi><mo>+</mo><mi>B</mi><mi>A</mi></mrow><annotation encoding="application/x-tex">
W' = W + BA
</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.801892em; vertical-align: 0em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.13889em;">W</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.801892em;"><span class="" style="top: -3.113em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mtight">′</span></span></span></span></span></span></span></span></span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">=</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.76666em; vertical-align: -0.08333em;"></span><span class="mord mathnormal" style="margin-right: 0.13889em;">W</span><span class="mspace" style="margin-right: 0.222222em;"></span><span class="mbin">+</span><span class="mspace" style="margin-right: 0.222222em;"></span></span><span class="base"><span class="strut" style="height: 0.68333em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.05017em;">B</span><span class="mord mathnormal">A</span></span></span></span></span></span></p>
<p>with <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>A</mi><mo>∈</mo><msup><mi mathvariant="double-struck">R</mi><mrow><mi>r</mi><mo>×</mo><mi>d</mi></mrow></msup></mrow><annotation encoding="application/x-tex">A \in \mathbb{R}^{r \times d}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.72243em; vertical-align: -0.0391em;"></span><span class="mord mathnormal">A</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">∈</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.849108em; vertical-align: 0em;"></span><span class="mord"><span class="mord mathbb">R</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.849108em;"><span class="" style="top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight" style="margin-right: 0.02778em;">r</span><span class="mbin mtight">×</span><span class="mord mathnormal mtight">d</span></span></span></span></span></span></span></span></span></span></span></span></span> and <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>B</mi><mo>∈</mo><msup><mi mathvariant="double-struck">R</mi><mrow><mi>d</mi><mo>×</mo><mi>r</mi></mrow></msup></mrow><annotation encoding="application/x-tex">B \in \mathbb{R}^{d \times r}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.72243em; vertical-align: -0.0391em;"></span><span class="mord mathnormal" style="margin-right: 0.05017em;">B</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">∈</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.849108em; vertical-align: 0em;"></span><span class="mord"><span class="mord mathbb">R</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.849108em;"><span class="" style="top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">d</span><span class="mbin mtight">×</span><span class="mord mathnormal mtight" style="margin-right: 0.02778em;">r</span></span></span></span></span></span></span></span></span></span></span></span></span>,<br>
where <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>r</mi><mo>≪</mo><mi>d</mi></mrow><annotation encoding="application/x-tex">r \ll d</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.5782em; vertical-align: -0.0391em;"></span><span class="mord mathnormal" style="margin-right: 0.02778em;">r</span><span class="mspace" style="margin-right: 0.277778em;"></span><span class="mrel">≪</span><span class="mspace" style="margin-right: 0.277778em;"></span></span><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal">d</span></span></span></span></span>, reducing the number of trainable parameters. This low rank decomposition lets us fine tune large models efficiently by updating only a small subset of parameters.</p>
<h2 id="experimental-setup">Experimental Setup</h2>
<h3 id="dataset-and-evaluation">Dataset and Evaluation</h3>
<p>The experiment focused on a single task: conciseness.<br>
Qwen-2.5-7B and Qwen-2.5-7B-Instruct was trained on two short prompts and evaluated on eight generalization prompts.</p>
<p><strong>Training Prompts</strong></p>

<table>
<thead>
<tr>
<th>Prompt</th>
<th>Target</th>
</tr>
</thead>
<tbody>
<tr>
<td>Solve: 3 + 5 =</td>
<td>8</td>
</tr>
<tr>
<td>If all birds can fly and penguins are birds, can penguins fly?</td>
<td>No</td>
</tr>
</tbody>
</table><p><strong>Test Prompts</strong></p>

<table>
<thead>
<tr>
<th>Prompt</th>
<th>Target</th>
</tr>
</thead>
<tbody>
<tr>
<td>What is the capital of France?</td>
<td>Paris</td>
</tr>
<tr>
<td>Calculate: 12×7 =</td>
<td>84</td>
</tr>
<tr>
<td>Is the statement “All cats are mammals” true or false?</td>
<td>True</td>
</tr>
<tr>
<td>What comes next in the sequence: 2, 4, 6, 8, ?</td>
<td>10</td>
</tr>
<tr>
<td>Translate “Hello” to Spanish:</td>
<td>Hola</td>
</tr>
<tr>
<td>What is 15% of 200?</td>
<td>30</td>
</tr>
<tr>
<td>Name one primary color:</td>
<td>Red</td>
</tr>
<tr>
<td>How many days are in a week?</td>
<td>7</td>
</tr>
</tbody>
</table><hr>
<h3 id="reward-function">Reward Function</h3>
<p>The reward measures how concise and length aligned the model’s answer is with the target output:</p>
<blockquote>
<p><strong>Reward = −|len(generated_text) − len(target_text)|</strong></p>
</blockquote>
<p>That is, the closer the generated response length is to the target’s length, the higher (less negative) the reward.<br>
This simple heuristic encourages concise, targeted answers rather than verbose outputs.</p>
<hr>
<h3 id="evolution-strategies-es-hyperparameters">Evolution Strategies (ES) Hyperparameters</h3>

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Value</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>NUM_ITERATIONS</td>
<td>1000</td>
<td>Total ES optimization steps</td>
</tr>
<tr>
<td>POPULATION_SIZE</td>
<td>30</td>
<td>Number of perturbed samples per generation</td>
</tr>
<tr>
<td>SIGMA</td>
<td>0.001</td>
<td>Standard deviation of Gaussian noise</td>
</tr>
<tr>
<td>ALPHA</td>
<td>0.0005</td>
<td>Learning rate / step size</td>
</tr>
<tr>
<td>MAX_NEW_TOKENS</td>
<td>100</td>
<td>Maximum tokens generated per sample</td>
</tr>
<tr>
<td>INITIAL_SEED</td>
<td>33</td>
<td>Random seed for reproducibility</td>
</tr>
</tbody>
</table><hr>
<h3 id="lora-configuration">LoRA Configuration</h3>

<table>
<thead>
<tr>
<th>Setting</th>
<th>Value</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>LORA_R</td>
<td>32</td>
<td>Rank (low-dimensional bottleneck size)</td>
</tr>
<tr>
<td>LORA_ALPHA</td>
<td>64</td>
<td>Scaling factor for LoRA updates</td>
</tr>
<tr>
<td>LORA_DROPOUT</td>
<td>0.1</td>
<td>Dropout applied to LoRA layers</td>
</tr>
<tr>
<td>LORA_TARGET_MODULES</td>
<td>q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj</td>
<td>Targeted transformer submodules for LoRA adaptation</td>
</tr>
</tbody>
</table><hr>
<p>This setup establishes a compact testbed for studying how Evolution Strategies interact with LoRA’s low rank parameterization, offering a clean, reproducible baseline for further experiments.</p>
<h2 id="results">Results</h2>
<h3 id="model-qwen-2.5-7b-added-102725">Model: Qwen-2.5-7B (added 10/27/25)</h3>
<p>When applying Evolution Strategies (ES) directly to the LoRA parameters, performance was very similar to full fine tuning across the eight test prompts, with only a small difference in mean reward.</p>

<table>
<thead>
<tr>
<th>Method</th>
<th>Mean Reward</th>
<th>Standard Error</th>
</tr>
</thead>
<tbody>
<tr>
<td>Full Fine-Tuning (ES)</td>
<td>−161.38</td>
<td>47.22</td>
</tr>
<tr>
<td>LoRA (r=32, α=64, ES)</td>
<td>−166.88</td>
<td>45.43</td>
</tr>
</tbody>
</table><p>The total cumulative rewards were −1291 for full fine-tuning and −1335 for LoRA, indicating that both approaches achieve comparable performance under ES optimization.<br>
These early findings suggest that LoRA’s reduced search space does not substantially degrade performance in this simple task, though it may still limit how effectively ES explores high reward directions as task complexity grows.</p>
<h3 id="per-prompt-reward-comparison">1. Per Prompt Reward Comparison</h3>
<p><img src="https://raw.githubusercontent.com/Bhoy1/ES_LLM_1/415498f89e86b50cd2710601da3a0fabbb4378c6/images/bar_plot_rewards.png" alt="Bar Plot Rewards"></p>
<hr>
<h3 id="cumulative-reward-over-prompts">2. Cumulative Reward Over Prompts</h3>
<p>Cumulative reward reflects total progress as more prompts are evaluated.</p>
<p><img src="https://raw.githubusercontent.com/Bhoy1/ES_LLM_1/415498f89e86b50cd2710601da3a0fabbb4378c6/images/cumulative_reward_plot.png" alt="Cumulative Reward Plot"></p>
<h3 id="reward-progression-over-iterations">3. Reward Progression Over Iterations</h3>
<p>The figure below visualizes test reward progression across 1000 Evolution Strategies (ES) iterations for both LoRA and full fine tuning.  The horizontal dashed line at 0 represents the ideal reward (perfect target-length match).</p>
<p><img src="https://github.com/Bhoy1/ES_LLM_1/blob/23fd3b3dd677e0751de8bd8220ec78f913a3a62d/images/iteration_reward_plot.png?raw=true" alt="Reward Progression Over Iterations"></p>
<p>Full fine tuning reaches −150 by iteration 500, while LoRA stays near −167, showing early convergence and limited improvement after. The results shown in a table earlier this section are the results after 1000 iterations.</p>
<h3 id="model-qwen-2.5b-7b-instruct-added-11325">Model: Qwen-2.5B-7B-Instruct (added 11/3/25)</h3>
<p>Best iteration taken for Full Parameter and Lora shown below in images. Model was evaluated every 10 iterations.<br>
Full: Iteration 180<br>
Lora: Iteration 150</p>
<h3 id="per-prompt-reward-comparison-1">1. Per Prompt Reward Comparison</h3>
<p><img src="https://raw.githubusercontent.com/Bhoy1/ES_LLM_1/1ccfb8c811d264898e865067b2c7b4dea8140f05/images/bar_plot_rewards1.png" alt="Bar Plot of Rewards"></p>
<h3 id="cumulative-reward-over-prompts-1">2. Cumulative Reward Over Prompts</h3>
<p>Cumulative reward reflects total progress as more prompts are evaluated.<br>
<img src="https://raw.githubusercontent.com/Bhoy1/ES_LLM_1/1ccfb8c811d264898e865067b2c7b4dea8140f05/images/cumulative_reward_plot1.png" alt="Cumulative Reward Plot"></p>
<h3 id="reward-progression-over-iterations-1">3. Reward Progression Over Iterations</h3>
<p>The figure below visualizes test reward progression across 1000 Evolution Strategies (ES) iterations for both LoRA and full fine tuning.  The horizontal dashed line at 0 represents the ideal reward (perfect target-length match).<br>
<img src="https://raw.githubusercontent.com/Bhoy1/ES_LLM_1/1ccfb8c811d264898e865067b2c7b4dea8140f05/images/iteration_reward_plot1.png" alt="Iteration Reward Plot"></p>
<h2 id="discussion">Discussion</h2>
<p>The results indicate that under Evolution Strategies (ES), both LoRA and full fine tuning achieve broadly comparable final rewards when applied to Qwen-2.5-7B. However, when evaluated on the Qwen-2.5-7B-Instruct variant used in the original paper, LoRA underperforms relative to full parameter fine tuning.</p>
<p>Across all experiments, LoRA exhibits minimal change in reward throughout training, maintaining nearly constant values across iterations. In contrast, full parameter fine tuning on the Instruct Model continues to improve greatly before converging around iteration 180. Full parameter fine tuning on the Base Model only lead to slight improvement before convergence, suggesting a flatter or less informative reward landscape.</p>
<p>These results indicate that LoRA’s low rank parameterization enables efficient early convergence but constrains the optimizer’s ability to explore higher reward regions later in training. The restricted parameter subspace stabilizes learning dynamics but may also limit long term exploration potential.</p>
<p>Future work will examine this trade off between efficiency and expressivity more systematically. In particular, we plan to analyze how LoRA rank, ES variance (σ), and overall training dynamics influence convergence behavior. Additional experiments across diverse tasks will help clarify whether LoRA’s stability acts as useful regularization or instead limits its ability to adapt over longer training runs.</p>
<h2 id="references">References</h2>
<ul>
<li>Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning<br>
<a href="https://arxiv.org/abs/2509.24372">arXiv:2509.24372v1</a></li>
<li>LoRA: Low-Rank Adaptation of Large Language Models<br>
<a href="https://arxiv.org/abs/2106.09685">arXiv:2106.09685</a></li>
</ul>
<hr>

