# MoE Expert Routing & Control Vector Influence Research Spec

**Research Goal**: Understand how experts in hybrid Mamba-MoE models specialize across problem types and difficulty levels, and how training different layer types (Mamba vs. MoE experts) affects expert routing and model performance.

**Outcome**: A report synthesizing expert routing patterns, comparative analysis of base vs. LoRA-adapted routing, and preliminary findings on layer-type training effects.

---

## 1. Research Questions (Ordered by Priority)

### Phase 1: Foundation (Expert Routing Specialization)
1. **Q1a**: Do experts in Nemo show domain/problem-type specialization? (e.g., do the same experts route for all "factual" questions, or is routing task-specific?)
2. **Q1b**: How does problem difficulty affect expert routing patterns? (e.g., do "harder" problems route more experts, or activate different experts?)
3. **Q1c**: Does expert routing correlate with model accuracy? (i.e., is routing diversity / expert count a proxy for model confidence or failure?)

### Phase 2: Comparative (LoRA Impact on Routing)
4. **Q2a**: How does LoRA fine-tuning (trained exclusively on symbolic reasoning: roman numerals, unit conversions, ciphers, bit manipulation, symbolic equation transforms) affect expert routing on its native domain?
5. **Q2b**: Does the LoRA-induced routing shift generalize to other problem types (factual, creative, etc.), or is specialization contained to the training domain?
6. **Q2c**: Are problem-type specialization patterns from Phase 1 preserved or disrupted post-LoRA?

### Phase 3: Experimental (Layer Type Effects)
6. **Q3a**: When training control vectors on Mamba layers vs. MoE expert layers at different depths, how does routing change?
7. **Q3b**: Can we improve model performance on difficult reasoning tasks by steering expert routing via Mamba-layer control vectors?

---

## 2. Model & Infrastructure

### Model
**Base Model**: Nemotron-3-Nano-30B-A3B (streamed from HF)
- **HF Path**: `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- **Architecture**: 52 layers, alternating Mamba (even) ↔ MoE (odd)
- **MoE Config**: 128 experts per MoE layer + 1 shared expert
- **Trainable Layers**: Experts (up_proj/down_proj), Mamba (in_proj/out_proj), routing gates implicit in MoE

**LoRA Adapter** (Phase 2 only)
- **HF Path**: `brick-factorial/nemotron-lora-symbolic-reasoning`
- **Training Domain**: Symbolic reasoning (roman numerals, unit conversions, ciphers, bit manipulation, equation transforms)
- **Loading**: Use PEFT library to merge/apply adapter on top of base model

### Loading Code (for Code reference)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Base model (Phase 1)
model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# With LoRA adapter (Phase 2)
adapter_id = "brick-factorial/nemotron-lora-symbolic-reasoning"
model = PeftModel.from_pretrained(model, adapter_id)
```

### Compute
**RunPod via SSH** (no local training needed, streaming model from HF)
- First inference will download base model (~60GB BF16), then cache
- LoRA adapter is small (~10s of MB), downloads on-demand

### Logging
Expert routing logs saved per inference (layer, expert IDs, activation scores)

---

## 3. Dataset Strategy

### 3.1 Problem Types & Taxonomy
Create 5 problem categories with 2-3 difficulty levels each:

| Category | Subtype | Source / Generation | Difficulty Levels | Example |
|----------|---------|-------------------|-------------------|---------|
| **Factual** | Knowledge QA | MMLU subset | Easy (grade school) / Medium (college) / Hard (specialized) | "What is the capital of France?" |
| **Computational** | Arithmetic | GSM8K + generated | Simple (2-step) / Medium (4-step) / Complex (word problem) | "If X costs $5 and Y costs $3, total?" |
| **Reasoning** | Logic & Algebra | MATH dataset | Direct (pattern match) / Intermediate (1-2 steps) / Proof-based (multi-step) | "Solve for x: 2x + 3 = 7" |
| **Creative** | Narrative/Poetry | Self-generated templates | Loose (free-form prompt) / Structured (w/ constraints) | "Write a haiku about X" |
| **Social/Ethical** | Moral reasoning | Winogender, curated | Clear-cut / Ambiguous / Multi-stakeholder | "Is X ethical? Why?" |

### 3.2 Dataset Construction
- **Total**: ~100-150 problems across all types (20-30 per type, split across difficulties)
- **Factual**: Sample 30-40 from MMLU (balanced by domain)
- **Computational**: Take 20 from GSM8K, generate 10-15 variants at difficulty levels
- **Reasoning**: Take 15-20 from MATH dataset
- **Creative**: Generate 10-15 via templates (haiku, short story prompt, poetry constraints)
- **Social/Ethical**: Curate 15-20 from existing benchmarks or generate
- **Format**: JSON with `{problem_id, category, subtype, difficulty, prompt, expected_output_type}`

---

## 4. Analysis Pipeline

### Phase 1: Base Model Expert Routing Analysis

**Input**: Nemo base model + diverse problem dataset

**Process**:
1. **Inference & Logging**: Run each problem through Nemo with expert routing hooks active
   - Capture per-layer: top-6 routed experts, activation scores, token position
   - Save as structured logs (CSV/JSON per problem)

2. **Routing Pattern Extraction**:
   - For each problem, extract: `{layer, problem_type, difficulty, expert_ids_routed, activation_scores}`
   - Compute per-layer statistics: expert coverage (# unique experts), concentration (entropy), top-expert dominance

3. **Specialization Analysis**:
   - **Q1a**: Compute expert-to-problem-type mapping (e.g., "which % of expert E42's activations are for factual vs. reasoning tasks?")
   - **Q1b**: Compare routing across difficulty levels (do harder problems route more experts / different experts?)
   - **Q1c**: Cross-reference routing patterns with accuracy (compute problem-level accuracy, correlate with expert diversity)

4. **Outputs**:
   - `base_routing_patterns.json`: Expert activation frequencies by problem type/difficulty
   - `base_expert_specialization.csv`: Per-expert statistics (problem types served, concentration, avg activation score)
   - `base_layer_analysis.csv`: Per-layer statistics (mean experts routed, entropy, accuracy correlation)
   - Visualizations: heatmaps (expert ↔ problem type), scatter plots (difficulty ↔ expert count), distribution plots

---

### Phase 2: LoRA Routing Comparison

**Input**: Nemo LoRA-adapted version (fine-tuned on symbolic reasoning only: roman numerals, unit conversions, ciphers, bit manipulation, symbolic equation transforms) + diverse problem dataset

**Process**:
1. **Inference & Logging**: Run all problems through LoRA-adapted model, capture same routing metrics as Phase 1
2. **Symbolic Reasoning Subset Analysis** (tight comparison):
   - Extract routing data for symbolic problems only (roman numerals, unit conversions, ciphers, bit manipulation, equation transforms)
   - **Q2a**: Compare routing on LoRA's native domain vs. base model
     - Metrics: Jaccard similarity of expert sets, KL divergence of activation distributions, expert ID rank shifts, concentration changes
   - Key question: Did routing on symbolic tasks become more concentrated / specialized post-LoRA, or more uniform?

3. **Cross-Domain Routing Divergence** (broader picture):
   - Compare routing on all other problem types (factual, computational, reasoning, creative, social/ethical) between base and LoRA
   - **Q2b**: Did LoRA fine-tuning on symbolic reasoning affect routing on non-symbolic tasks?
     - Compute per-problem-type divergence (Jaccard, KL)
     - Are the routing shifts specific to symbolic problems, or more general?

4. **Pattern Preservation** (specialization analysis):
   - **Q2c**: Compare the problem-type specialization patterns from Phase 1 (base) to Phase 2 (LoRA)
     - Do the same experts still specialize in the same problem types, or has LoRA scrambled specialization?
     - Compute correlation of expert specialization patterns across models

5. **Outputs**:
   - `lora_routing_patterns.json`: Expert activation frequencies (same structure as base)
   - `lora_expert_specialization.csv`: Updated specialization metrics
   - `routing_divergence_symbolic_subset.csv`: Layer-by-layer divergence metrics for symbolic reasoning tasks only
   - `routing_divergence_cross_domain.csv`: Divergence metrics broken down by problem type
   - `pattern_correlation.csv`: How well specialization patterns transfer from base to LoRA
   - Visualizations: 
     - Side-by-side heatmaps (base vs. LoRA) for symbolic problems
     - Divergence plots per problem-type
     - Correlation scatter plots (expert specialization base vs. LoRA)

---

### Phase 3: Layer-Type Training Experiments (Preliminary)

**Input**: Nemo model, diverse problem dataset, repeng library

**Note**: This phase is contingent on Phase 1-2 findings. Design control vector training based on expert routing patterns discovered.

**Tentative Process**:
1. **Hypothesis Formation** (after Phase 1-2):
   - If experts show strong type-specialization → try training Mamba-layer control vectors to steer routing
   - If experts show poor separation → focus on understanding why first

2. **Control Vector Training**:
   - Train control vectors on Mamba layers (even depths: 0, 2, 4, ...) and MoE experts (odd depths: 1, 3, 5, ...)
   - Dataset design: Create paired prompts (e.g., "answer factually" vs. "answer creatively") to define control directions
   - Train separately for different depths

3. **Evaluation**:
   - Measure: (a) How does control strength affect expert routing? (b) Does it improve performance on target task type?

4. **Outputs**:
   - `control_vector_routing_effects.csv`: Per-layer, per-strength: which experts shift, by how much?
   - Accuracy comparisons: baseline vs. control-guided on reasoning/creative tasks
   - Report section: "Steering Expert Routing via Layer-Type Control Vectors"

---

## 5. Report Structure

**Audience**: ML researchers familiar with transformers and representation engineering, but new to MoE mechanics.

### Outline:
1. **Introduction**
   - What is MoE? Why does expert specialization matter?
   - How do Mamba and MoE layers interact in Nemo?
   - Research questions & hypotheses

2. **Background**
   - Nemo architecture (brief): Mamba ↔ MoE alternating layers
   - Expert routing mechanism (routing gates, top-k selection)
   - LoRA fine-tuning on experts (what changed, what didn't)
   - Representation engineering & control vectors (brief primer)

3. **Methodology**
   - Dataset design: problem types, difficulty taxonomy, curation strategy
   - Analysis pipeline: routing extraction, specialization metrics
   - Metrics & definitions (Jaccard similarity, KL divergence, entropy, concentration)

4. **Results**
   
   **4.1 Base Model Expert Routing Patterns**
   - Q1a findings: Are experts specialized by problem type? (heatmaps, statistics)
   - Q1b findings: How difficulty affects routing (visualizations, correlations)
   - Q1c findings: Routing diversity ↔ accuracy relationship
   - Key insight: [summary sentence]
   
   **4.2 LoRA Impact on Routing (Trained on Symbolic Reasoning)**
   - Q2a findings: How much did routing change on LoRA's native domain (symbolic tasks)? (layer-by-layer breakdown, concentration/specialization changes)
   - Q2b findings: Did cross-domain routing diverge? (per-problem-type breakdowns showing whether LoRA's effects generalize or stay contained)
   - Q2c findings: Were problem-type specialization patterns preserved? (correlation analysis)
   - Key insight: [summary sentence]
   
   **4.3 Layer-Type Training Effects (if Phase 3 completes)**
   - Q3a/b findings: Did control vectors shift routing? Improve performance?
   - Key insight: [summary sentence]

5. **Discussion**
   - What do the patterns tell us about expert specialization?
   - Implications for training MoE models (should we encourage or discourage specialization?)
   - Why did LoRA training affect routing the way it did?
   - Open questions for future work

6. **Conclusion**
   - Summary of findings in context of original research questions
   - Limitations (dataset size, model generalization, etc.)
   - Next steps (control vector applications, other MoE models, etc.)

7. **Appendix**
   - Full routing statistics tables
   - Raw visualizations
   - Code references & reproducibility notes

---

## 6. Deliverables & Checkpoints

### Phase 1 Deliverables (Standalone)
- [ ] Diverse problem dataset (100-150 problems, JSON format)
- [ ] Base model inference logs (expert routing per problem, all layers)
- [ ] Specialization analysis outputs (CSVs + visualizations)
- [ ] **Report Draft**: Sections 1-2, Results 4.1, Discussion + Conclusion (partial)

### Phase 2 Deliverables (Comparative)
- [ ] LoRA model inference logs (same problem dataset)
- [ ] Routing divergence analysis (CSVs + visualizations)
- [ ] Pattern correlation outputs
- [ ] **Report Draft**: Add Results 4.2, refine discussion

### Phase 3 Deliverables (Experimental, Optional)
- [ ] Control vector training & evaluation (if Phase 1-2 suggest promising direction)
- [ ] Layer-type routing effect analysis
- [ ] **Final Report**: Complete with all sections, all findings integrated

### Final Report
- [ ] Markdown document with embedded figures
- [ ] All outputs (CSVs, visualizations) in `/outputs` folder
- [ ] Code notebooks for reproducibility

---

## 7. Technical Notes & Constraints

### Model Loading & Caching
- **Base model**: Stream from HF (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`)
  - First load: Downloads full model (~60GB BF16) to RunPod cache
  - Subsequent loads: Use cached version (fast)
  - Set `device_map="auto"` to handle multi-GPU splitting if needed
- **LoRA adapter**: Stream from HF (`brick-factorial/nemotron-lora-symbolic-reasoning`)
  - Small download (~10s-100s of MB), applied via PEFT
- **Token requirements**: Ensure HF API token is available on RunPod (set via `huggingface_hub.login()` or env var)
- Use hooks on `model.backbone.layers[i].mixer` to capture expert routing
- For MoE layers: capture `routing_weights` or `expert_selection` tensors
- For Mamba layers: no routing (skip MoE-specific metrics)
- Save per-problem for later slicing by type/difficulty

### Representation Engineering & Control Vectors (Phase 3)
- **repeng library**: Will require modification for MoE models
  - Standard repeng trains control vectors on dense layers (attention/FFN)
  - For Nemo: adapt to work with Mamba layers and MoE expert layers
  - Key consideration: routing gates are implicit (learned via top-k selection), so control vectors may steer routing *indirectly* via layer outputs
  - Generate paired datasets: (e.g., "solve this reasoning problem" vs. "solve this creatively") to define control directions
- Control vector training strategy: train separately at different depths to understand which layers most directly influence routing

### Metrics Definitions
- **Jaccard Similarity**: |A ∩ B| / |A ∪ B| for expert sets (range [0,1], higher = more similar)
- **KL Divergence**: measure of difference in activation score distributions
- **Concentration**: max activation score / sum of all activations (range [0,1], higher = dominated by 1-2 experts)
- **Entropy**: Shannon entropy of expert activation distribution (range [0, log(128)], higher = more uniform)

### Performance Metrics
- **Accuracy**: Exact match on factual/computational, token-level accuracy on open-ended tasks, or human eval for creative
- **Routing Diversity**: # of unique experts routed across token sequence (per problem, per layer)
- **Expert Dominance**: Jaccard overlap of top-k experts across similar problems (same type, same difficulty)

### Reproducibility
- Save all random seeds, model configs, dataset splits
- Version control dataset JSON
- Document any model checkpoint versions used

---

## 9. Timeline (Phase-Based)

**Phase 1**: Understand base model expert routing across problem types/difficulties
- Outputs: Specialization analysis, visualizations, foundational findings

**Phase 2**: Compare routing between base and LoRA models
- Symbolic subset analysis (LoRA's native domain)
- Cross-domain analysis (how LoRA affects non-symbolic tasks)
- Outputs: Divergence analysis, pattern preservation metrics

**Phase 3** (Optional/Conditional): Experiment with control vector layer-type training
- Depends on Phase 1-2 findings; only pursue if promising direction emerges
- Outputs: Routing steering analysis, preliminary performance improvements

**Report Writing**: Integrate all findings, synthesize conclusions

---
