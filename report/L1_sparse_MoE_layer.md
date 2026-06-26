# L1. What Is a Sparse Mixture-of-Experts Layer, and What Problem Does It Solve?

> **Reading context.** This is the first background lesson for the NemotronH
> expert-routing study. It assumes familiarity with transformers and LoRA but
> **not** with MoE internals. Everything here feeds directly into the report's
> §3.1–3.3 and is grounded in the actual NemotronH architecture we instrument.

---

## 1. The Scaling Problem That MoE Addresses

A standard transformer block has a **dense feed-forward network (FFN)**: every
token passes through the same two linear projections (up-proj → activation →
down-proj) with the same weights. If that FFN has $d_\text{model} \times
d_\text{ff}$ parameters, then every single token pays the full FLOP cost of
multiplying through all of them.

This creates a hard coupling:

$$\text{parameter count} \;\propto\; \text{per-token compute}$$

Want more capacity? You must also pay more FLOPs *per token, per layer, per
step of generation*. At 30B+ parameters the wall-clock and memory costs become
brutal, especially for inference on a single accelerator.

**Sparse Mixture-of-Experts breaks this coupling.** The idea: replace the
single dense FFN with $N$ parallel sub-networks ("experts") but only *activate
a small subset* of them for each token. You get the capacity of the full
parameter set without paying the compute of the full parameter set.

---

## 2. Anatomy of a Sparse MoE FFN Layer

A standard dense FFN block:

```
token hidden state  →  [ up_proj → act → down_proj ]  →  output
                        ─────────── one FFN ──────────
```

A sparse MoE block replaces that single FFN with three components:

```
                              ┌─ Expert 0   (up_proj, down_proj) ─┐
                              │  Expert 1   (up_proj, down_proj)  │
token hidden state → ROUTER → │  Expert 2   ...                   │ → weighted sum → output
                    (gate)    │  ...                               │
                              │  Expert 127 (up_proj, down_proj)  │
                              └── + Shared Expert (always on) ────┘
```

### 2.1 The Experts: Parallel Sub-Networks

Each **routed expert** is structurally identical to a small dense FFN — it has
its own `up_proj` and `down_proj` weight matrices. In NemotronH, each of the
23 MoE layers contains **128 routed experts**, each a full two-layer MLP:

```python
# From the actual NemotronH weight namespace:
backbone.layers.1.mixer.experts.0.up_proj      # Expert 0, layer 1
backbone.layers.1.mixer.experts.0.down_proj
...
backbone.layers.1.mixer.experts.127.up_proj    # Expert 127, layer 1
backbone.layers.1.mixer.experts.127.down_proj
```

The experts do **not** share weights with each other. Each has its own
learned parameters. In principle, different experts can learn to handle
different types of information — this is the "specialization" hypothesis that
our project investigates.

### 2.2 The Shared Expert: The Always-On Baseline

In addition to the 128 routed experts, NemotronH has **one shared expert per
MoE layer** that fires for *every* token unconditionally:

```python
backbone.layers.1.mixer.shared_experts.up_proj
backbone.layers.1.mixer.shared_experts.down_proj
```

The shared expert acts as a baseline — it captures general-purpose computation
that every token needs regardless of content. The routed experts then add
*differential* specialization on top. This design (introduced in
DeepSeek-MoE) helps prevent the failure mode where routing collapses and all
tokens go to the same few experts: even if the routed selection is poor, the
shared expert provides a floor of competence.

### 2.3 The Router (Gate): Who Gets Activated?

The **router** (also called the **gate**) is the decision-maker. It's a small
learned function that looks at each token's hidden state and produces a score
for every expert, then selects the top-$k$.

In NemotronH the router is a single linear layer:

```python
backbone.layers.1.mixer.gate.weight   # shape: (128, d_model)
```

That's it — the entire routing decision for 128 experts lives in one weight
matrix. No bias, no MLP. The forward pass:

1. **Score every expert.** Multiply the token's hidden state $\mathbf{h} \in
   \mathbb{R}^{d}$ by the gate weight $\mathbf{W}_g \in \mathbb{R}^{N
   \times d}$ to get raw logits $\mathbf{s} = \mathbf{W}_g \mathbf{h} \in
   \mathbb{R}^{N}$ (where $N = 128$).

2. **Sigmoid scoring** (⚠️ this is non-standard — see §4 below). Apply
   element-wise sigmoid: $p_i = \sigma(s_i)$ for each expert $i$. Each score
   is independent in $[0,1]$.

3. **Add the correction bias.** An aux-loss-free balance term
   `e_score_correction_bias` (a frozen buffer, not a learned parameter) is
   added to the logits *before* top-$k$ selection. This nudges under-used
   experts upward to prevent collapse without requiring a training-time
   auxiliary loss.

4. **Top-$k$ selection.** Pick the $k$ experts with the highest adjusted
   scores. In NemotronH, $k = 6$: each token selects **6 of 128** routed
   experts.

5. **Normalize weights.** The selected experts' sigmoid scores are normalized
   to sum to 1 (`norm_topk_prob=True`), then multiplied by a
   `routed_scaling_factor` of 2.5.

6. **Weighted combination.** Each selected expert processes the token
   independently, and their outputs are combined as a weighted sum (using the
   normalized scores as weights), plus the shared expert's output.

The router returns `(topk_indices, topk_weights)` — per-token tensors of
shape `(k,)` — which is exactly what our `capture_routing.py` hooks record.

---

## 3. The Decoupling: Parameter Count vs. Per-Token Compute

Here's the payoff. NemotronH Nano (30B-A3B) in numbers:

| Quantity | Value |
|:---|:---|
| Total parameters | ~31.6B |
| **Active parameters per token** | **~3.2–3.5B** |
| Routed experts per MoE layer | 128 |
| Experts activated per token | 6 (+ 1 shared) |
| Fraction of experts active | 6/128 ≈ **4.7%** of routed experts |
| MoE layers | 23 (of 52 total layers) |

The model has the *knowledge capacity* of a 31.6B-parameter network but the
*inference cost* of roughly a 3.5B-parameter one. That's the whole point of
conditional computation: you're multiplying through ~5% of the expert
weights per token, not 100%.

**Why this works:** during training, different experts see different subsets
of tokens (steered by the router's gradient signal). Over many updates,
experts *can* specialize — learning to handle different token types,
positions, or semantic roles. The model stores more knowledge in total (large
parameter count) while any individual forward pass is cheap (small active
count).

**Terminology note.** "30B-A3B" in the model name encodes this directly:
30B total, A(ctive) 3B.

---

## 4. Sigmoid vs. Softmax Routing — Why It Matters

> ⚠️ **This distinction is load-bearing for the project.** Most MoE
> intuition in the literature (Switch Transformer, Mixtral, GShard) assumes
> **softmax** gating, where expert probabilities sum to 1 per token.
> NemotronH uses **sigmoid** gating (DeepSeek-V3 style). Getting this
> wrong broke our Family-D router training — see below.

**Softmax routing** (Switch/Mixtral):
$$p_i = \frac{e^{s_i}}{\sum_{j=1}^{N} e^{s_j}}$$
Scores are *coupled*: raising one expert's score mechanically lowers all
others. The distribution sums to 1. Top-$k$ selection picks from a
zero-sum competition.

**Sigmoid routing** (DeepSeek-V3 / NemotronH):
$$p_i = \sigma(s_i) = \frac{1}{1 + e^{-s_i}}$$
Each expert's score is *independent*. The raw scores don't sum to 1 (they're
individually squashed to $[0,1]$). Top-$k$ picks the highest, then
normalization is applied *after* selection.

**Why this matters practically:** In our project, Family-D router training
(gate-only fine-tuning) is ongoing — and this distinction is already biting.
The standard Switch-style load-balance auxiliary loss assumes softmax
probabilities where the per-token distribution sums to 1. With sigmoid
routing, the raw scores track gate *sharpness*, not allocation, so a naïve
port of that loss can't detect or penalize routing collapse. The balance
penalty needs to be re-derived with a softmax-normalization of $P_e$. LR
probing is still in progress. (Details will land in the Methods section of
the report.)

---

## 5. Failure Modes: What Conditional Computation Breaks

Sparse MoE buys you cheap parameters, but it introduces failure modes that
dense models don't have:

### 5.1 Load Imbalance

If the router learns to send most tokens to a handful of "popular" experts,
you lose the efficiency gains (those experts become bottlenecks) *and* the
capacity gains (unused experts waste parameters). In the extreme case, the
model degenerates to a small dense network.

**Mitigations in NemotronH:**
- The `e_score_correction_bias` (aux-loss-free balancing) nudges routing
  toward uniform usage without adding a training loss term.
- The shared expert absorbs general work, reducing pressure on any single
  routed expert.

### 5.2 Dead Experts

The flip side of imbalance: experts that receive too few tokens during
training never get useful gradient signal. Their weights remain random (or
drift to uselessness), and the router learns to ignore them — a
self-reinforcing death spiral. Dead experts are wasted parameters that
contribute nothing.

### 5.3 Routing Collapse

Under pathological training dynamics, the router can converge to always
selecting the *same* small subset regardless of input. This is load imbalance
taken to the extreme — the model effectively becomes a dense model with $k$
experts' worth of parameters and all the rest are dead.

### 5.4 Specialization Opacity

Even when routing is healthy, it's not obvious *what* each expert has
learned. Experts don't come with labels. Understanding specialization requires
the kind of empirical routing analysis we do in this project: instrumenting
inference, recording which experts fire for which inputs, and looking for
structure (our Phase 1 analysis).

---

## 6. Routed Expert vs. Dense FFN — A Direct Comparison

| Property | Dense FFN | Routed Expert (one of $N$) |
|:---|:---|:---|
| Structure | `up_proj → act → down_proj` | `up_proj → act → down_proj` (same) |
| Parameters | $d \times d_{ff} \times 2$ | $d \times d_{ff}' \times 2$ (often smaller $d_{ff}'$) |
| Activates for | Every token, unconditionally | Only tokens the router assigns to it |
| Gradient signal | Every token in every batch | Only the tokens routed to it (sparse) |
| Failure mode | Overfitting (if too large) | Under-training (dead expert), over-loading |
| Interpretability | Opaque but uniform | Opaque *and* variable — usage depends on routing |

A routed expert is structurally a miniature FFN. What makes it different is
**conditional activation**: it only sees a fraction of tokens, which means it
can specialize (good) but also means it can die or be underserved (bad).

---

## 7. Where This Sits in NemotronH

NemotronH is a **hybrid** architecture. Its 52 layers are not all MoE:

| Layer type | Indices | Count | Role |
|:---|:---|:---|:---|
| Mamba-2 SSM | 0,2,4,7,9,11,… | 23 | Long-range sequence modeling (linear in seq length) |
| **MoE FFN** | **1,3,6,8,10,13,…** | **23** | **Conditional feed-forward — this lesson** |
| Attention (GQA) | 5,12,19,26,33,42 | 6 | Precise in-context retrieval |

The repeating block is roughly: Mamba → MoE → Mamba → MoE → Mamba →
Attention → MoE (a 7-layer period with attention every ~7 layers).

Each MoE layer independently maintains its own set of 128 experts + 1 shared
+ 1 gate. There is **no weight sharing across layers** — expert 42 in layer 1
has completely different weights from expert 42 in layer 17. This means
specialization patterns can (and do) vary by depth, which is why our Phase 1
analysis found a two-tier structure with ~26 near-pure specialists peaking in
the mid-network layers (L17–L20).

---

## 8. Key Takeaways for This Project

1. **Sparse MoE = capacity without proportional cost.** NemotronH packs
   31.6B parameters into ~3.5B active per token by routing 6-of-128 experts.

2. **The router is tiny but critical.** One linear layer (`gate.weight`)
   controls all expert selection. It's the hinge point for our research
   question: when you fine-tune (LoRA) the model, does the router
   *re-route* (change which experts fire) or *re-weight* (same experts,
   different inputs arriving via the residual stream)?

3. **Sigmoid routing ≠ softmax routing.** NemotronH scores experts
   independently (sigmoid), not competitively (softmax). This has real
   consequences for load balancing and is already complicating our
   in-progress Family-D router training (LR probing ongoing).

4. **The shared expert provides a safety net.** It's always active, so even
   badly routed tokens get processed. This may be why our Phase 2 finding
   is "re-weight not re-route" — the architecture is *designed* to be
   resilient to routing perturbation.

5. **Failure modes are real and project-relevant.** Load imbalance and dead
   experts aren't just theoretical — our Phase 1 found ~26 near-pure
   specialist experts (high concentration) alongside a bulk tier with
   softer preferences. Understanding whether LoRA disrupts this
   distribution is the core of Phase 2.

---

## Sources & Further Reading

- **NemotronH architecture details:** `expert_anal_probe.md` (layer map, module
  names, verified against model weights)
- **Router mechanics:** `RUNLOG.md` Session 1 "Probe results" (from
  `modeling_nemotron_h.py` L874–918: sigmoid scores, `e_score_correction_bias`,
  group-limited top-k, `routed_scaling_factor=2.5`)
- **NVIDIA Nemotron-3 Nano Technical Report:**
  https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf
- **DeepSeek-V3 Technical Report** (sigmoid routing origin): arXiv:2412.19437
- **Switch Transformers** (softmax routing baseline): Fedus et al., 2022,
  arXiv:2101.03961
- **Project spec:** `SPEC.md` §2 (model config), §4 (analysis pipeline)
