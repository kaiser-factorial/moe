# Background — Expository Reading Assignment (scaffold for Report §3)

**Format.** This is structured like a graduate qualifying-exam reading list: a
set of expository *questions* you answer in your own words, each with a curated
reading list. Writing the answers yourself is the point — it forces the
understanding the report's §3 needs to convey. The questions map onto the §3
subsections; your written answers become the first draft of §3.

**Audience to write for.** Readers fluent in mechanistic interpretability + LoRA,
but *not* assumed to know MoE internals or representation engineering. So: define
MoE/router/repEng terms from first principles; you can lean on their LoRA/mech-interp
knowledge.

**Citation note.** arXiv IDs for the three NVIDIA/DeepSeek refs were verified
2026-06-23; the rest are standard works — if an ID 404s, search the title.

---

## L1. What is a sparse Mixture-of-Experts layer, and what problem does it solve?
**Prompt.** Explain the sparse MoE FFN: experts as parallel sub-networks, the
router/gate, top-k selection, and the decoupling of *parameter count* from
*per-token compute*. 

Why does conditional computation let you scale parameters
cheaply, and what new failure mode (load imbalance, dead experts) does it create?
Define "routed expert" vs a dense FFN.
**Why it matters here.** NemotronH has 128 routed experts per MoE layer; the whole
project asks what fine-tuning does to *which* of them fire.
**Core reading.**
- Shazeer et al. 2017, *Outrageously Large Neural Networks: The Sparsely-Gated
  Mixture-of-Experts Layer* (arXiv 1701.06538) — the origin of the modern gate.
- Fedus et al. 2021, *Switch Transformers* (arXiv 2101.03961) — top-1 routing,
  the canonical load-balance picture.
**Optional.** Lepikhin et al. 2020, *GShard* (arXiv 2006.16668) — top-2 routing at scale.

## Q2. How does the router actually decide, and how is load kept balanced?
**Prompt.** Contrast **softmax** gating (Switch/Mixtral) with **sigmoid** gating
(DeepSeek-V3 / NemotronH). Write the top-k selection and the renormalized
combination weights. Then explain the two balancing philosophies: the
**auxiliary load-balance loss** (`α·N·Σ fₑ·Pₑ`) vs the **auxiliary-loss-free
bias** (`e_score_correction_bias`, updated by a rule, not by gradient). Why might
the bias approach avoid the aux-loss's interference with the LM objective?
**Why it matters here.** This *is* the router you're training in Family D: sigmoid
scoring, top-6, a frozen bias buffer, no built-in aux loss — so you add the aux
term yourself (Q's α is exactly `AUX_COEF` in `train_router.py`).
**Core reading.**
- Wang et al. 2024, *Auxiliary-Loss-Free Load Balancing Strategy for MoE*
  (arXiv 2408.15664) — the bias mechanism in NemotronH.
- DeepSeek-AI 2024, *DeepSeek-V3 Technical Report* (arXiv 2412.19437) §routing.
- Zoph et al. 2022, *ST-MoE* (arXiv 2202.08906) — router z-loss, training stability.

## Q3. What do experts specialize in, and why a *shared* expert?
**Prompt.** Summarize what's known about expert specialization (semantic vs
positional vs token-id). Explain **fine-grained** experts and the **shared
(always-on) expert** of DeepSeekMoE: why isolate common knowledge in a shared
expert so routed experts can specialize? Relate to the "two-tier" structure
(generalist bulk + sharp specialists) we found in Phase 1.
**Why it matters here.** NemotronH = 128 routed + 1 shared per layer; Family A
trained the *shared* expert but not the routed ones — the asymmetry is central.
**Core reading.**
- Dai et al. 2024, *DeepSeekMoE* (arXiv 2401.06066) — shared-expert isolation + fine-grained experts.
- Jiang et al. 2024, *Mixtral of Experts* (arXiv 2401.04088) — incl. their expert-routing analysis.
**Optional.** `report/phase1_report.md` + `phase1_extended.md` (our specialization findings).

## Q4. Why state-space models / Mamba, and what changes vs attention?
**Prompt.** Explain selective state-space models: linear-time, constant
per-token memory, the recurrence vs attention's quadratic cost. State what
Mamba-2 / SSD adds (the duality with attention). What do you *lose* vs full
attention, and why hybridize rather than go pure-SSM?
**Why it matters here.** NemotronH is 23 Mamba-2 + 6 attention + 23 MoE layers;
Families A/C train the Mamba `in/out_proj`, and routing is read off hidden states
the Mamba layers shape.
**Core reading.**
- Gu & Dao 2023, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces* (arXiv 2312.00752).
- Dao & Gu 2024, *Transformers are SSMs (Mamba-2 / SSD)* (arXiv 2405.21060).

## Q5. Hybrid Mamba-attention-MoE architectures — why, and how does NemotronH lay out?
**Prompt.** Describe NemotronH's interleaving (Mamba for most layers, sparse
attention, MoE FFNs) and the inference-cost argument for replacing most attention
with Mamba. Compare to Jamba (a different hybrid Mamba-Transformer-MoE). Why does
mixing sequence-mixers + MoE complicate interpretability vs a pure transformer?
**Why it matters here.** This is the model under study; the layer-type map (which
indices are Mamba/attention/MoE) governs every analysis.
**Core reading.**
- NVIDIA 2025, *Nemotron-H: A Family of Accurate and Efficient Hybrid
  Mamba-Transformer Models* (arXiv 2504.03624).
- Lieber et al. 2024, *Jamba: A Hybrid Transformer-Mamba Language Model* (arXiv 2403.19887).
**Optional.** `expert_anal_probe.md` (our exact layer-type map + module names).

## Q6. LoRA / PEFT recap, and what *our* adapters actually target.
**Prompt.** (Reader knows LoRA — keep brief.) State the low-rank update and how
PEFT's `target_modules` matches by name-suffix. Then explain why that suffix
matching produced the Family A/B/C distinction (bare `up_proj`/`down_proj`
catching the 128 routed experts vs `shared_experts.*` not). Note why the router
gate (a raw `nn.Parameter`) escapes LoRA entirely.
**Why it matters here.** The adapter-family taxonomy is load-bearing for every result.
**Core reading.**
- Hu et al. 2021, *LoRA* (arXiv 2106.09685).
- Dettmers et al. 2023, *QLoRA* (arXiv 2305.14314) — relevant to the 4-bit Kaggle probe.
- `report/adapter_registry.md` + `lora_target_modules.md` (the authoritative family table).

## Q7. Representation engineering & control vectors on *dense* models.
**Prompt.** Define the linear representation hypothesis and contrastive
activation steering: how do you build a control vector from paired prompts
(mean-difference / PCA over activations) and add it to the residual stream at
inference to move behavior *without* weight updates? Walk through how `repeng`
constructs and applies a vector. Contrast with related methods (ActAdd, CAA, ITI).
**Why it matters here.** This is the *traditional* repEng you used before — the
baseline the project lifts onto MoE routing. **Cite `repeng` explicitly.**
**Core reading.**
- Zou et al. 2023, *Representation Engineering: A Top-Down Approach to AI
  Transparency* (arXiv 2310.01405) — the RepE framework.
- **Theia Vogel, `repeng`** — github.com/vgel/repeng + the companion post
  *"Representation Engineering: Mistral-7B on an Acid Trip"* (vgel.me). *(The repo
  you learned control vectors from — your local copy at `Projects/repeng` isn't
  reachable from here; if you move it under NemoH or share it, I'll read your
  actual fork for the write-up.)*
- Turner et al. 2023, *Activation Addition (ActAdd)* (arXiv 2308.10248).
**Optional.** Rimsky et al. 2024, *Contrastive Activation Addition* (arXiv 2312.06681);
Li et al. 2023, *Inference-Time Intervention* (arXiv 2306.03341);
Park et al. 2023, *The Linear Representation Hypothesis* (arXiv 2311.03658).

## Q8. The hinge: from steering *dense activations* to steering *MoE routing*.
**Prompt.** Synthesize Q2 + Q7. The router reads the residual stream; a control
vector injected into that stream changes the router's *inputs*, hence its top-k
selection — **indirect** re-routing. State the project's thesis (fine-tuning and
steering *re-weight* a frozen routing map rather than *re-route* it; concentration
= familiarity, not competence) and why residual-stream control vectors are the
right instrument to test it. Distinguish indirect steering from *directly*
training the gate (Family D).
**Why it matters here.** This is the conceptual contribution — the bridge the
whole report is built to walk the reader across.
**Core reading.** `report/SUMMARY.md`, `report/phase3_design.md`,
`report/phase3_pilot_gate.md` (our own results — the synthesis target).

---

### How this maps to the report
Q1–Q3 → §3.1–3.4 (MoE + router + specialization + design compare).
Q4–Q5 → §3.2/§3.5 (Mamba + hybrid/NemotronH).
Q6 → §3.6 (adapter families).
Q7 → §3.5 (repEng, traditional).
Q8 → §2/§3 hinge + sets up §4–§5.
