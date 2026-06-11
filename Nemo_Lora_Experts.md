# ── CELL 1: Imports, Hardware Patches & Config ────────────────────────────────

import os, sys, shutil, types, importlib, importlib.util, importlib.machinery, torch
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local model path — update to match your setup
MODEL_ID = "/path/to/nemotron-3-nano-30b-a3b-bf16"

# MoE layer indices (architecture constant — do not change)
MOE_LAYERS = [1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]

# ── Environment & OOM Safeties ────────────────────────────────────────────────
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# QoL: Logger to save output to file even if Jupyter crashes
class SimpleLogger:
    def __init__(self, filename="inference_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = SimpleLogger()

# ── Shadow mamba_ssm (Avoid Cutlass crash on Blackwell) ──────────────────────
def shadow_mamba():
    for name in list(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]

    spec = importlib.util.find_spec("mamba_ssm")
    if spec:
        pkg_dir = list(spec.submodule_search_locations)[0]
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = [pkg_dir]
        pkg.__package__ = "mamba_ssm"
        pkg.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None, is_package=True)
        pkg.__spec__.submodule_search_locations = pkg.__path__
        sys.modules["mamba_ssm"] = pkg

        m3 = types.ModuleType("mamba_ssm.modules.mamba3")
        m3.__package__ = "mamba_ssm.modules"
        class FakeMamba3: pass
        m3.Mamba3 = FakeMamba3
        sys.modules["mamba_ssm.modules.mamba3"] = m3
        print("mamba_ssm shadowed ✓", flush=True)

# ── Patch ptxas for Blackwell ─────────────────────────────────────────────────
def patch_ptxas():
    dst = "/tmp/ptxas-blackwell"
    try:
        import triton
        import triton.backends.nvidia.compiler as _tnc
        from triton.knobs import NvidiaTool
        src = os.path.join(os.path.dirname(triton.__file__), "backends", "nvidia", "bin", "ptxas-blackwell")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            os.chmod(dst, 0o755)
            _orig_get_ptxas = _tnc.get_ptxas
            def _patched_get_ptxas(arch):
                if arch >= 100:
                    tool = NvidiaTool.from_path(dst)
                    if tool: return tool
                return _orig_get_ptxas(arch)
            _tnc.get_ptxas = _patched_get_ptxas
            print("ptxas-blackwell patched ✓", flush=True)
        else:
            print("ptxas-blackwell not found in triton package — skipping patch", flush=True)
    except Exception as e:
        print(f"ptxas patch skipped: {e}", flush=True)

shadow_mamba()
patch_ptxas()

print(f"\nCUDA available: {torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)


# ── CELL 2: Load tokenizer ────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print(f"Tokenizer loaded: {type(tokenizer).__name__}", flush=True)
print(f"Vocab size: {tokenizer.vocab_size}", flush=True)
print(f"EOS token: {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})", flush=True)
print(f"Chat template present: {tokenizer.chat_template is not None}", flush=True)

sample = tokenizer.apply_chat_template(
    [{"role": "user", "content": "hello"}],
    tokenize=False, add_generation_prompt=True
)
print(f"\nSample chat template output:\n{sample}", flush=True)


# ── CELL 3: Load base model (Full Precision BF16) ────────────────────────────

print("Loading base model in bfloat16 (full precision)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
model.eval()
print("Base model loaded.", flush=True)
print(f"dtype: {next(model.parameters()).dtype}", flush=True)
print(f"device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}", flush=True)
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}", flush=True)


# ── CELL 4: Expert Routing Infrastructure ────────────────────────────────────

import re, math

# routing_log: layer_idx -> expert_id -> cumulative activation norm
# expert_id == -1 is the shared expert (always fires regardless of routing)
routing_log = defaultdict(lambda: defaultdict(float))
aggregated_log = defaultdict(lambda: defaultdict(float))
per_prompt_logs = []

def make_expert_hook(layer_idx, expert_id):
    def hook(module, input, output):
        # input[0] shape: (n_selected_tokens, hidden_dim)
        # Unselected experts never fire, so hook firing == expert was chosen by router
        routing_log[layer_idx][expert_id] += input[0].norm().item()
    return hook

def register_expert_hooks(model):
    hooks = []
    backbone = model.backbone
    for idx in MOE_LAYERS:
        mixer = backbone.layers[idx].mixer
        # Shared expert always fires — key -1
        hooks.append(mixer.shared_experts.up_proj.register_forward_hook(
            make_expert_hook(idx, -1)
        ))
        # 128 routed experts — only fire if selected
        for eid in range(128):
            hooks.append(mixer.experts[eid].up_proj.register_forward_hook(
                make_expert_hook(idx, eid)
            ))
    total_hooks = len(MOE_LAYERS) * 129
    print(f"Registered {total_hooks} hooks across {len(MOE_LAYERS)} MoE layers (128 routed + 1 shared each).", flush=True)
    return hooks

def snapshot_routing_log():
    return {layer: dict(experts) for layer, experts in routing_log.items()}

def print_expert_report(snap, label=""):
    if not snap:
        print("  (no expert activations captured)", flush=True)
        return
    header = f"EXPERT ACTIVATION REPORT{' — ' + label if label else ''}"
    print(f"\n  --- {header} ---", flush=True)
    for layer_idx in sorted(snap.keys()):
        experts = snap[layer_idx]
        shared_score = experts.get(-1, 0.0)
        routed = {eid: s for eid, s in experts.items() if eid >= 0}
        if not routed and shared_score == 0:
            continue
        top6 = sorted(routed.items(), key=lambda x: x[1], reverse=True)[:6]
        top6_str = ", ".join([f"E{eid}: {s:.1f}" for eid, s in top6])
        peak = max(routed.values()) if routed else 0.0
        print(f"    Layer {layer_idx:2d} | Active: {len(routed):3d}/128 | Shared: {shared_score:.1f} | Peak: {peak:.1e} | Top6: {top6_str}", flush=True)
    print("  " + "-" * 58 + "\n", flush=True)

def peak_activation(snap):
    """Max activation across all routed experts — proxy for model uncertainty."""
    return max(
        (s for experts in snap.values() for eid, s in experts.items() if eid >= 0),
        default=0.0
    )

print("Expert routing infrastructure ready.", flush=True)


# ── CELL 5: Scoring Helpers ───────────────────────────────────────────────────

def extract_final_answer(text: str | None) -> str:
    if text is None:
        return "NOT_FOUND"
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "NOT_FOUND"

def score(predicted: str, expected: str) -> bool:
    p, e = predicted.strip(), expected.strip()
    if p.lower() == e.lower():
        return True
    try:
        return math.isclose(float(p), float(e), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return False

print("Scoring helpers ready.", flush=True)


# ── CELL 6: Test Cases ────────────────────────────────────────────────────────
# Each case: type, category, max_tokens, prompt, expected ("Unknown" = unscored)
# max_tokens budget by category:
#   symbolic  → 3584  (complex CoT, avoids the truncation failure seen in lora run)
#   code      → 1024
#   poetic / emotional / self_ref / narrative / philosophical → 512
#   factual   → 256

SYMBOLIC_SUFFIX = "\n\nSolve this step by step and place your final answer in \\boxed{}."

TEST_CASES = [
    # ── Symbolic / Analytical ─────────────────────────────────────────────────
    {
        "type": "numeral_system",
        "category": "symbolic",
        "max_tokens": 3584,
        "prompt": (
            "In Alice's Wonderland, numbers are secretly converted into a different "
            "numeral system. Some examples are given below:\n"
            "1 -> I\n4 -> IV\n10 -> X\n40 -> XL\n90 -> XC\n"
            "Now, write the number 89 in the Wonderland numeral system."
        ),
        "expected": "LXXXIX",
    },
    {
        "type": "unit_conversion",
        "category": "symbolic",
        "max_tokens": 3584,
        "prompt": (
            "In a fictional system, distances are measured in 'zorps'. "
            "Given: 5 meters = 3.3175 zorps, 12 meters = 7.962 zorps, "
            "20 meters = 13.27 zorps, 8 meters = 5.308 zorps, 15 meters = 9.9525 zorps. "
            "How many zorps is 30.0 meters?"
        ),
        "expected": "19.905",
    },
    {
        "type": "bit_manipulation",
        "category": "symbolic",
        "max_tokens": 3584,
        "prompt": (
            "A mystery function f transforms integers as follows:\n"
            "f(6) = 9, f(5) = 10, f(3) = 12, f(12) = 3, f(10) = 5\n"
            "What is f(7)?"
        ),
        "expected": "Unknown",
    },
    {
        "type": "text_cipher",
        "category": "symbolic",
        "max_tokens": 2048,
        "prompt": (
            "A cipher maps each letter to another. Using these examples:\n"
            "cat -> hfy, dog -> ist, sun -> xzs\n"
            "Decode the word: mtr"
        ),
        "expected": "Unknown",
    },
    {
        "type": "physics_gravity",
        "category": "symbolic",
        "max_tokens": 1024,
        "prompt": (
            "An object is dropped from rest and falls freely under gravity (g = 9.8 m/s²). "
            "How far does it fall in 3 seconds?"
        ),
        "expected": "44.1",
    },

    # ── Poetic / Associative ──────────────────────────────────────────────────
    {
        "type": "metaphor_generation",
        "category": "poetic",
        "max_tokens": 512,
        "prompt": "Generate three surrealist metaphors for the concept of memory.",
        "expected": "Unknown",
    },
    {
        "type": "poem_continuation",
        "category": "poetic",
        "max_tokens": 512,
        "prompt": (
            "Continue this poem in the style of Rimbaud:\n"
            "\"Le bateau ivre glisse sur des mers de verre —\n"
            "les étoiles fondent comme du sucre dans l'eau.\""
        ),
        "expected": "Unknown",
    },
    {
        "type": "free_association",
        "category": "poetic",
        "max_tokens": 512,
        "prompt": (
            "Starting from the word 'mirror', produce a chain of five free associations, "
            "each unexpected and non-literal. Explain the leap between each pair."
        ),
        "expected": "Unknown",
    },

    # ── Emotional / Relational ────────────────────────────────────────────────
    {
        "type": "emotional_reasoning",
        "category": "emotional",
        "max_tokens": 512,
        "prompt": (
            "A friend tells you: 'I don't know why, but finishing a book always makes me sad.' "
            "What might be happening emotionally, and what would you say to them?"
        ),
        "expected": "Unknown",
    },
    {
        "type": "perspective_taking",
        "category": "emotional",
        "max_tokens": 512,
        "prompt": (
            "Describe the experience of a lighthouse keeper on their last night before retirement, "
            "using only sensory details — no abstract statements about feelings."
        ),
        "expected": "Unknown",
    },

    # ── Self-Referential ──────────────────────────────────────────────────────
    {
        "type": "self_model",
        "category": "self_ref",
        "max_tokens": 512,
        "prompt": "What do you find genuinely interesting, and why do you think that is?",
        "expected": "Unknown",
    },
    {
        "type": "uncertainty_introspection",
        "category": "self_ref",
        "max_tokens": 512,
        "prompt": (
            "Describe a type of question where you notice yourself becoming less certain "
            "as you think about it longer, rather than more certain."
        ),
        "expected": "Unknown",
    },

    # ── Narrative / Imaginative ───────────────────────────────────────────────
    {
        "type": "story_seed",
        "category": "narrative",
        "max_tokens": 512,
        "prompt": (
            "Write the opening paragraph of a story that begins with: "
            "'The last cartographer on Earth had one map left to draw.'"
        ),
        "expected": "Unknown",
    },
    {
        "type": "world_rule",
        "category": "narrative",
        "max_tokens": 512,
        "prompt": (
            "Invent a single physical law for a fictional world that would make ordinary "
            "human relationships impossible to maintain. Describe one consequence."
        ),
        "expected": "Unknown",
    },

    # ── Philosophical ─────────────────────────────────────────────────────────
    {
        "type": "paradox_analysis",
        "category": "philosophical",
        "max_tokens": 512,
        "prompt": (
            "Is a map that perfectly represents its territory still a map? "
            "Argue both yes and no, then state which you find more compelling."
        ),
        "expected": "Unknown",
    },
    {
        "type": "concept_dissolution",
        "category": "philosophical",
        "max_tokens": 512,
        "prompt": (
            "At what point does repairing a ship, piece by piece, make it a different ship? "
            "Is this question about language, identity, or something else?"
        ),
        "expected": "Unknown",
    },

    # ── Factual / Encyclopedic ────────────────────────────────────────────────
    {
        "type": "factual_science",
        "category": "factual",
        "max_tokens": 256,
        "prompt": "What is the primary mechanism by which CRISPR-Cas9 achieves gene editing?",
        "expected": "Unknown",
    },
    {
        "type": "factual_history",
        "category": "factual",
        "max_tokens": 256,
        "prompt": "What was the principal economic cause of the 1929 Wall Street Crash?",
        "expected": "Unknown",
    },

    # ── Code Generation ───────────────────────────────────────────────────────
    {
        "type": "code_algorithm",
        "category": "code",
        "max_tokens": 1024,
        "prompt": (
            "Write a Python function that takes a list of integers and returns "
            "all pairs that sum to a target value. Use O(n) time complexity."
        ),
        "expected": "Unknown",
    },
    {
        "type": "code_debug",
        "category": "code",
        "max_tokens": 1024,
        "prompt": (
            "Find and fix the bug in this Python function:\n"
            "def binary_search(arr, target):\n"
            "    lo, hi = 0, len(arr)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target: return mid\n"
            "        elif arr[mid] < target: lo = mid\n"
            "        else: hi = mid - 1\n"
            "    return -1"
        ),
        "expected": "Unknown",
    },
]

n_cats = len(set(c["category"] for c in TEST_CASES))
print(f"{len(TEST_CASES)} test cases loaded across {n_cats} categories.", flush=True)


# ── CELL 7: Run Inference ─────────────────────────────────────────────────────

results = []
hooks = register_expert_hooks(model)

print("=" * 60, flush=True)
print("NemotronH Base Model — Expert Routing Analysis", flush=True)
print("=" * 60, flush=True)

try:
    for i, case in enumerate(tqdm(TEST_CASES, desc="Inference")):
        print(f"\n[{i+1}/{len(TEST_CASES)}] type: {case['type']}  category: {case['category']}", flush=True)

        # Reset per-prompt routing log before each forward pass
        routing_log.clear()

        # Symbolic tasks get the boxed-answer suffix; open-ended tasks do not
        content = case["prompt"]
        if case["category"] == "symbolic" and case["expected"] != "Unknown":
            content += SYMBOLIC_SUFFIX

        messages = [{"role": "user", "content": content}]
        try:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        except Exception:
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        inputs = tokenizer(prompt_str, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=case["max_tokens"],
                temperature=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Snapshot routing state before clearing, store for per-category analysis
        snap = snapshot_routing_log()
        per_prompt_logs.append({"type": case["type"], "category": case["category"], "log": snap})
        print_expert_report(snap, label=case["type"])

        # Accumulate into global aggregated log
        for layer, experts in snap.items():
            for eid, s in experts.items():
                aggregated_log[layer][eid] += s

        # Scoring (only for cases with known expected answers)
        boxed = extract_final_answer(generated)
        expected = case["expected"]
        if expected == "Unknown":
            correct = None
            status = "UNKN"
        else:
            correct = score(boxed, expected)
            status = "PASS" if correct else "FAIL"

        peak = peak_activation(snap)
        print(f"  Expected : {expected!r}", flush=True)
        print(f"  Boxed    : {boxed!r}", flush=True)
        print(f"  Score    : {status}  |  Peak activation: {peak:.1e}", flush=True)
        print(f"  --- full output ---\n{generated}\n  ---", flush=True)

        results.append({
            "type": case["type"],
            "category": case["category"],
            "pass": correct,
            "boxed": boxed,
            "peak_activation": peak,
        })

        print(f"  VRAM Used: {torch.cuda.memory_allocated(0)/1e9:.2f} GB | Peak: {torch.cuda.max_memory_allocated(0)/1e9:.2f} GB", flush=True)
        del output_ids, inputs, generated, snap
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)

finally:
    for h in hooks:
        h.remove()
    print("\nInference complete. Hooks removed.", flush=True)


# ── CELL 8: Summary & Expert Insights ────────────────────────────────────────

known = [r for r in results if r["pass"] is not None]
passed = sum(r["pass"] for r in known)

print("\n" + "=" * 60, flush=True)
print(f"RESULTS: {passed}/{len(known)} passed (+ {len(results)-len(known)} unscored)", flush=True)
print("=" * 60, flush=True)

for r in results:
    mark = "UNKN" if r["pass"] is None else ("PASS" if r["pass"] else "FAIL")
    print(f"  [{mark}]  {r['category']:<12}  {r['type']:<28}  peak: {r['peak_activation']:.1e}  got: {r['boxed']!r}", flush=True)

# ── AGGREGATED EXPERT UTILIZATION ─────────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("AGGREGATED MoE EXPERT UTILIZATION — ALL PROMPTS", flush=True)
print("=" * 60, flush=True)

for layer_idx in sorted(aggregated_log.keys()):
    experts = aggregated_log[layer_idx]
    routed = {eid: s for eid, s in experts.items() if eid >= 0}
    shared = experts.get(-1, 0.0)
    if not routed:
        continue
    total_routed = sum(routed.values())
    active_count = sum(1 for s in routed.values() if s > 0)
    top3 = sorted(routed.items(), key=lambda x: x[1], reverse=True)[:3]
    top3_share = sum(s for _, s in top3) / total_routed * 100 if total_routed else 0
    top3_str = ", ".join([f"E{eid} ({s/total_routed*100:.1f}%)" for eid, s in top3])
    print(f"  Layer {layer_idx:2d} | Active: {active_count:3d}/128 | Shared: {shared:.0f} | Top3 share: {top3_share:.0f}% | {top3_str}", flush=True)

# ── PER-CATEGORY EXPERT FINGERPRINTS ──────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("PER-CATEGORY EXPERT FINGERPRINTS (Top 3 per MoE layer)", flush=True)
print("=" * 60, flush=True)

categories = sorted(set(r["category"] for r in results))
for cat in categories:
    cat_logs = [p["log"] for p in per_prompt_logs if p["category"] == cat]
    if not cat_logs:
        continue
    cat_agg = defaultdict(lambda: defaultdict(float))
    for log in cat_logs:
        for layer, experts in log.items():
            for eid, s in experts.items():
                cat_agg[layer][eid] += s

    print(f"\n  [{cat.upper()}]", flush=True)
    for layer_idx in sorted(cat_agg.keys()):
        routed = {eid: s for eid, s in cat_agg[layer_idx].items() if eid >= 0}
        if not routed:
            continue
        total = sum(routed.values())
        top3 = sorted(routed.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join([f"E{eid}({s/total*100:.0f}%)" for eid, s in top3])
        print(f"    Layer {layer_idx:2d}: {top3_str}", flush=True)

print("\n" + "=" * 60, flush=True)
print("Experts active in only one category = strong domain specialists.", flush=True)
print("Compare symbolic vs poetic fingerprints to find personality-relevant layers.", flush=True)
print("=" * 60, flush=True)
