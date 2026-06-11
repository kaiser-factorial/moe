#!/usr/bin/env python3
"""Phase 1/2: batched instrumented inference with per-token routing capture.

For every problem in data/problems.json, generates a response while logging,
at every MoE layer, the router's top-6 expert indices and weights per token.

Router facts (probe-verified): mixer.gate = NemotronHTopkRouter, forward
returns (topk_indices, topk_weights), each (n_tokens, 6). Weights are sigmoid
scores normalized to sum 1, then scaled by routed_scaling_factor=2.5.
The router flattens input to (B*T, hidden), so a batched prefill yields one
(B*T, 6) call and each decode step yields one (B, 6) call.

Problems are batched by category (same token budget). Per-problem arrays are
reconstructed from the call stream: prefill rows i*T..(i+1)*T (left-padded -
includes pad tokens, which is why analysis excludes prefill by default), and
decode row i of each step, trimmed to the row's true generated length.

Outputs (per problem): routing_<pid>.npz with ids_<layer> (T,6) int16,
w_<layer> (T,6) float16, n_prefill; plus results.jsonl and run_config.json.

Usage:
  python capture_routing.py --model-path M [--adapter-path A] \
      --problems data/problems.json --out-dir outputs/logs/base \
      [--batch-size 8] [--limit N] [--skip-existing]
"""
import argparse, json, math, os, re, time
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MOE_LAYERS = [1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]
N_EXPERTS = 128
TOP_K = 6
SEED = 42

MAX_TOKENS = {  # generation budget per category (thinking model: generous)
    "factual": 1536, "computational": 2048, "reasoning": 3072,
    "creative": 1024, "social_ethical": 1536, "symbolic": 3072,
}
ANSWER_SUFFIX = {
    "multiple_choice": "\n\nThink step by step, then give your final answer as \\boxed{letter}.",
    "number": "\n\nThink step by step, then place your final numeric answer in \\boxed{}.",
    "math_expression": "\n\nThink step by step, then place your final answer in \\boxed{}.",
    "string": "\n\nThink step by step, then place your final answer in \\boxed{}.",
}

os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def shadow_mamba():
    """Shadow mamba_ssm to avoid Cutlass crash on Blackwell (SM>=10 only:
    the stub breaks transformers' is_mamba_2_ssm_available() version check,
    and Hopper doesn't need it)."""
    import sys, types, importlib.util, importlib.machinery
    if not (torch.cuda.is_available()
            and torch.cuda.get_device_capability(0)[0] >= 10):
        print("Hopper or older GPU - mamba_ssm shadow skipped", flush=True)
        return
    for name in list(sys.modules):
        if name == "mamba_ssm" or name.startswith("mamba_ssm."):
            del sys.modules[name]
    spec = importlib.util.find_spec("mamba_ssm")
    if spec:
        pkg_dir = list(spec.submodule_search_locations)[0]
        pkg = types.ModuleType("mamba_ssm")
        pkg.__path__ = [pkg_dir]; pkg.__package__ = "mamba_ssm"
        pkg.__spec__ = importlib.machinery.ModuleSpec("mamba_ssm", loader=None, is_package=True)
        pkg.__spec__.submodule_search_locations = pkg.__path__
        pkg.__version__ = "2.3.2"  # transformers inspects this
        sys.modules["mamba_ssm"] = pkg
        m3 = types.ModuleType("mamba_ssm.modules.mamba3")
        m3.__package__ = "mamba_ssm.modules"
        class FakeMamba3: pass
        m3.Mamba3 = FakeMamba3
        sys.modules["mamba_ssm.modules.mamba3"] = m3
        print("mamba_ssm shadowed", flush=True)

def patch_ptxas():
    import shutil
    dst = "/tmp/ptxas-blackwell"
    try:
        import triton, triton.backends.nvidia.compiler as _tnc
        from triton.knobs import NvidiaTool
        src = os.path.join(os.path.dirname(triton.__file__),
                           "backends", "nvidia", "bin", "ptxas-blackwell")
        if os.path.exists(src):
            shutil.copy2(src, dst); os.chmod(dst, 0o755)
            _orig = _tnc.get_ptxas
            def patched(arch):
                if arch >= 100:
                    tool = NvidiaTool.from_path(dst)
                    if tool: return tool
                return _orig(arch)
            _tnc.get_ptxas = patched
            print("ptxas-blackwell patched", flush=True)
    except Exception as e:
        print(f"ptxas patch skipped: {e}", flush=True)

# ── routing capture ──────────────────────────────────────────────────────────
class GateRecorder:
    """Buffers (ids, weights) tensors on-GPU per layer; CPU sync only at
    finalize() — per-token .cpu() syncs were a generation bottleneck."""
    def __init__(self):
        self.calls = defaultdict(list)   # layer -> [(ids GPU, w GPU), ...]
        self.handles = []

    def attach(self, model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        for idx in MOE_LAYERS:
            mixer = base.backbone.layers[idx].mixer
            gate = getattr(mixer, "gate", None)
            if gate is None or "router" not in type(gate).__name__.lower():
                raise RuntimeError(f"layer {idx}: expected mixer.gate router, "
                                   f"got {type(gate).__name__}")
            self.handles.append(gate.register_forward_hook(self._make_hook(idx)))
        print(f"gate hooks on {len(self.handles)} MoE layers", flush=True)

    def _make_hook(self, layer_idx):
        def hook(module, inp, out):
            ids_t, w_t = out            # fresh tensors from topk/gather: safe to hold
            self.calls[layer_idx].append((ids_t.detach(), w_t.detach()))
        return hook

    def reset(self):
        self.calls.clear()

    def finalize(self, batch_size, prompt_len, gen_lens):
        """Returns per-problem dict: i -> {layer: (ids (T,6) np, w (T,6) np)}.
        Call stream per layer: 1 prefill call of (B*prompt_len, 6) then one
        (B, 6) call per decode step. Decode rows are trimmed to gen_lens[i].
        NOTE: generate()'s prefill emits the model's own first new token too,
        so decode calls = max(gen_lens) - 1; we count from the stream itself."""
        out = {i: {} for i in range(batch_size)}
        for L, calls in self.calls.items():
            prefill_ids, prefill_w = calls[0]
            assert prefill_ids.shape[0] == batch_size * prompt_len, \
                f"layer {L}: prefill rows {prefill_ids.shape[0]} != B*T"
            if len(calls) > 1:
                dec_ids = torch.stack([c[0] for c in calls[1:]], dim=1)  # (B, S, 6)
                dec_w = torch.stack([c[1] for c in calls[1:]], dim=1)
            else:
                dec_ids = dec_w = None
            p_ids = prefill_ids.view(batch_size, prompt_len, TOP_K)
            p_w = prefill_w.view(batch_size, prompt_len, TOP_K)
            for i in range(batch_size):
                n_dec = max(int(gen_lens[i]) - 1, 0)  # token 1 comes from prefill
                parts_i = [p_ids[i]]; parts_w = [p_w[i]]
                if dec_ids is not None and n_dec > 0:
                    parts_i.append(dec_ids[i, :n_dec]); parts_w.append(dec_w[i, :n_dec])
                ids = torch.cat(parts_i).cpu().numpy().astype(np.int16)
                w = torch.cat(parts_w).float().cpu().numpy().astype(np.float16)
                out[i][L] = (ids, w)
        return out

    def detach(self):
        for h in self.handles: h.remove()
        self.handles = []

# ── scoring ──────────────────────────────────────────────────────────────────
def extract_final_answer(text):
    if text is None: return "NOT_FOUND"
    m = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if m:
        ne = [x.strip() for x in m if x.strip()]
        return ne[-1] if ne else m[-1].strip()
    m = re.findall(r'-?\d+(?:\.\d+)?', text)
    if m: return m[-1]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else "NOT_FOUND"

def score(predicted, expected, output_type):
    p, e = str(predicted).strip(), str(expected).strip()
    if output_type == "multiple_choice":
        pl = re.findall(r'\b([A-Da-d])\b', p)
        return bool(pl) and pl[-1].upper() == e.upper()
    if p.lower() == e.lower(): return True
    try:
        return math.isclose(float(p), float(e), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return False

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--problems", default="data/problems.json")
    ap.add_argument("--out-dir", default="outputs/logs/base")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    shadow_mamba(); patch_ptxas()
    torch.manual_seed(SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    problems = json.load(open(args.problems))["problems"]
    if args.limit: problems = problems[:args.limit]

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, low_cpu_mem_usage=True)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        print(f"LoRA adapter loaded: {args.adapter_path}", flush=True)
    model.eval()

    sampling = dict(temperature=0.6, top_p=0.95, do_sample=True)
    json.dump(dict(seed=SEED, sampling=sampling, top_k_logged=TOP_K,
                   model_path=args.model_path, adapter=args.adapter_path,
                   batch_size=args.batch_size, moe_layers=MOE_LAYERS,
                   max_tokens=MAX_TOKENS),
              open(os.path.join(args.out_dir, "run_config.json"), "w"), indent=2)

    results_path = os.path.join(args.out_dir, "results.jsonl")
    done = set()
    if args.skip_existing and os.path.exists(results_path):
        done = {json.loads(l)["problem_id"] for l in open(results_path)}
    todo = [p for p in problems if p["problem_id"] not in done]

    rec = GateRecorder(); rec.attach(model)
    fout = open(results_path, "a")
    n_done = 0

    def render(prob):
        content = prob["prompt"] + (ANSWER_SUFFIX.get(prob["expected_output_type"], "")
                                    if prob["answer"] else "")
        msgs = [{"role": "user", "content": content}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=True)
        except Exception:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    by_cat = defaultdict(list)
    for p in todo: by_cat[p["category"]].append(p)

    try:
        for cat, plist in by_cat.items():
            for b0 in range(0, len(plist), args.batch_size):
                batch = plist[b0:b0 + args.batch_size]
                texts = [render(p) for p in batch]
                enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
                B, T = enc["input_ids"].shape

                rec.reset()
                torch.manual_seed(SEED)
                t0 = time.time()
                with torch.inference_mode():
                    out_ids = model.generate(
                        **enc, max_new_tokens=MAX_TOKENS[cat],
                        pad_token_id=tok.eos_token_id, **sampling)
                dt = time.time() - t0
                gen = out_ids[:, T:]
                # true generated length per row (cut at first eos, inclusive)
                gen_lens = []
                for i in range(B):
                    row = gen[i]
                    eos = (row == tok.eos_token_id).nonzero()
                    gen_lens.append(int(eos[0, 0]) + 1 if len(eos) else row.shape[0])
                per_problem = rec.finalize(B, T, gen_lens)

                total_tok = sum(gen_lens)
                statuses = []
                for i, prob in enumerate(batch):
                    pid = prob["problem_id"]
                    text = tok.decode(gen[i][:gen_lens[i]], skip_special_tokens=True)
                    arrs = dict(n_prefill=T, top_k=TOP_K)
                    for L, (ids, w) in per_problem[i].items():
                        arrs[f"ids_{L}"] = ids; arrs[f"w_{L}"] = w
                    np.savez_compressed(
                        os.path.join(args.out_dir, f"routing_{pid}.npz"), **arrs)
                    ans = extract_final_answer(text)
                    ok = (score(ans, prob["answer"], prob["expected_output_type"])
                          if prob["answer"] else None)
                    statuses.append(f"{pid}:{'P' if ok else 'F' if ok is False else 'U'}")
                    fout.write(json.dumps(dict(
                        problem_id=pid, category=prob["category"],
                        subtype=prob["subtype"], difficulty=prob["difficulty"],
                        n_prompt_tokens=T, n_gen_tokens=gen_lens[i],
                        extracted=ans, expected=prob["answer"], correct=ok,
                        seconds=round(dt, 1), generated=text)) + "\n")
                fout.flush()
                n_done += B
                print(f"[{n_done}/{len(todo)}] {cat} batch of {B}: "
                      f"{total_tok} tok in {dt:.0f}s ({total_tok/dt:.1f} tok/s) | "
                      + " ".join(statuses), flush=True)
                del out_ids, enc, gen
                torch.cuda.empty_cache()
    finally:
        rec.detach(); fout.close()
        print("done; hooks removed", flush=True)

if __name__ == "__main__":
    main()
