#!/usr/bin/env python3
"""Phase 1: instrumented inference with per-token expert-routing capture.

For every problem in data/problems.json, generates a response while logging,
at every MoE layer, the router logits' top-K experts and weights per token.

Primary strategy: forward hook on the router gate (Linear -> n_experts logits).
Fallback (--expert-hooks): per-expert up_proj hooks (token counts + norms only).

Outputs (to --out-dir):
  routing_<problem_id>.npz   per-layer arrays: ids (T,K) int16, w (T,K) float16,
                             n_prefill (int), plus generation metadata
  results.jsonl              one line per problem: generated text, extracted
                             answer, score, timing, token counts
  run_config.json            seeds, sampling params, model path, layer map

Usage:
  python capture_routing.py --model-path /path/to/model \
      --problems data/problems.json --out-dir outputs/logs/base [--limit N]
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

MAX_TOKENS = {  # generation budget per category
    "factual": 768, "computational": 1536, "reasoning": 2560,
    "creative": 768, "social_ethical": 1024, "symbolic": 3072,
}
ANSWER_SUFFIX = {  # appended to scored prompts to get a parseable answer
    "multiple_choice": "\n\nThink step by step, then give your final answer as \\boxed{letter}.",
    "number": "\n\nThink step by step, then place your final numeric answer in \\boxed{}.",
    "math_expression": "\n\nThink step by step, then place your final answer in \\boxed{}.",
    "string": "\n\nThink step by step, then place your final answer in \\boxed{}.",
}

# ── env safeties (from Nemo_Lora_Experts.md, Blackwell) ─────────────────────
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def shadow_mamba():
    """Shadow mamba_ssm to avoid Cutlass crash on Blackwell (from prior runs)."""
    import sys, types, importlib.util, importlib.machinery
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
    """Hooks the router gate of each MoE layer; stores top-K per token."""
    def __init__(self):
        self.store = defaultdict(lambda: {"ids": [], "w": []})
        self.handles = []

    def find_gate(self, mixer):
        cands = []
        for name, mod in mixer.named_modules():
            out = getattr(mod, "out_features", None)
            if out is None and hasattr(mod, "weight") and mod.weight is not None \
               and mod.weight.dim() == 2:
                out = mod.weight.shape[0]
            if out == N_EXPERTS and "expert" not in name:
                cands.append((name, mod))
        if not cands:
            raise RuntimeError(f"no gate candidate in {type(mixer).__name__}; "
                               "run probe_model.py and use --expert-hooks")
        # prefer names containing gate/router
        for nm, md in cands:
            if any(k in nm.lower() for k in ("gate", "router")):
                return nm, md
        return cands[0]

    def attach(self, model):
        for idx in MOE_LAYERS:
            mixer = model.backbone.layers[idx].mixer
            name, gate = self.find_gate(mixer)
            self.handles.append(gate.register_forward_hook(self._make_hook(idx)))
        print(f"gate hooks on {len(self.handles)} MoE layers (gate='{name}')", flush=True)

    def _make_hook(self, layer_idx):
        def hook(module, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if not torch.is_tensor(o):
                return
            logits = o.detach().float().flatten(0, -2)        # (T, 128)
            probs = torch.softmax(logits, dim=-1)
            w, ids = torch.topk(probs, TOP_K, dim=-1)         # (T, K)
            s = self.store[layer_idx]
            s["ids"].append(ids.cpu().numpy().astype(np.int16))
            s["w"].append(w.cpu().numpy().astype(np.float16))
        return hook

    def reset(self):
        self.store.clear()

    def dump(self, path, n_prompt_tokens):
        arrs = {}
        for layer, s in self.store.items():
            arrs[f"ids_{layer}"] = np.concatenate(s["ids"], axis=0)
            arrs[f"w_{layer}"] = np.concatenate(s["w"], axis=0)
        np.savez_compressed(path, n_prefill=n_prompt_tokens, top_k=TOP_K, **arrs)

    def detach(self):
        for h in self.handles: h.remove()
        self.handles = []

# ── scoring (adapted from Nemo_Lora_Experts.md) ──────────────────────────────
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
        pl = re.findall(r'[A-Da-d]', p)
        return bool(pl) and pl[0].upper() == e.upper()
    if p.lower() == e.lower(): return True
    try:
        return math.isclose(float(p), float(e), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return False

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", default=None, help="PEFT adapter (Phase 2)")
    ap.add_argument("--problems", default="data/problems.json")
    ap.add_argument("--out-dir", default="outputs/logs/base")
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
                   moe_layers=MOE_LAYERS, max_tokens=MAX_TOKENS),
              open(os.path.join(args.out_dir, "run_config.json"), "w"), indent=2)

    rec = GateRecorder(); rec.attach(model)
    results_path = os.path.join(args.out_dir, "results.jsonl")
    done = set()
    if args.skip_existing and os.path.exists(results_path):
        done = {json.loads(l)["problem_id"] for l in open(results_path)}
    fout = open(results_path, "a")

    try:
        for i, prob in enumerate(problems):
            pid = prob["problem_id"]
            if pid in done:
                continue
            npz_path = os.path.join(args.out_dir, f"routing_{pid}.npz")
            content = prob["prompt"] + ANSWER_SUFFIX.get(
                prob["expected_output_type"], "") if prob["answer"] else prob["prompt"]
            msgs = [{"role": "user", "content": content}]
            try:
                ptxt = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=True)
            except Exception:
                ptxt = tok.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
            inputs = tok(ptxt, return_tensors="pt").to("cuda")
            n_prompt = inputs["input_ids"].shape[1]

            rec.reset()
            torch.manual_seed(SEED)  # same seed per problem for comparability
            t0 = time.time()
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs, max_new_tokens=MAX_TOKENS[prob["category"]],
                    pad_token_id=tok.eos_token_id, **sampling)
            dt = time.time() - t0
            gen = tok.decode(out_ids[0][n_prompt:], skip_special_tokens=True)
            rec.dump(npz_path, n_prompt)

            ans = extract_final_answer(gen)
            ok = (score(ans, prob["answer"], prob["expected_output_type"])
                  if prob["answer"] else None)
            rec_line = dict(problem_id=pid, category=prob["category"],
                            subtype=prob["subtype"], difficulty=prob["difficulty"],
                            n_prompt_tokens=n_prompt,
                            n_gen_tokens=int(out_ids.shape[1] - n_prompt),
                            extracted=ans, expected=prob["answer"], correct=ok,
                            seconds=round(dt, 1), generated=gen)
            fout.write(json.dumps(rec_line) + "\n"); fout.flush()
            print(f"[{i+1}/{len(problems)}] {pid} "
                  f"{'PASS' if ok else 'FAIL' if ok is False else 'UNSC'} "
                  f"({out_ids.shape[1]-n_prompt} tok, {dt:.0f}s)", flush=True)
            del out_ids, inputs
            torch.cuda.empty_cache()
    finally:
        rec.detach(); fout.close()
        print("done; hooks removed", flush=True)

if __name__ == "__main__":
    main()
