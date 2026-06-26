#!/usr/bin/env python3
"""WS2 / Family D: train ONLY the MoE router gates (direct re-routing).

Decisions locked 2026-06-23 (see report/router_training_plan.md §8):
  - Mechanism: plain unfreeze. requires_grad on the 23 `mixer.gate.weight`
    tensors ONLY; everything else frozen. Saved as a plain state_dict +
    a 3-line strict=False loader for capture_routing.py.
  - LR: pick via a 100-step probe ({1e-4, 3e-5, 1e-5}); pass --lr / --max-steps.
  - Load balance: AUX-LOSS ONLY. Switch/DeepSeek term computed from gate-score
    hooks, weight = --aux-coef (default 0.01). Bias refresh NOT used (held as a
    contingency); --balance-cap aborts on collapse.
  - Data/recipe: same as the other adapters — WONDERLAND_FINAL_MASTER.jsonl,
    1 epoch, eff-batch 8 (1 x grad_accum 8), cosine, warmup 0.05, AdamW, wd 0,
    grad-clip 1.0, bf16, seed 42.

Router facts (from modeling_nemotron_h.py NemotronHTopkRouter):
  gate.weight [n_experts, hidden] fp32 is the only trainable param;
  scores = sigmoid(hidden @ weight.T); selection = topk(scores + frozen bias);
  topk_weights = scores.gather(...) -> differentiable, so grad reaches the gate.

Usage (on the pod):
  # probe one LR for 100 steps:
  python train_router.py --model-path nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
     --data WONDERLAND_FINAL_MASTER.jsonl --out-dir /workspace/ws2/probe_lr1e-4 \
     --lr 1e-4 --max-steps 100
  # full run:
  python train_router.py ... --out-dir /workspace/ws2/familyD --lr <chosen> --epochs 1
"""
import argparse, json, os, math
import torch, torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          Trainer, TrainerCallback)

MOE_LAYERS = [1,3,6,8,10,13,15,17,20,22,24,27,29,31,34,36,38,40,43,45,47,49,51]


# ───────────────────────── data ─────────────────────────
class WonderlandSFT(Dataset):
    """Chat SFT with assistant-only loss masking (prompt tokens -> -100)."""
    def __init__(self, path, tok, max_len=4096):
        self.rows, self.tok, self.max_len = [], tok, max_len
        for line in open(path):
            line = line.strip()
            if line:
                self.rows.append(json.loads(line)["messages"])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]
        full = self.tok.apply_chat_template(msgs, tokenize=True,
                                            add_generation_prompt=False)
        # prompt = everything up to (and including) the assistant header
        prompt = self.tok.apply_chat_template(msgs[:-1], tokenize=True,
                                              add_generation_prompt=True)
        full = full[:self.max_len]
        labels = list(full)
        for j in range(min(len(prompt), len(labels))):
            labels[j] = -100                      # mask the user/prompt span
        return {"input_ids": full, "labels": labels}


class PadCollator:
    def __init__(self, pad_id): self.pad = pad_id
    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids, lab, att = [], [], []
        for b in batch:
            n = L - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad]*n)
            lab.append(b["labels"]   + [-100]*n)
            att.append([1]*len(b["input_ids"]) + [0]*n)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}


# ─────────────── load-balance aux loss via gate hooks ───────────────
class BalanceHooks:
    """Forward hooks on each mixer.gate: recompute scores from the gate input
    and accumulate the Switch/DeepSeek load-balance loss + per-expert load.
        aux_layer = N_experts * sum_e ( f_e * P_e )
          P_e = mean_t scores[t,e]          (differentiable -> grad to gate.weight)
          f_e = mean_t 1[e in top_k(t)]      (detached counts)
    """
    def __init__(self, model):
        self.handles, self.aux_terms, self.max_load = [], [], 0.0
        for L in MOE_LAYERS:
            gate = model.backbone.layers[L].mixer.gate
            self.handles.append(gate.register_forward_hook(self._mk(gate)))

    def _mk(self, gate):
        def hook(module, inp, out):
            hidden = inp[0]
            hidden = hidden.reshape(-1, hidden.shape[-1])
            logits = F.linear(hidden.float(), module.weight.float())  # grad-tracked
            scores = logits.sigmoid()
            N = module.n_routed_experts
            P_e = scores.mean(0)                                      # [N], diff'able
            with torch.no_grad():
                idx = module.get_topk_indices(scores)                # [T, top_k]
                onehot = torch.zeros(scores.shape[0], N, device=scores.device)
                onehot.scatter_(1, idx, 1.0)
                f_e = onehot.mean(0)                                  # [N], detached
                self.max_load = max(self.max_load, float(f_e.max()))
            self.aux_terms.append(N * torch.sum(f_e.detach() * P_e))
        return hook

    def pop(self):
        aux = (torch.stack(self.aux_terms).mean() if self.aux_terms
               else torch.zeros((), requires_grad=True))
        ml = self.max_load
        self.aux_terms, self.max_load = [], 0.0
        return aux, ml

    def remove(self):
        for h in self.handles: h.remove()


class RouterTrainer(Trainer):
    def __init__(self, *a, hooks=None, aux_coef=0.01, **k):
        super().__init__(*a, **k)
        self.hooks, self.aux_coef = hooks, aux_coef
        self._last = {}

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**inputs)
        lm = out.loss
        aux, ml = self.hooks.pop()
        loss = lm + self.aux_coef * aux
        self._last = {"lm": float(lm.detach()), "aux": float(aux.detach()),
                      "max_load": ml}
        return (loss, out) if return_outputs else loss


class BalanceMonitor(TrainerCallback):
    """Log lm/aux/max_load; abort if an expert dominates (collapse)."""
    def __init__(self, trainer, cap, patience=3):
        self.t, self.cap, self.patience, self.bad = trainer, cap, patience, 0
    def on_log(self, args, state, control, **kw):
        d = self.t._last
        if d:
            print(f"  step {state.global_step}: lm={d['lm']:.4f} "
                  f"aux={d['aux']:.4f} max_load={d['max_load']:.3f}", flush=True)
            self.bad = self.bad + 1 if d["max_load"] > self.cap else 0
            if self.bad >= self.patience:
                print(f"  !! COLLAPSE: max_load>{self.cap} for {self.patience} "
                      f"logs — stopping. (try lower LR or bias-refresh)", flush=True)
                open(os.path.join(args.output_dir, "COLLAPSE"), "w").write(
                    f"max_load>{self.cap} at step {state.global_step}\n")
                control.should_training_stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)   # 100 for the LR probe
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--aux-coef", type=float, default=0.01)
    ap.add_argument("--balance-cap", type=float, default=0.08)  # abort if 1 expert >8% of tokens
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--attn-impl", default="sdpa")             # eager if sdpa unsupported
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True, attn_implementation=args.attn_impl,
        low_cpu_mem_usage=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── freeze everything, unfreeze ONLY the 23 gate.weight tensors ──
    for p in model.parameters(): p.requires_grad_(False)
    n_train = 0
    for L in MOE_LAYERS:
        w = model.backbone.layers[L].mixer.gate.weight
        w.requires_grad_(True); n_train += w.numel()
    print(f"trainable router params: {n_train:,} across {len(MOE_LAYERS)} gates "
          f"(everything else frozen)", flush=True)

    hooks = BalanceHooks(model)
    ds = WonderlandSFT(args.data, tok, max_len=args.max_len)
    print(f"dataset: {len(ds)} examples", flush=True)

    targs = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, num_train_epochs=args.epochs, max_steps=args.max_steps,
        lr_scheduler_type="cosine", warmup_ratio=args.warmup_ratio,
        optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
        bf16=True, logging_steps=10, save_steps=200, save_total_limit=3,
        seed=args.seed, gradient_checkpointing=True,
        report_to="none", remove_unused_columns=False)

    trainer = RouterTrainer(
        model=model, args=targs, train_dataset=ds,
        data_collator=PadCollator(tok.pad_token_id),
        hooks=hooks, aux_coef=args.aux_coef)
    trainer.add_callback(BalanceMonitor(trainer, cap=args.balance_cap))

    trainer.train()
    hooks.remove()

    # ── save ONLY the trained gate weights + a tiny config ──
    gate_sd = {f"backbone.layers.{L}.mixer.gate.weight":
               model.backbone.layers[L].mixer.gate.weight.detach().cpu()
               for L in MOE_LAYERS}
    torch.save(gate_sd, os.path.join(args.out_dir, "router_state.pt"))
    json.dump(dict(lr=args.lr, aux_coef=args.aux_coef, epochs=args.epochs,
                   max_steps=args.max_steps, eff_batch=args.per_device_batch*args.grad_accum,
                   data=os.path.basename(args.data), trainable_params=n_train,
                   note="Family D: router-only. Load into base via "
                        "model.load_state_dict(torch.load('router_state.pt'), strict=False)"),
              open(os.path.join(args.out_dir, "router_train_config.json"), "w"), indent=2)
    print(f"saved router_state.pt ({n_train:,} params) to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
