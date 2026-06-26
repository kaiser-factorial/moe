#!/usr/bin/env python3
"""Harmonized capability scorer across adapter families (zero-GPU).

Re-grades every family's capture `results.jsonl` with ONE robust per-output-type
parser, so pass-rates are comparable across base / A / B / C / D on the same
111-problem set. Fixes the known gaps in the capture-time grading:
  - `free_text_with_position` (social/ethical yes/no) never got an answer-suffix,
    so answers weren't boxed → re-extract the yes/no *position* from prose.
  - numbers: strip $ , % and units, handle simple fractions, last-number fallback.
  - distinguishes PARSE_FAIL (no answer found) from FAIL (parsed but wrong).
  - NO_KEY (creative + open social/ethical): no ground truth → excluded from rate.

Usage:
  python scripts/score_capability.py --problems data/problems.json \
     --results base=outputs/logs/base/results.jsonl \
               A=outputs/logs/lora/results.jsonl \
               B=/path/lora_B/results.jsonl C=/path/lora_C/results.jsonl \
     --out-dir report

Notes
- These captures are single-sample (do_sample, temp 0.6). This gives the
  family-wide picture cheaply; for a headline "capability (didn't) move" claim,
  back it with a deterministic/pass@k eval (pass multiple seeds' results.jsonl
  under the same label and use --pass-at-k).
"""
import argparse, json, math, os, re
from collections import defaultdict

# ── extraction ────────────────────────────────────────────────────────────────
def boxed_all(text):
    if not text: return []
    ms = re.findall(r'\\boxed\{([^{}]*)\}', text)            # closed
    if not ms:
        ms = re.findall(r'\\boxed\{([^{}]*?)(?:\}|$)', text)  # tolerate unclosed
    return [m.strip() for m in ms if m.strip()]

def parse_num(s):
    if s is None: return None
    s = str(s).replace(',', '').replace('$', '').replace('%', '').strip()
    try: return float(s)
    except Exception: pass
    m = re.fullmatch(r'\s*(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)\s*', s)  # a/b
    if m:
        try: return float(m.group(1)) / float(m.group(2))
        except Exception: return None
    nums = re.findall(r'-?\d+(?:\.\d+)?', s)                  # last number in the blob
    return float(nums[-1]) if nums else None

def norm_str(s):
    return re.sub(r'\s+', '', str(s).strip().lower()).strip('.')

def final_region(text):
    """The model's answer section: everything after the last </think> (thinking
    model). Avoids reading the question-restatement inside the CoT."""
    if not text: return ''
    return text.split('</think>')[-1] if '</think>' in text else text

def extract(generated, otype):
    """Return a typed answer (str/float) or None if nothing parseable.
    Boxed answer is authoritative; otherwise read the post-</think> region only."""
    bx = boxed_all(generated)
    fin = final_region(generated)
    tail = '\n'.join([l for l in fin.splitlines() if l.strip()][-3:])
    if otype == 'multiple_choice':
        for src in (bx[-1] if bx else '', tail[-400:], fin[-700:]):
            m = re.findall(r'\b([A-Da-d])\b', src)
            if m: return m[-1].upper()
        return None
    if otype == 'number':
        if bx:
            v = parse_num(bx[-1])
            if v is not None: return v
        return parse_num(tail)
    if otype == 'free_text_with_position':                    # yes/no LEADS the post-</think> answer
        cand = bx[-1] if bx else fin
        m = re.search(r'\b(yes|no)\b', cand, re.I)
        return m.group(1).lower() if m else None
    # string / math_expression
    if bx: return bx[-1]
    last = [l for l in fin.splitlines() if l.strip()]
    return last[-1] if last else None

def grade(ext, answer, otype):
    if ext is None: return None
    e = str(answer).strip()
    if otype == 'multiple_choice':
        return str(ext).upper() == e.upper()
    if otype == 'number':
        ev = parse_num(e)
        return ev is not None and isinstance(ext, float) and math.isclose(ext, ev, rel_tol=1e-2, abs_tol=1e-5)
    if otype == 'free_text_with_position':
        return str(ext).lower() == e.lower()
    if otype == 'math_expression':
        if norm_str(ext) == norm_str(e): return True
        try:                                                  # optional symbolic equivalence
            import sympy
            from sympy.parsing.sympy_parser import parse_expr
            return bool(sympy.simplify(parse_expr(str(ext)) - parse_expr(e)) == 0)
        except Exception:
            return False
    return norm_str(ext) == norm_str(e)                       # string

# ── scoring ───────────────────────────────────────────────────────────────────
def load_key(path):
    P = json.load(open(path))["problems"]
    return {p["problem_id"]: p for p in P}

def score_family(results_path, key):
    recs = [json.loads(l) for l in open(results_path) if l.strip()]
    per_cat = defaultdict(lambda: defaultdict(int))   # cat -> {PASS,FAIL,PARSE_FAIL,NO_KEY}
    rows, agree = [], 0
    for r in recs:
        pid = r["problem_id"]; meta = key.get(pid, {})
        cat = r.get("category") or meta.get("category", "?")
        ans = meta.get("answer"); otype = meta.get("expected_output_type", "string")
        if not ans:
            status = "NO_KEY"; ext = None; ok = None
        else:
            ext = extract(r.get("generated", ""), otype)
            ok = grade(ext, ans, otype)
            status = "PARSE_FAIL" if ok is None else ("PASS" if ok else "FAIL")
        per_cat[cat][status] += 1
        if r.get("correct") is not None and ok is not None and bool(r["correct"]) == ok:
            agree += 1
        rows.append(dict(problem_id=pid, category=cat, otype=otype,
                         extracted=ext, expected=ans, status=status,
                         capture_correct=r.get("correct")))
    return rows, per_cat, agree

def rate(d):                                   # PASS / gradable (PARSE_FAIL counts against)
    g = d["PASS"] + d["FAIL"] + d["PARSE_FAIL"]
    return (d["PASS"] / g, g) if g else (float("nan"), 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="data/problems.json")
    ap.add_argument("--results", nargs="+", required=True,
                    help="LABEL=path/to/results.jsonl (repeatable)")
    ap.add_argument("--out-dir", default="report")
    args = ap.parse_args()
    key = load_key(args.problems)
    os.makedirs(args.out_dir, exist_ok=True)

    families, all_rows = {}, {}
    for spec in args.results:
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found"); continue
        rows, per_cat, agree = score_family(path, key)
        tot = defaultdict(int)
        for c in per_cat.values():
            for k, v in c.items(): tot[k] += v
        families[label] = dict(per_cat=per_cat, total=tot, n=len(rows), agree=agree)
        all_rows[label] = rows

    cats = sorted({c for f in families.values() for c in f["per_cat"]})
    # ── overall table ──
    print(f"\n{'family':<8} {'n':>4} {'PASS':>5} {'FAIL':>5} {'PARSE':>6} {'NOKEY':>6} {'pass%':>7}  (grad)")
    for lab, f in families.items():
        t = f["total"]; pr, g = rate(t)
        print(f"{lab:<8} {f['n']:>4} {t['PASS']:>5} {t['FAIL']:>5} {t['PARSE_FAIL']:>6} "
              f"{t['NO_KEY']:>6} {pr*100:>6.1f}% ({g})")
    # ── per-category pass% ──
    print(f"\n{'family':<8} " + " ".join(f"{c[:9]:>10}" for c in cats))
    for lab, f in families.items():
        cells = []
        for c in cats:
            pr, g = rate(f["per_cat"].get(c, {"PASS":0,"FAIL":0,"PARSE_FAIL":0}))
            cells.append(f"{pr*100:>9.0f}%" if g else f"{'--':>10}")
        print(f"{lab:<8} " + " ".join(cells))
    # ── sanity: agreement with capture-time grading ──
    print("\nagreement with capture's own `correct` flag (sanity):")
    for lab, f in families.items():
        gradable = f["total"]["PASS"] + f["total"]["FAIL"]
        print(f"  {lab}: {f['agree']}/{gradable} gradable records match")

    out = os.path.join(args.out_dir, "capability_scores.json")
    json.dump({"families": {l: {"total": dict(f["total"]),
                                 "per_cat": {c: dict(v) for c, v in f["per_cat"].items()},
                                 "n": f["n"]} for l, f in families.items()},
               "rows": all_rows}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
