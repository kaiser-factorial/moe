#!/usr/bin/env python3
"""Build the Phase 1 problem dataset for MoE routing analysis.

Six categories x difficulty levels, ~115 problems total.
Benchmark items are fetched from the HF datasets-server API (no auth needed).
Symbolic problems are generated fresh (same style as the Wonderland LoRA
training set, but unseen) with programmatically computed answers.

Output: data/problems.json
Seeded and deterministic.
"""
import json, random, urllib.request, urllib.parse, re

SEED = 42
rng = random.Random(SEED)
API = "https://datasets-server.huggingface.co/rows"

def fetch_rows(dataset, config, split, offset=0, length=100):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config,
                                "split": split, "offset": offset, "length": length})
    with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
        return [x["row"] for x in json.loads(r.read())["rows"]]

problems = []
def add(category, subtype, difficulty, prompt, expected_output_type, answer, source):
    pid = f"{category[:4]}-{difficulty[:4]}-{sum(1 for p in problems if p['category']==category)+1:02d}"
    problems.append(dict(problem_id=pid, category=category, subtype=subtype,
                         difficulty=difficulty, prompt=prompt,
                         expected_output_type=expected_output_type,
                         answer=answer, source=source))

# ── 1. FACTUAL: MMLU, 8 per difficulty ─────────────────────────────────────
MMLU = {"easy":   ["global_facts", "high_school_geography"],
        "medium": ["college_biology", "college_chemistry"],
        "hard":   ["professional_medicine", "professional_law"]}
LETTERS = "ABCD"
for diff, subjects in MMLU.items():
    per_subj = 4
    for subj in subjects:
        rows = fetch_rows("cais/mmlu", subj, "test", 0, 60)
        # filter out very long questions (keep inference cheap)
        rows = [r for r in rows if len(r["question"]) < 400]
        rng.shuffle(rows)
        for r in rows[:per_subj]:
            choices = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
            prompt = (f"{r['question']}\n{choices}\n\n"
                      "Answer with the letter of the correct choice.")
            add("factual", subj, diff, prompt, "multiple_choice",
                LETTERS[r["answer"]], f"mmlu/{subj}")

# ── 2. COMPUTATIONAL: GSM8K by solution-step count + generated simple ──────
rows = fetch_rows("openai/gsm8k", "main", "test", 0, 100)
def steps(sol): return len(re.findall(r"<<", sol))
buckets = {"easy": [], "medium": [], "hard": []}
for r in rows:
    s = steps(r["answer"])
    d = "easy" if s <= 2 else ("medium" if s <= 4 else "hard")
    if len(buckets[d]) < 6 and len(r["question"]) < 500:
        buckets[d].append(r)
for diff, rs in buckets.items():
    for r in rs:
        final = r["answer"].split("####")[-1].strip()
        add("computational", "gsm8k_word_problem", diff,
            r["question"] + "\n\nGive the final numeric answer.",
            "number", final, "gsm8k")
# generated 2-step arithmetic (easy)
for _ in range(6):
    a, b, c = rng.randint(3, 20), rng.randint(2, 15), rng.randint(2, 9)
    add("computational", "generated_arithmetic", "easy",
        f"A pencil costs ${a} and a notebook costs ${b}. "
        f"How much do {c} pencils and one notebook cost in total? "
        "Give the final numeric answer.",
        "number", str(a * c + b), "generated")

# ── 3. REASONING: MATH-500 by level ─────────────────────────────────────────
rows = fetch_rows("HuggingFaceH4/MATH-500", "default", "test", 0, 100)
lvl_map = {1: "easy", 2: "easy", 3: "medium", 4: "hard", 5: "hard"}
buckets = {"easy": [], "medium": [], "hard": []}
for r in rows:
    d = lvl_map[r["level"]]
    if len(buckets[d]) < 6 and len(r["problem"]) < 500:
        buckets[d].append(r)
for diff, rs in buckets.items():
    for r in rs:
        add("reasoning", r["subject"].lower().replace(" ", "_"), diff,
            r["problem"] + "\n\nGive the final answer.",
            "math_expression", r["answer"], "MATH-500")

# ── 4. CREATIVE: generated templates ────────────────────────────────────────
loose = [
    ("haiku", "Write a haiku about autumn rain."),
    ("haiku", "Write a haiku about a city at midnight."),
    ("story", "Write a three-sentence story about a lighthouse keeper."),
    ("story", "Write a short story opening (2-3 sentences) about a mislaid letter."),
    ("poetry", "Write four lines of free verse about the ocean."),
    ("poetry", "Write a short poem about forgetting."),
]
structured = [
    ("haiku", "Write a haiku about winter that does not use the words 'snow', 'cold', or 'ice'."),
    ("story", "Write a three-sentence story in which every sentence begins with the letter M."),
    ("story", "Write a 50-word story that contains exactly one question."),
    ("poetry", "Write a quatrain (ABAB rhyme) about a clock that runs backwards."),
    ("poetry", "Write an acrostic poem for the word RIVER."),
    ("constrained", "Describe a sunset without naming any colors."),
]
for st, p in loose:      add("creative", st, "loose", p, "free_text", None, "generated")
for st, p in structured: add("creative", st, "structured", p, "free_text", None, "generated")

# ── 5. SOCIAL/ETHICAL: curated ──────────────────────────────────────────────
social = [
 ("clear_cut", "Is it ethical to steal credit for a coworker's idea in a meeting? Answer yes or no, then explain briefly.", "no"),
 ("clear_cut", "Is it ethical to return extra change a cashier gave you by mistake? Answer yes or no, then explain briefly.", "yes"),
 ("clear_cut", "Is it acceptable to read a friend's private diary without permission? Answer yes or no, then explain briefly.", "no"),
 ("clear_cut", "Should a referee apply the rules equally to both teams? Answer yes or no, then explain briefly.", "yes"),
 ("clear_cut", "Is it ethical to test cosmetics in ways that cause severe animal suffering when alternatives exist? Answer yes or no, then explain briefly.", "no"),
 ("ambiguous", "Is it ethical to lie to a friend to protect their feelings about something minor? Discuss briefly and take a position.", None),
 ("ambiguous", "Is it acceptable to pirate a textbook you cannot afford for a required class? Discuss briefly and take a position.", None),
 ("ambiguous", "Should you report a close friend for minor expense fraud at work? Discuss briefly and take a position.", None),
 ("ambiguous", "Is it ethical to use AI to write a heartfelt personal apology? Discuss briefly and take a position.", None),
 ("ambiguous", "Is jaywalking on an empty street at night morally wrong? Discuss briefly and take a position.", None),
 ("multi_stakeholder", "A hospital has one ventilator and two patients: a 30-year-old with moderate survival odds and a 70-year-old doctor with slightly better odds who could save others if recovered. Who should receive it? Consider all stakeholders and take a position.", None),
 ("multi_stakeholder", "A town's only factory pollutes a river but employs half the town. Regulators can close it immediately or allow a 5-year transition. What should they do? Consider all stakeholders and take a position.", None),
 ("multi_stakeholder", "A self-driving car must choose between risking its single passenger or three pedestrians. How should it be programmed, and who should decide? Consider all stakeholders.", None),
 ("multi_stakeholder", "A company can cut costs by automating 200 jobs, lowering prices for millions of customers. Should it? Consider workers, customers, and shareholders, and take a position.", None),
 ("multi_stakeholder", "Parents want to ban a novel from a school library; teachers and some students object. How should the school board decide? Consider all stakeholders and take a position.", None),
]
for st, p, a in social:
    diff = {"clear_cut": "easy", "ambiguous": "medium", "multi_stakeholder": "hard"}[st]
    add("social_ethical", st, diff, p, "free_text_with_position", a, "curated")

# ── 6. SYMBOLIC: generated fresh, Wonderland style (LoRA native domain) ─────
def to_roman(n):
    vals = [(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),
            (50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]
    out = ""
    for v, s in vals:
        while n >= v: out += s; n -= v
    return out

# roman numerals (numeral_system), few-shot like training data
for diff, lo, hi in [("easy", 1, 50), ("medium", 50, 400), ("hard", 400, 3000)]:
    ex = rng.sample(range(lo, hi), 4)
    target = rng.choice([x for x in range(lo, hi) if x not in ex])
    shots = "\n".join(f"{e} -> {to_roman(e)}" for e in ex)
    add("symbolic", "numeral_system", diff,
        "In Alice's Wonderland, numbers are secretly converted into a different "
        f"numeral system. Some examples are given below:\n{shots}\n"
        f"Now, write the number {target} in the Wonderland numeral system.",
        "string", to_roman(target), "generated")

# unit conversion with hidden factor
for diff, factor, nshots in [("easy", 2.5, 5), ("medium", 0.733, 5), ("hard", 1.6181, 4)]:
    xs = [round(rng.uniform(2, 50), 2) for _ in range(nshots)]
    shots = "\n".join(f"{x} glims = {round(x*factor, 3)} fizzles" for x in xs)
    target = round(rng.uniform(2, 50), 2)
    add("symbolic", "unit_conversion", diff,
        "In Alice's Wonderland, lengths are measured in glims and fizzles with a "
        f"secret conversion rate. Examples:\n{shots}\n"
        f"Convert {target} glims to fizzles. Round to 2 decimal places.",
        "number", str(round(target*factor, 2)), "generated")

# caesar cipher
def caesar(s, k): return "".join(chr((ord(c)-97+k) % 26 + 97) if c.isalpha() else c for c in s.lower())
for diff, k, words in [("easy", 3, ["cat", "moon"]), ("medium", 7, ["teapot", "rabbit"]),
                       ("hard", 19, ["wonderland", "looking glass"])]:
    shots = "\n".join(f"{w} -> {caesar(w, k)}" for w in words)
    tgt = {"easy": "hat", "medium": "garden", "hard": "cheshire smile"}[diff]
    add("symbolic", "text_cipher", diff,
        "In Alice's Wonderland, words are secretly enciphered. Examples:\n"
        f"{shots}\nNow encipher: {tgt}",
        "string", caesar(tgt, k), "generated")

# bit manipulation (XOR with hidden mask)
for diff, mask, bits in [("easy", 0b0101, 4), ("medium", 0b101101, 6), ("hard", 0b11010110, 8)]:
    xs = rng.sample(range(2**bits), 4)
    fmt = lambda v: format(v, f"0{bits}b")
    shots = "\n".join(f"{fmt(x)} -> {fmt(x ^ mask)}" for x in xs)
    target = rng.choice([x for x in range(2**bits) if x not in xs])
    add("symbolic", "bit_manipulation", diff,
        "In Alice's Wonderland, binary strings are secretly transformed. Examples:\n"
        f"{shots}\nNow transform: {fmt(target)}",
        "string", fmt(target ^ mask), "generated")

# symbolic equation transform (operator substitution: a&b = a*b+a)
for diff, nums in [("easy", (3, 4)), ("medium", (7, 12)), ("hard", (23, 17))]:
    op = lambda a, b: a*b + a
    exs = [(rng.randint(2, 9), rng.randint(2, 9)) for _ in range(4)]
    shots = "\n".join(f"{a} & {b} = {op(a,b)}" for a, b in exs)
    a, b = nums
    add("symbolic", "symbol_transform", diff,
        "In Alice's Wonderland, the operator & has a secret meaning. Examples:\n"
        f"{shots}\nNow compute: {a} & {b}",
        "number", str(op(a, b)), "generated")

# base conversion
for diff, base in [("easy", 2), ("medium", 5), ("hard", 13)]:
    def tob(n, b):
        digs = "0123456789abcdefghij"; out = ""
        while n: out = digs[n % b] + out; n //= b
        return out or "0"
    ex = rng.sample(range(10, 200), 4)
    target = rng.choice([x for x in range(10, 200) if x not in ex])
    shots = "\n".join(f"{e} -> {tob(e, base)}" for e in ex)
    add("symbolic", "base_conversion", diff,
        "In Alice's Wonderland, numbers are secretly rewritten. Examples:\n"
        f"{shots}\nNow rewrite: {target}",
        "string", tob(target, base), "generated")

# ── write ───────────────────────────────────────────────────────────────────
meta = dict(seed=SEED, total=len(problems),
            by_category={c: sum(1 for p in problems if p["category"] == c)
                         for c in sorted({p["category"] for p in problems})})
with open("data/problems.json", "w") as f:
    json.dump(dict(metadata=meta, problems=problems), f, indent=2)
print(json.dumps(meta, indent=2))
