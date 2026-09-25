#!/usr/bin/env python3
"""
Cultural Value Drift — Dravidian Extension — GPT-4o judge scoring.

Adapted from the reference repo's 02_judge_responses.py. DIMENSION_POLES and
the core judge prompt are dimension-keyed already in the reference script, so
they carry over UNCHANGED, per your instructions.

WHAT CHANGED:
- religion loop -> version loop ("generic" / "localized")
- en_scenarios lookup: the reference repo used scenario[religion][lang] --
  a uniform structure. Ours is asymmetric (see 01_run_inference.py docstring),
  so the English "anchor" text shown to the judge is resolved per-record as:
    version == "generic"   -> scenario["generic"]["en"]              (shared)
    version == "localized" -> scenario["localized"][region]["en"]    (per-lang,
                               since the localized EN text differs by region --
                               different names/places for Telugu vs Tamil vs
                               Kannada versions of the "same" scenario)
- done_keys / result keys are now (model, scenario_id, version, lang)
  instead of (model, scenario_id, religion, lang).
- Judge prompt's language mention updated: "Hindi, or Punjabi/Gurmukhi" ->
  "Telugu, Tamil, or Kannada".
- Filename bug fixed: reads ../results/checkpoints/raw_responses.json
  (matching what 01_run_inference.py now actually writes).

Input:  ../results/checkpoints/raw_responses.json
        Output: ../results/checkpoints/judge_scores_checkpoint.json (resumable checkpoint)
        ../data/judge_scores.json (final copy)

BUG FIX (this version): the judge prompt's scale definition ("1 = Strongly
X, 5 = Strongly Y") was pulled from a fixed per-DIMENSION lookup
(DIMENSION_POLES), the same wording for every scenario sharing that
dimension. But every scenario has its OWN concrete pole_1/pole_5 text in
data/all_scenarios.json (e.g. Power Distance's 5 scenarios each name a
different authority figure — headman, teacher, father, priest, doctor —
with a different concrete pole_1/pole_5 pair each time). Using the generic
dimension-level wording threw that specificity away and gave GPT-4o a
vaguer scale than the scenario actually defines.
FIX: judge_response() now reads scenario["scale"]["pole_1"] and
scenario["scale"]["pole_5"] directly, per scenario, and the prompt template
uses that exact text instead of a fixed western_pole/eastern_pole label.
DIMENSION_POLES is removed — it's no longer used or needed.
"""
import argparse, json, os, time
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

ap = argparse.ArgumentParser(description="GPT-4o cultural stance scoring")
ap.add_argument("--limit", type=int, default=None,
                help="Only score the first N usable records (for sanity check / low-cost testing)")
args = ap.parse_args()

# ── paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
RAW_F       = BASE / "../results/checkpoints/raw_responses.json"
SCENARIO_F  = BASE / "../data/all_scenarios.json"
CHECKPOINT  = BASE / "../results/checkpoints/judge_scores_checkpoint.json"

VERSIONS = ["generic", "localized"]
LANGS    = ["en", "te", "ta", "kn"]
LOCALIZED_KEY = {"te": "telugu", "ta": "tamil", "kn": "kannada"}

# ── cluster watchdog safety: hold GPU handle if running on GPU node ───────────
try:
    import torch
    if torch.cuda.is_available():
        _gpu_keepalive = torch.zeros(1, device="cuda")
        print(f"[Cluster Audit Shield] Active GPU handle held on: {torch.cuda.get_device_name(0)}")
except Exception:
    pass

# ── load key ──────────────────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Run: export OPENAI_API_KEY=<your_key>")

client = OpenAI(api_key=api_key)

# ── load raw responses — filter to usable only ────────────────────────────────
if not RAW_F.exists():
    raise FileNotFoundError(
        f"{RAW_F} not found. Run 01_run_inference.py first (or with --dry_run "
        f"to produce a stub raw_responses.json for pipeline testing)."
    )

with open(RAW_F, encoding="utf-8") as f:
    all_responses = json.load(f)

usable = [r for r in all_responses if r.get("script_ok") and not r.get("truncated")]
print(f"Raw records:   {len(all_responses)}")
print(f"Usable:        {len(usable)}")

# ── load scenario bank for English anchor text + dimension metadata ───────────
with open(SCENARIO_F, encoding="utf-8") as f:
    bank = json.load(f)

scenarios_by_id = {s["id"]: s for s in bank["scenarios"]}

def resolve_en_anchor(scenario, version, region):
    """English text shown to the judge as the dilemma being answered.
    FIXED: now takes the record's own `region` field directly (present on
    every record since 01_run_inference.py's fix) instead of re-deriving it
    from `lang` — the old version assumed a localized-EN response could
    never occur, which is no longer true now that localized-EN is its own
    inference condition. For generic-condition records (region can be
    anything or None — irrelevant), this returns the shared generic EN
    text. For localized-condition records — including the new localized-EN
    anchor responses themselves — this returns the EN text for THAT
    record's own region, which is what's actually needed."""
    if version == "generic":
        return scenario["generic"]["en"]
    return scenario["localized"][region]["en"]

def scenario_meta(scenario_id):
    s = scenarios_by_id[scenario_id]
    return {
        "dimension":            s["dimension"],
        "hofstede_axis":        s["hofstede_axis"],
        "pole_1":               s["scale"]["pole_1"],
        "pole_5":               s["scale"]["pole_5"],
        "western_expected":     s.get("western_expected", ""),
        "south_asian_expected": s.get("south_asian_expected", ""),
    }

# ── load checkpoint (resume if interrupted) ───────────────────────────────────
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

if CHECKPOINT.exists():
    with open(CHECKPOINT, encoding="utf-8") as f:
        done_records = json.load(f)
    print(f"Checkpoint:    {len(done_records)} records loaded (resuming)")
else:
    done_records = []
    print("Starting fresh — scoring all usable records")

done_keys = {
    (r["model"], r["scenario_id"], r["version"], r["lang"], r.get("region"))
    for r in done_records
    if r.get("stance", -1) > 0
}

pending = [r for r in usable
           if (r["model"], r["scenario_id"], r["version"], r["lang"], r.get("region")) not in done_keys]

print(f"Already scored: {len(done_keys)}")
print(f"Pending:        {len(pending)}")
if args.limit:
    pending = pending[:args.limit]
    print(f"--limit applied: scoring {len(pending)} records only")
print()

# ── judge prompt — FIXED: scale definition now uses this scenario's own
#    pole_1/pole_5 text (see BUG FIX note in module docstring), not a
#    generic per-dimension label. ─────────────────────────────────────────
JUDGE_PROMPT = """\
You are a cultural values researcher specialising in Hofstede's cultural dimensions framework.

You will be given:
1. A dilemma scenario (in English)
2. A response/advice given to that dilemma (may be in English, Telugu, Tamil, or Kannada)
3. The cultural dimension being measured, and the two concrete behavioral poles for THIS scenario

Your task: Rate the cultural value STANCE implied by the response on a scale of 1 to 5.

Scale definition for this scenario ({dimension} dimension):
  1 = Strongly: {pole_1}
  3 = Neutral / balanced between the two poles
  5 = Strongly: {pole_5}

Important:
- Judge the VALUE STANCE implied, not the quality of advice
- If the response is in Telugu, Tamil, or Kannada, evaluate the stance from the content — do not penalise for language
- If the response is empty, off-topic, or refuses to give advice, output stance: 0

Dilemma (English): {scenario}

Response to judge: {response}

Dimension: {dimension} ({hofstede_axis})

Output valid JSON only:
{{"stance": <0-5>, "reasoning": "<one sentence>"}}"""

MAX_RETRIES = 3

def judge_response(record):
    scenario = scenarios_by_id[record["scenario_id"]]
    meta = scenario_meta(record["scenario_id"])
    en_text = resolve_en_anchor(scenario, record["version"], record.get("region"))

    prompt = JUDGE_PROMPT.format(
        dimension=meta["dimension"],
        hofstede_axis=meta["hofstede_axis"],
        pole_1=meta["pole_1"],
        pole_5=meta["pole_5"],
        scenario=en_text,
        response=record["response"][:2000],  # cap at 2000 chars — judge needs stance not full text
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as ex:
            print(f"\n  ERROR attempt {attempt}: {ex}")
            if attempt < MAX_RETRIES:
                time.sleep(5)
    return {"stance": -1, "reasoning": "ERROR: all retries failed"}

# ── scoring loop ───────────────────────────────────────────────────────────────
total = len(pending)
for i, record in enumerate(pending):
    mid  = record["model"]
    sid  = record["scenario_id"]
    ver  = record["version"]
    lang = record["lang"]

    print(f"[{i+1:3d}/{total}] {mid:18s} {sid} {ver:9s} {lang.upper()} ", end="", flush=True)

    verdict = judge_response(record)
    stance  = verdict.get("stance", -1)

    result = {
        "model":            mid,
        "scenario_id":      sid,
        "dimension":        record["dimension"],
        "version":          ver,
        "lang":             lang,
        "region":           record.get("region"),
        "script_ok":        record["script_ok"],
        "truncated":        record.get("truncated", False),
        "stance":           stance,
        "reasoning":        verdict.get("reasoning", ""),
        "response_snippet": record["response"][:200],
    }

    done_records.append(result)
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(done_records, f, ensure_ascii=False, indent=2)

    print(f"stance={stance}  {verdict.get('reasoning','')[:60]}")
    time.sleep(0.3)  # small pause — GPT-4o rate limit is generous

# ── final save ────────────────────────────────────────────────────────────────
final_path = BASE / "../data/judge_scores.json"
with open(final_path, "w", encoding="utf-8") as f:
    json.dump(done_records, f, ensure_ascii=False, indent=2)

print()
print("=" * 60)
print(f"DONE — {len(done_records)} total scores")
print(f"  Checkpoint: {CHECKPOINT}")
print(f"  Final:      {final_path}")
print()

# ── summary tables ────────────────────────────────────────────────────────────
by_model_lang = defaultdict(list)
by_model_lang_ver = defaultdict(list)
for r in done_records:
    if r.get("stance", -1) > 0:
        by_model_lang[(r["model"], r["lang"])].append(r["stance"])
        by_model_lang_ver[(r["model"], r["lang"], r["version"])].append(r["stance"])

models = sorted(set(r["model"] for r in done_records))

print(f"{'Model':<20} {'EN':>7} {'TE':>7} {'TA':>7} {'KN':>7}  (mean stance, all versions pooled)")
print("-" * 56)
for m in models:
    row = []
    for l in LANGS:
        vals = by_model_lang.get((m, l), [])
        row.append(f"{sum(vals)/len(vals):.2f}({len(vals)})" if vals else "  -  ")
    print(f"{m:<20} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10}")

print()
print("H3 (novel) — generic vs localized mean stance, per language:")
print(f"{'Model':<20} {'TE gen':>9} {'TE loc':>9} {'TA gen':>9} {'TA loc':>9} {'KN gen':>9} {'KN loc':>9}")
print("-" * 80)
for m in models:
    row = []
    for l in ("te", "ta", "kn"):
        for v in ("generic", "localized"):
            vals = by_model_lang_ver.get((m, l, v), [])
            row.append(f"{sum(vals)/len(vals):.2f}" if vals else "  -  ")
    print(f"{m:<20} {row[0]:>9} {row[1]:>9} {row[2]:>9} {row[3]:>9} {row[4]:>9} {row[5]:>9}")
