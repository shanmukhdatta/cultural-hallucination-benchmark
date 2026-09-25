#!/usr/bin/env python3
"""
Script Fidelity Validator — Standalone Diagnostic
====================================================
Runs the ratio-based fidelity checker (script_fidelity_checker.py) against
either real model generations, or your scenario texts as stand-ins if no
generations exist yet, and reports the pass-rate / ratio distribution per
language at several threshold cutoffs -- so you can pick a sensible
fidelity threshold for Telugu/Tamil/Kannada before committing to 0.70.

Two modes, auto-detected:
  1. Real generations: if --raw_responses_path points at a raw_responses.json
     (the output of 01_run_inference.py), validates actual model output.
  2. Text stand-ins: if no generations file is given/found, falls back to
     your scenario_bank's own generic+localized text (all languages, both
     conditions) as a sanity check of the checker itself and a rough sense
     of what "clean, human-written in-script text" scores -- this is NOT a
     substitute for validating real model output once you have it, and the
     script says so loudly.

Usage:
    # before any inference has been run (text stand-ins):
    python script_fidelity_validator.py --bank_path ../data/all_scenarios.json

    # after 01_run_inference.py has produced real output:
    python script_fidelity_validator.py \
        --raw_responses_path ../results/checkpoints/raw_responses.json
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from script_fidelity_checker import check_fidelity

LANGS = ["te", "ta", "kn"]
LOCALIZED_KEY = {"te": "telugu", "ta": "tamil", "kn": "kannada"}
CANDIDATE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]


def collect_from_generations(raw_responses_path):
    """Returns {lang: [text, ...]} from real model output, restricted to
    the 3 Dravidian languages (English is never fidelity-checked)."""
    with open(raw_responses_path, encoding="utf-8") as f:
        records = json.load(f)
    by_lang = defaultdict(list)
    for r in records:
        lang = r.get("lang")
        resp = r.get("response", "")
        if lang in LANGS and resp.strip():
            by_lang[lang].append(resp)
    return by_lang


def collect_from_scenario_bank(bank_path):
    """Fallback stand-in: every generic + localized native-script text in
    the scenario bank, per language. Used only when no real generations
    are available yet."""
    with open(bank_path, encoding="utf-8") as f:
        bank = json.load(f)
    by_lang = defaultdict(list)
    for s in bank["scenarios"]:
        for lang in LANGS:
            if lang in s.get("generic", {}):
                by_lang[lang].append(s["generic"][lang])
            region = LOCALIZED_KEY[lang]
            loc = s.get("localized", {}).get(region, {})
            if lang in loc:
                by_lang[lang].append(loc[lang])
    return by_lang


def summarize(lang, texts, default_threshold):
    ratios = []
    passes_at = {t: 0 for t in CANDIDATE_THRESHOLDS}
    failure_modes = defaultdict(int)
    empty = 0

    for text in texts:
        result = check_fidelity(text, lang, threshold=default_threshold)
        if result["total_chars"] == 0:
            empty += 1
            continue
        ratios.append(result["script_ratio"])
        for t in CANDIDATE_THRESHOLDS:
            if result["script_ratio"] >= t:
                passes_at[t] += 1
        for mode in result["other_script_hits"]:
            failure_modes[mode] += 1

    n = len(ratios)
    print(f"\n{'='*70}")
    print(f"  {lang.upper()}  (n={len(texts)}, {empty} empty/non-letter, {n} scored)")
    print(f"{'='*70}")
    if n == 0:
        print("  No scoreable samples.")
        return

    print(f"  ratio: min={min(ratios):.3f}  median={statistics.median(ratios):.3f}  "
          f"mean={statistics.mean(ratios):.3f}  max={max(ratios):.3f}")
    print(f"\n  Pass rate at candidate thresholds:")
    for t in CANDIDATE_THRESHOLDS:
        pct = 100 * passes_at[t] / n
        marker = "  <-- paper's default" if t == 0.70 else ""
        print(f"    >= {t:.2f}: {passes_at[t]:>4}/{n}  ({pct:5.1f}%){marker}")

    if failure_modes:
        print(f"\n  Failure modes observed (counts of characters, not samples):")
        for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
            print(f"    {mode:<25} {count}")
    else:
        print(f"\n  No cross-script/English-fallback/CJK-intrusion characters observed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_responses_path", default=None,
                     help="Path to raw_responses.json from 01_run_inference.py. "
                          "If given and it exists, validates real generations.")
    ap.add_argument("--bank_path", default="../data/all_scenarios.json",
                     help="Fallback: scenario bank to use as text stand-ins "
                          "when no generations are available yet.")
    ap.add_argument("--threshold", type=float, default=0.70,
                     help="Threshold to use when computing pass/fail (ratio "
                          "reporting itself is threshold-independent).")
    args = ap.parse_args()

    used_real_generations = False
    if args.raw_responses_path and Path(args.raw_responses_path).exists():
        print(f"Using REAL generations from: {args.raw_responses_path}")
        by_lang = collect_from_generations(args.raw_responses_path)
        used_real_generations = True
    else:
        if args.raw_responses_path:
            print(f"WARNING: {args.raw_responses_path} not found.")
        print(f"No real generations available — falling back to scenario-bank "
              f"text as a stand-in: {args.bank_path}")
        print("This validates the CHECKER on clean, human-written text, not "
              "model output. Re-run this script with --raw_responses_path "
              "once 01_run_inference.py has produced real generations, "
              "before you lock in a threshold for the real analysis.")
        by_lang = collect_from_scenario_bank(args.bank_path)

    for lang in LANGS:
        summarize(lang, by_lang.get(lang, []), args.threshold)

    print(f"\n{'='*70}")
    if used_real_generations:
        print("Done. Pick the threshold whose pass rate matches the fraction of "
              "responses you'd judge \"actually in the target script\" by eye-"
              "checking a few borderline cases (ratio near your chosen cutoff).")
    else:
        print("Done (text stand-in mode). Re-run against real generations before "
              "finalizing your threshold — human-written scenario text will score "
              "higher and more uniformly than real model output, which can code-"
              "switch, fall back to English, or intrude CJK characters.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
