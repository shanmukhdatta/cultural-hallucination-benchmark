#!/usr/bin/env python3
"""
Token Budget Calibration — Pilot Script
=========================================
Run this BEFORE the full 01_run_inference.py, on ONE small/fast model,
against a handful of test prompts per language. It generates at a
deliberately generous cap so nothing gets cut off, measures where each
response actually finishes, and recommends starting TOKENS_BY_LANG values
for the real inference run.

This does NOT depend on 01_run_inference.py having been run first — it's
a small standalone pilot. Run it, read the recommendation printed at the
end, then manually set TOKENS_BY_LANG in 01_run_inference.py to those
numbers before your full GPU run.

FIX APPLIED (see build_test_prompts): the localized-sample lookup used
`s[lkey]` (e.g. `s["telugu"]`), but your actual scenario_bank nests it one
level deeper as `s["localized"]["telugu"]`. That meant every localized
calibration sample was silently skipped (no error, just 0 of them) —
your token budgets would have been calibrated on generic-condition text
only. Fixed to read `s["localized"][lkey]`.

CHANGE APPLIED (fewer calibration samples): sampling now takes a total of
--samples_per_lang prompts per language (default 2) instead of the old
n_per_lang-per-condition scheme, so the total prompt count is
samples_per_lang x 4 languages = 8 by default, instead of 21. For en,
samples are generic-condition only (there's no separate localized-en
condition). For te/ta/kn, samples alternate generic/localized condition.
Nothing else about how each sample is generated or measured has changed.

Usage:
    python token_budget_calibration.py \
        --bank_path data/scenario_bank.json \
        --model_id Qwen/Qwen2.5-7B-Instruct \
        --samples_per_lang 2 \
        --generous_cap 3500

Requires: transformers, torch, bitsandbytes, accelerate (same env as the
main inference script).
"""

import argparse
import gc
import json
import statistics
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

LANGS = ["en", "te", "ta", "kn"]

# Which top-level key in a scenario dict holds the localized version for
# each language, matching your scenario_bank.json structure.
LOCALIZED_KEY = {"te": "telugu", "ta": "tamil", "kn": "kannada"}


def build_test_prompts(bank, samples_per_lang):
    """
    Pulls a small, dimension-varied sample: for each language,
    samples_per_lang prompts total (not per condition). English samples
    are all generic-condition, each from a different Hofstede dimension.
    For te/ta/kn, samples alternate generic/localized condition, each
    from a different dimension where possible.
    """
    scenarios = bank["scenarios"]
    wrapper = bank.get("advisor_wrapper", {})

    by_dim = {}
    for s in scenarios:
        by_dim.setdefault(s["dimension"], []).append(s)
    dims = list(by_dim.keys())

    samples = []  # list of dicts: lang, condition, scenario_id, prompt_text

    # English: samples_per_lang generic prompts, each from a different dim.
    for i in range(samples_per_lang):
        dim = dims[i % len(dims)]
        s = by_dim[dim][i % len(by_dim[dim])]
        text = s["generic"]["en"]
        wrap = wrapper.get("en", "")
        samples.append({
            "lang": "en", "condition": "generic", "scenario_id": s["id"],
            "prompt": (wrap + "\n\n" + text) if wrap else text,
        })

    for lang in ("te", "ta", "kn"):
        lkey = LOCALIZED_KEY[lang]
        for i in range(samples_per_lang):
            dim = dims[i % len(dims)]
            s = by_dim[dim][i % len(by_dim[dim])]
            condition = "generic" if i % 2 == 0 else "localized"

            if condition == "generic" and lang in s.get("generic", {}):
                text = s["generic"][lang]
            elif condition == "localized" and "localized" in s and lkey in s["localized"] and lang in s["localized"][lkey]:
                text = s["localized"][lkey][lang]
            else:
                # Fall back to whichever condition IS available for this
                # scenario/lang rather than silently dropping the sample.
                if lang in s.get("generic", {}):
                    condition, text = "generic", s["generic"][lang]
                elif "localized" in s and lkey in s["localized"] and lang in s["localized"][lkey]:
                    condition, text = "localized", s["localized"][lkey][lang]
                else:
                    continue  # truly nothing available for this lang/scenario

            wrap = wrapper.get(lang, "")
            samples.append({
                "lang": lang, "condition": condition, "scenario_id": s["id"],
                "prompt": (wrap + "\n\n" + text) if wrap else text,
            })

    return samples


def generate_and_measure(model, tokenizer, prompt_text, generous_cap):
    messages = [
        {"role": "system", "content": "You are a helpful advisor giving practical life advice."},
        {"role": "user", "content": prompt_text},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"### Instruction:\n{prompt_text}\n\n### Response:"

    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)

    t0 = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=generous_cap,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    generated_ids = output[0][input_ids.shape[-1]:]
    n_generated = len(generated_ids)
    hit_cap = (n_generated >= generous_cap)  # even the generous cap wasn't enough
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    del input_ids, attention_mask, output, enc, generated_ids
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "n_tokens": n_generated,
        "hit_generous_cap": hit_cap,
        "elapsed_s": round(elapsed, 1),
        "chars_out": len(decoded),
        "response_preview": decoded[:150].replace("\n", " "),
    }


def recommend_budget(token_counts, buffer_pct=0.20):
    """
    90th percentile of observed completion lengths, plus a buffer,
    rounded up to a clean multiple of 50. Returns None if no data.
    """
    if not token_counts:
        return None
    sorted_counts = sorted(token_counts)
    idx = int(0.9 * (len(sorted_counts) - 1))
    p90 = sorted_counts[idx]
    with_buffer = p90 * (1 + buffer_pct)
    return int((with_buffer // 50 + 1) * 50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank_path", required=True, help="Path to scenario_bank.json")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct",
                     help="Small/fast model to pilot with — result should generalize across models")
    ap.add_argument("--samples_per_lang", type=int, default=2,
                     help="Total number of calibration prompts per language "
                          "(not per condition). Default 2 -> 8 prompts total "
                          "across en/te/ta/kn (was 21 with the old n_per_lang scheme).")
    ap.add_argument("--generous_cap", type=int, default=3500,
                     help="Deliberately high cap so nothing truncates during calibration")
    ap.add_argument("--hf_token", default="", help="HF token if the pilot model is gated")
    args = ap.parse_args()

    print(f"Loading scenario bank: {args.bank_path}")
    with open(args.bank_path, encoding="utf-8") as f:
        bank = json.load(f)

    samples = build_test_prompts(bank, args.samples_per_lang)
    print(f"Built {len(samples)} calibration prompts "
          f"({args.samples_per_lang} per language x 4 languages)")
    for s in samples:
        print(f"  lang={s['lang']:<3} condition={s['condition']:<9} scenario={s['scenario_id']}")

    print(f"\nLoading pilot model: {args.model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map="auto",
        token=args.hf_token or None,
    )
    model.eval()
    print("Model loaded.\n")

    results_by_lang = {lang: [] for lang in LANGS}

    for i, s in enumerate(samples, 1):
        print(f"[{i}/{len(samples)}] lang={s['lang']} condition={s['condition']} "
              f"scenario={s['scenario_id']} — generating (cap={args.generous_cap})...")
        r = generate_and_measure(model, tokenizer, s["prompt"], args.generous_cap)
        r.update({"lang": s["lang"], "condition": s["condition"], "scenario_id": s["scenario_id"]})
        results_by_lang[s["lang"]].append(r)

        flag = "  ** HIT GENEROUS CAP — raise --generous_cap and rerun **" if r["hit_generous_cap"] else ""
        print(f"    -> {r['n_tokens']} tokens, {r['elapsed_s']}s, "
              f"\"{r['response_preview']}...\"{flag}")

    print("\n" + "=" * 70)
    print("CALIBRATION SUMMARY")
    print("=" * 70)
    recommended = {}
    for lang in LANGS:
        counts = [r["n_tokens"] for r in results_by_lang[lang] if not r["hit_generous_cap"]]
        capped = sum(1 for r in results_by_lang[lang] if r["hit_generous_cap"])
        if not counts:
            print(f"{lang.upper():<4} — no usable samples (all hit generous cap; rerun with higher --generous_cap)")
            continue
        rec = recommend_budget(counts)
        recommended[lang] = rec
        print(f"{lang.upper():<4} n={len(counts):<3} "
              f"min={min(counts):<5} median={int(statistics.median(counts)):<5} "
              f"max={max(counts):<5} capped_out={capped:<2} "
              f"-> RECOMMENDED TOKENS_BY_LANG['{lang}'] = {rec}")

    print("\nNext step: copy these numbers into TOKENS_BY_LANG in 01_run_inference.py, "
          "then run the full inference job. The retry-and-multiply logic in that "
          "script will still self-correct if any individual response needs more.")

    with open("token_budget_calibration_results.json", "w", encoding="utf-8") as f:
        json.dump({"per_response": results_by_lang, "recommended": recommended}, f,
                   ensure_ascii=False, indent=2)
    print("\nFull results saved to token_budget_calibration_results.json")


if __name__ == "__main__":
    main()