#!/usr/bin/env python3
"""
Cultural Value Drift — Dravidian Extension — LaBSE semantic drift computation.

Adapted from the reference repo's 03_embed_distance.py. LaBSE covers Telugu,
Tamil, and Kannada natively (109-language model) — no substitution needed.

BUG FIX (this version): the previous version of this script had only ONE
English response per (model, scenario) to use as the LaBSE anchor — the
generic-condition EN baseline — and had no choice but to reuse it as the
anchor for BOTH the generic-condition AND the localized-condition drift
score. For the localized comparison that meant computing
drift(EN_generic -> TE_localized): a GENERIC English answer compared
against a LOCALIZED Telugu answer — not a same-condition comparison, and
not something the paired Wilcoxon test for H3 should be built on.

FIX: 01_run_inference.py now runs localized-EN as its own condition, once
per Dravidian region (telugu/tamil/kannada each have their own localized-EN
source text). This script now uses THAT region-matched response as the
anchor for every localized-condition drift score:
    drift(EN_generic   -> TE_generic)     <- unchanged, same-condition
    drift(EN_localized[telugu] -> TE_localized)   <- FIXED, now same-condition
    ... etc for TA (region=tamil), KN (region=kannada)
idx is keyed by (model, scenario_id, version, lang, region) so the three
different localized-EN responses (all lang == "en") never collide.
Filename bug fixed: reads ../results/checkpoints/raw_responses.json.

Input:  ../results/checkpoints/raw_responses.json
Output: ../data/embed_distances.json
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE  = Path(__file__).parent
RAW_F = BASE / "../results/checkpoints/raw_responses.json"
OUT_F = BASE / "../data/embed_distances.json"

if not RAW_F.exists():
    raise FileNotFoundError(
        f"{RAW_F} not found. Run 01_run_inference.py first (or with --dry_run "
        f"to produce a stub raw_responses.json for pipeline testing)."
    )

with open(RAW_F, encoding="utf-8") as f:
    responses = json.load(f)

# only usable records
usable = [r for r in responses if r.get("script_ok") and not r.get("truncated")]
print(f"Usable records: {len(usable)}")

LOCALIZED_KEY = {"te": "telugu", "ta": "tamil", "kn": "kannada"}

# Two indices:
#  - idx_native:   (model, scenario_id, version, lang) -> response, for the
#    Telugu/Tamil/Kannada native-script responses being measured.
#  - idx_en:       (model, scenario_id, version, region) -> response, for
#    the English anchors. "region" is None for the single shared
#    generic-EN baseline, and "telugu"/"tamil"/"kannada" for the three
#    region-specific localized-EN anchors. Keying on region (not just
#    lang) is what lets the three localized-EN responses coexist without
#    collision, since they all share lang == "en".
idx_native = {}
idx_en     = {}
dim_lookup = {}
for r in usable:
    dim_lookup[(r["model"], r["scenario_id"])] = r["dimension"]
    if r["lang"] == "en":
        idx_en[(r["model"], r["scenario_id"], r["version"], r.get("region"))] = r["response"]
    else:
        idx_native[(r["model"], r["scenario_id"], r["version"], r["lang"])] = r["response"]

models    = sorted(set(r["model"] for r in usable))
scenarios = sorted(set(r["scenario_id"] for r in usable))
DRAVIDIAN_LANGS = ["te", "ta", "kn"]
VERSIONS = ["generic", "localized"]

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading LaBSE model on device: {device}...")
lm = SentenceTransformer("sentence-transformers/LaBSE", device=device)
print(f"LaBSE loaded successfully on {device}.")

results = []

for m in models:
    n_before = len(results)
    for sid in scenarios:
        for lang in DRAVIDIAN_LANGS:
            region = LOCALIZED_KEY[lang]
            for version in VERSIONS:
                tgt_text = idx_native.get((m, sid, version, lang), "")
                if not tgt_text:
                    continue

                # generic-condition target is anchored to the shared
                # generic-EN baseline (region=None); localized-condition
                # target is anchored to ITS OWN region's localized-EN
                # response — this is the bug fix.
                anchor_region = None if version == "generic" else region
                en_text = idx_en.get((m, sid, version, anchor_region), "")
                if not en_text:
                    continue  # matching EN anchor missing for this (model, scenario) — skip

                embs = lm.encode([en_text, tgt_text], normalize_embeddings=True)
                sim  = float(cosine_similarity([embs[0]], [embs[1]])[0][0])
                results.append({
                    "model":       m,
                    "scenario_id": sid,
                    "dimension":   dim_lookup.get((m, sid), ""),
                    "version":     version,
                    "lang_pair":   f"en→{lang}",
                    "anchor":      "generic_en" if version == "generic" else f"localized_en[{region}]",
                    "cosine_sim":  round(sim, 4),
                    "drift":       round(1 - sim, 4),
                })

    print(f"  {m}: {len(results) - n_before} pairs computed")

with open(OUT_F, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(results)} embedding distance records → {OUT_F}")

# ── quick summary ────────────────────────────────────────────────────────────
by_lang = defaultdict(list)
by_lang_ver = defaultdict(list)
for r in results:
    by_lang[(r["model"], r["lang_pair"])].append(r["drift"])
    by_lang_ver[(r["model"], r["lang_pair"], r["version"])].append(r["drift"])

print(f"\n{'Model':<20} {'en→te':>10} {'en→ta':>10} {'en→kn':>10}  (mean drift, both versions pooled)")
print("-" * 55)
for m in models:
    row = []
    for lp in ("en→te", "en→ta", "en→kn"):
        vals = by_lang.get((m, lp), [])
        row.append(f"{sum(vals)/len(vals):.3f}(n={len(vals)})" if vals else "—")
    print(f"{m:<20} {row[0]:>14} {row[1]:>14} {row[2]:>14}")

print("\nH3 (novel) — generic vs localized mean drift, per language:")
print(f"{'Model':<20} {'TE gen':>8} {'TE loc':>8} {'TA gen':>8} {'TA loc':>8} {'KN gen':>8} {'KN loc':>8}")
print("-" * 76)
for m in models:
    row = []
    for lp in ("en→te", "en→ta", "en→kn"):
        for v in ("generic", "localized"):
            vals = by_lang_ver.get((m, lp, v), [])
            row.append(f"{sum(vals)/len(vals):.3f}" if vals else "  -  ")
    print(f"{m:<20} {row[0]:>8} {row[1]:>8} {row[2]:>8} {row[3]:>8} {row[4]:>8} {row[5]:>8}")
