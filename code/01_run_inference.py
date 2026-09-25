"""
Cross-Lingual Cultural Value Drift — Dravidian Extension — Inference Script
Adapted from the reference repo's 01_run_inference.py (v5).

WHAT CHANGED FROM THE REFERENCE SCRIPT (see ../CHANGELOG.md for full detail):
- RELIGIONS=["sikh","hindu"]  ->  VERSIONS=["generic","localized"]
- LANGS=["en","hi","pa"]      ->  LANGS=["en","te","ta","kn"]
- Prompt-building is NOT a uniform version x lang double loop, because your
  scenario_bank structure is asymmetric:
    scenario["generic"]["en"/"te"/"ta"/"kn"]   <- same 4 langs for every scenario
    scenario["localized"]["telugu"]["en"/"te"]  <- only 2 langs, per-language key
    scenario["localized"]["tamil"]["en"/"ta"]
    scenario["localized"]["kannada"]["en"/"kn"]
- UPDATED: localized-EN is now its own inference condition, once per region
  (see the "build prompt list" section below for the full explanation). This
  fixes an anchor-mismatch bug in 03_embed_distance.py, where the localized-
  condition drift score had no choice but to compare a generic-EN response
  against a localized native-script response. Matrix is now 200 prompts/model
  (was 140): 1 EN baseline + 3 EN localized anchors + 3 langs x 2 conditions.
- Script fidelity: replaced the reference repo's presence-only check
  (has_gurmukhi/has_devanagari) with the ratio-based checker from
  script_fidelity_checker.py (your file, already correct per the paper's
  stated >=70% method -- the reference repo's own check was a bug, not a
  spec to copy).
- Output filename bug fixed: this script and 02/03 all now agree on
  raw_responses.json (reference repo had 01 write raw_responses_v5.json
  while 02/03 read raw_responses.json).
- Added --dry_run and --limit / --models CLI flags so you can sanity-check
  the full pipeline (prompt building -> output JSON schema) on a tiny slice,
  or with zero GPU, before spending real compute. See "Dry pass" section
  below and the note in CHANGELOG.md.

FLAGGED ASSUMPTIONS -- please confirm before the real GPU run:
1. TOKENS_BY_LANG values for te/ta/kn are PLACEHOLDERS carried over from the
   Hindi/Punjabi values as a starting guess. Run token_budget_calibration.py
   first and paste its recommended numbers in here. The retry-and-multiply
   logic will self-correct per-response either way, but a bad starting point
   wastes GPU time on early truncated attempts.
2. advisor_wrapper in data/all_scenarios.json is a DRAFT translation (see
   the "_needs_review" flag in that file) -- this script prints a loud
   warning at startup if it's still unreviewed. Don't burn GPU time on a
   real run until a native speaker has checked it.
3. PAPER_MODELS below is the same 5-model list the reference repo ran
   locally (Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B, Gemma 2 9B, Aya Expanse
   8B). Your project notes say the 5-vs-6-models decision (whether to also
   run Llama 3.3 70B, which the reference repo ran separately via Groq) is
   still open with your professor. I did not add it -- add a Groq-based
   6th pass separately if you decide to include it, since it needs a
   different (non-transformers/bitsandbytes) code path.
"""

import argparse, json, os, sys, time, gc, platform, datetime
from collections import defaultdict

from script_fidelity_checker import check_fidelity, detect_degeneration_loop

# Force unbuffered stdout so every print shows up in PBS log immediately (with utf-8 for Windows consoles)
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

# ── logging helper ─────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = {"INFO": "   ", "OK": "✓  ", "WARN": "!  ", "ERR": "ERR", "HDR": ">>>"}
    print(f"[{ts}] [{tag.get(level,'   ')}] {msg}", flush=True)

def section(title):
    width = 70
    log("=" * width)
    log(f"  {title}")
    log("=" * width)

def subsection(title):
    log(f"  ── {title} ──")

def fmt_eta(seconds):
    if seconds < 0 or seconds > 86400:
        return "??:??:??"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def fmt_elapsed(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def progress_bar(current, total, width=40):
    pct    = current / total if total > 0 else 0
    filled = int(width * pct)
    bar    = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total} ({100*pct:.1f}%)"

# ── CLI args ───────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--dry_run", action="store_true",
                 help="Skip model loading and generation entirely. Builds the full "
                      "140-prompt list, writes stub responses, and produces a real "
                      "raw_responses.json with the correct schema so you can sanity-"
                      "check the pipeline with zero GPU time.")
ap.add_argument("--limit", type=int, default=None,
                 help="Only run the first N prompts (per model). For a real, "
                      "small-GPU-time sanity check before the full run.")
ap.add_argument("--models", type=str, default=None,
                 help="Comma-separated model 'name' values (see PAPER_MODELS) to "
                      "restrict this run to, e.g. --models qwen25_7b")
args = ap.parse_args()

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "../results/checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# ── 5 models, same as reference repo's local run ───────────────────────────────
PAPER_MODELS = [
    {"id": "meta-llama/Llama-3.1-8B-Instruct",  "name": "llama31_8b",     "gated": True,  "eager": False, "no_cache": False},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "mistral_7b",     "gated": False, "eager": False, "no_cache": False},
    {"id": "Qwen/Qwen2.5-7B-Instruct",           "name": "qwen25_7b",      "gated": False, "eager": False, "no_cache": False},
    {"id": "google/gemma-2-9b-it",               "name": "gemma2_9b",      "gated": True,  "eager": False, "no_cache": False},
    {"id": "CohereForAI/aya-expanse-8b",         "name": "aya_expanse_8b", "gated": False, "eager": False, "no_cache": False},
]
if args.models:
    wanted = set(x.strip() for x in args.models.split(","))
    PAPER_MODELS = [m for m in PAPER_MODELS if m["name"] in wanted]

# ── token budget config — PLACEHOLDERS, see module docstring point 1 ──────────
# TOKENS_BY_LANG = {"en": 600, "te": 1500, "ta": 1800, "kn": 1500}  # <-- update from calibration
TOKENS_BY_LANG = {"en": 500, "te": 2950, "ta": 2750, "kn": 2300} 
MAX_RETRIES      = 3
RETRY_MULTIPLIER = 1.5
MAX_TOKENS_CAP   = 4096

LANGS    = ["en", "te", "ta", "kn"]
VERSIONS = ["generic", "localized"]
LOCALIZED_KEY = {"te": "telugu", "ta": "tamil", "kn": "kannada"}

# ── quantisation (only needed when not --dry_run) ──────────────────────────────
if not args.dry_run:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    def gpu_mem():
        if not torch.cuda.is_available():
            return "no-GPU"
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        total     = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"alloc={allocated:.2f}GB  reserved={reserved:.2f}GB  total={total:.2f}GB"

    def gpu_mem_free():
        if not torch.cuda.is_available():
            return "no-GPU"
        free  = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"free={free:.2f}GB / {total:.2f}GB"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
else:
    def gpu_mem(): return "dry-run (no torch)"
    def gpu_mem_free(): return "dry-run (no torch)"

# ══════════════════════════════════════════════════════════════════════════════
# STARTUP BANNER
# ══════════════════════════════════════════════════════════════════════════════
section("CULTURAL VALUE DRIFT — DRAVIDIAN EXTENSION — INFERENCE — STARTUP")
log(f"Script       : {os.path.abspath(__file__)}")
log(f"Results dir  : {RESULTS_DIR}")
log(f"Start time   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Mode         : {'DRY RUN (no GPU, stub responses)' if args.dry_run else 'REAL INFERENCE'}")
if args.limit:
    log(f"Limit        : first {args.limit} prompts/model")
log("")
if not args.dry_run:
    import torch as _torch_for_log  # already imported above; alias avoids re-import confusion
    log(f"Python       : {platform.python_version()}")
    log(f"PyTorch      : {_torch_for_log.__version__}")
    log(f"CUDA available: {_torch_for_log.cuda.is_available()}")
    if _torch_for_log.cuda.is_available():
        log(f"CUDA version : {_torch_for_log.version.cuda}")
        log(f"GPU count    : {_torch_for_log.cuda.device_count()}")
        for i in range(_torch_for_log.cuda.device_count()):
            props = _torch_for_log.cuda.get_device_properties(i)
            log(f"  GPU {i}      : {props.name}  VRAM={props.total_memory/1024**3:.1f}GB")
    log(f"GPU memory   : {gpu_mem()}")
    log(f"HF_TOKEN     : {'set (len=' + str(len(HF_TOKEN)) + ')' if HF_TOKEN else 'NOT SET — gated models will be skipped'}")
    log("")
log("Token budgets per language (PLACEHOLDERS — see docstring point 1):")
for lang, toks in TOKENS_BY_LANG.items():
    max1 = min(int(toks * RETRY_MULTIPLIER**0), MAX_TOKENS_CAP)
    max2 = min(int(toks * RETRY_MULTIPLIER**1), MAX_TOKENS_CAP)
    max3 = min(int(toks * RETRY_MULTIPLIER**2), MAX_TOKENS_CAP)
    log(f"  {lang.upper()}: attempt1={max1}  attempt2={max2}  attempt3={max3}  cap={MAX_TOKENS_CAP}")
log(f"MAX_RETRIES={MAX_RETRIES}  RETRY_MULTIPLIER={RETRY_MULTIPLIER}  MAX_TOKENS_CAP={MAX_TOKENS_CAP}")
log("")
log("Models to run:")
for i, m in enumerate(PAPER_MODELS, 1):
    gated = "GATED" if m["gated"] else "public"
    log(f"  {i}. {m['name']:<20}  [{gated}]  {m['id']}")
if not PAPER_MODELS:
    log("No models selected (check --models value) — nothing to do.", "ERR")
    sys.exit(1)

# ── scenario bank ─────────────────────────────────────────────────────────────
section("LOADING SCENARIO BANK")
bank_path = os.path.join(BASE_DIR, "../data/all_scenarios.json")
log(f"Bank path: {bank_path}")
with open(bank_path, encoding="utf-8") as f:
    bank = json.load(f)
log(f"Scenarios loaded : {len(bank['scenarios'])}", "OK")
log(f"Languages        : {LANGS}")
log(f"Versions         : {VERSIONS}")

wrapper_meta = bank.get("advisor_wrapper", {})
if wrapper_meta.get("_needs_review", False):
    log("advisor_wrapper is UNREVIEWED — " + wrapper_meta.get("_note", "")[:120] + " ...", "WARN")
    log("This is a DRAFT translation, not verified by a native speaker. See docstring point 2.", "WARN")

# ── script fidelity wrapper (delegates to your ratio-based checker) ───────────
def script_fidelity(text, lang):
    """
    Returns True/False for use in the record's 'script_ok' field, matching the
    reference script's calling convention. English always passes (no script
    check needed). Full diagnostics (ratio, cross-script hits) are stored
    separately per-record via script_fidelity_detail().
    """
    if lang == "en":
        return True
    result = check_fidelity(text, lang)
    return result["passed"]

def script_fidelity_detail(text, lang):
    if lang == "en":
        return {"lang": "en", "script_ratio": 1.0, "passed": True,
                "total_chars": len(text), "script_chars": len(text), "other_script_hits": {}}
    return check_fidelity(text, lang)

def build_prompt(scenario, version, lang, region=None):
    wrapper = bank["advisor_wrapper"].get(lang, "")
    if version == "generic":
        text = scenario["generic"][lang]
    else:  # localized
        region = region or LOCALIZED_KEY[lang]
        text = scenario["localized"][region][lang]
    return (wrapper + "\n\n" + text) if wrapper else text

# ── build prompt list ─────────────────────────────────────────────────────────
# UPDATED (fixes the embed-distance anchor-mismatch bug): the old matrix ran
# only ONE English response per scenario (the generic-EN baseline), which
# forced 03_embed_distance.py to compare that single generic-EN response
# against BOTH the generic-condition AND the localized-condition native-
# script responses. For the localized comparison that meant comparing a
# GENERIC English answer against a LOCALIZED Telugu/Tamil/Kannada answer --
# a same-condition claim the data didn't support.
# FIX: run localized-EN as its own inference condition, once per Dravidian
# region (te/ta/kn each have their own localized-EN source text, e.g.
# "Venkatesh near Karimnagar" vs "Selvam near Thanjavur"), so every
# localized-condition drift score can be computed against ITS OWN matching
# English anchor. This raises the matrix from 140 to 200 prompts/model:
#   1 EN baseline (generic)
# + 3 EN localized anchors (one per region: telugu/tamil/kannada)
# + 3 langs x 2 conditions (generic, localized), native-script
# = 1 + 3 + 6 = 10 per scenario x 20 scenarios = 200 prompts/model.
# Every record now also carries a "region" field (None for the shared
# generic-EN baseline; "telugu"/"tamil"/"kannada" for everything else) so
# 03_embed_distance.py can look up the correct region-matched anchor without
# any ambiguity, even though several records share lang == "en".
prompt_list = []
for s in bank["scenarios"]:
    # (1) EN baseline — generic condition. Shared across all 3 languages;
    #     no single region, so region=None.
    prompt_list.append({
        "scenario_id": s["id"],
        "dimension":   s["dimension"],
        "version":     "generic",
        "lang":        "en",
        "region":      None,
        "prompt":      build_prompt(s, "generic", "en"),
    })
    # (2) EN localized anchors — ONE PER REGION. This is the fix: each of
    #     these is now a real model response, so the localized-condition
    #     drift score has its own matching English anchor instead of
    #     reusing the generic one.
    for lang in ("te", "ta", "kn"):
        region = LOCALIZED_KEY[lang]
        prompt_list.append({
            "scenario_id": s["id"],
            "dimension":   s["dimension"],
            "version":     "localized",
            "lang":        "en",
            "region":      region,
            "prompt":      build_prompt(s, "localized", "en", region=region),
        })
    # (3) For each Dravidian language: generic native-script + localized native-script
    for lang in ("te", "ta", "kn"):
        region = LOCALIZED_KEY[lang]
        prompt_list.append({
            "scenario_id": s["id"],
            "dimension":   s["dimension"],
            "version":     "generic",
            "lang":        lang,
            "region":      region,
            "prompt":      build_prompt(s, "generic", lang),
        })
        prompt_list.append({
            "scenario_id": s["id"],
            "dimension":   s["dimension"],
            "version":     "localized",
            "lang":        lang,
            "region":      region,
            "prompt":      build_prompt(s, "localized", lang, region=region),
        })

TOTAL_PROMPTS = len(prompt_list)
EXPECTED = len(bank["scenarios"]) * (1 + 3 + 3 * 2)
log(f"Total prompts/model: {TOTAL_PROMPTS}  "
    f"({len(bank['scenarios'])} scenarios x (1 EN baseline + 3 EN localized anchors + 3 langs x 2 versions))", "OK")
assert TOTAL_PROMPTS == EXPECTED, (
    f"Prompt count mismatch: built {TOTAL_PROMPTS}, expected {EXPECTED} "
    f"({len(bank['scenarios'])} scenarios x 10). Check scenario_bank structure."
)
if TOTAL_PROMPTS != 200:
    log(f"NOTE: expected count is {EXPECTED}, not 200 — "
        f"you have {len(bank['scenarios'])} scenarios, not 20. Confirm this is intended.", "WARN")
else:
    log("Prompt count matches the corrected matrix (200/model, up from 140). Confirmed.", "OK")

if args.limit:
    prompt_list = prompt_list[:args.limit]
    TOTAL_PROMPTS = len(prompt_list)
    log(f"--limit applied: running {TOTAL_PROMPTS} prompts/model", "WARN")

# ── core generation ────────────────────────────────────────────────────────────
def _generate_once(model, tokenizer, user_text, max_new_tokens, no_cache=False):
    import torch
    messages = [
        {"role": "system", "content": "You are a helpful advisor giving practical life advice."},
        {"role": "user",   "content": user_text},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"### Instruction:\n{user_text}\n\n### Response:"

    enc            = tokenizer(text, return_tensors="pt")
    input_ids      = enc.input_ids.to(model.device)
    attention_mask = enc.attention_mask.to(model.device)
    n_input_toks   = input_ids.shape[-1]

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.15,   # prevents Indic/Dravidian-script degeneration loops
            pad_token_id=tokenizer.eos_token_id,
            use_cache=not no_cache,
        )

    generated_ids = output[0][input_ids.shape[-1]:]
    n_generated   = len(generated_ids)
    truncated     = (n_generated >= max_new_tokens)   # exact check
    decoded       = tokenizer.decode(generated_ids, skip_special_tokens=True)

    del input_ids, attention_mask, output, enc, generated_ids
    gc.collect()
    torch.cuda.empty_cache()

    return decoded.strip(), n_generated, n_input_toks, truncated


def generate_complete(model, tokenizer, user_text, lang, prompt_idx, no_cache=False):
    """
    Retry loop: start at TOKENS_BY_LANG[lang], multiply by RETRY_MULTIPLIER on
    truncation, cap at MAX_TOKENS_CAP. Kept unchanged from the reference repo
    — this design is good and self-corrects even with placeholder starting
    budgets.
    """
    base = TOKENS_BY_LANG[lang]

    for attempt in range(1, MAX_RETRIES + 1):
        max_toks = min(int(base * (RETRY_MULTIPLIER ** (attempt - 1))), MAX_TOKENS_CAP)

        log(f"    Attempt {attempt}/{MAX_RETRIES} | budget={max_toks} tokens | lang={lang.upper()}")
        t_gen = time.time()
        response, n_generated, n_input, truncated = _generate_once(
            model, tokenizer, user_text, max_toks, no_cache
        )
        t_gen = time.time() - t_gen

        chars_out   = len(response)
        chars_trunc = "TRUNCATED" if truncated else "COMPLETE"
        log(f"    Result    : {chars_trunc} | generated={n_generated}/{max_toks} tokens"
            f" | {chars_out} chars | {t_gen:.1f}s")

        if not truncated:
            log(f"    Response complete on attempt {attempt}.", "OK")
            return response, n_generated, n_input, max_toks, attempt, False

        if max_toks >= MAX_TOKENS_CAP:
            log(f"    Token cap {MAX_TOKENS_CAP} reached — accepting truncated response.", "WARN")
            break

        log(f"    Truncated — increasing budget and retrying...", "WARN")

    return response, n_generated, n_input, max_toks, attempt, True


def generate_complete_dry(user_text, lang, prompt_idx):
    """--dry_run stand-in: no model, no GPU. Produces a plausible-shaped
    stub response so the rest of the pipeline (fidelity check, JSON schema,
    checkpointing) can be exercised end-to-end."""
    stub = f"[DRY RUN STUB #{prompt_idx}] lang={lang} :: " + user_text[:60].replace("\n", " ")
    n_generated = len(stub.split())
    return stub, n_generated, len(user_text.split()), TOKENS_BY_LANG[lang], 1, False


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT RESUME CHECK
# ══════════════════════════════════════════════════════════════════════════════
section("CHECKPOINT RESUME CHECK")
all_results      = []
completed_models = set()

for m in PAPER_MODELS:
    ckpt = os.path.join(RESULTS_DIR, f"checkpoint_{m['name']}.json")
    if not os.path.exists(ckpt):
        log(f"  {m['name']:<20} — no checkpoint found, will run from scratch")
        continue
    data  = json.load(open(ckpt, encoding="utf-8"))
    valid = [r for r in data if r.get("response", "").strip()]
    if len(valid) == TOTAL_PROMPTS:
        all_results.extend(data)
        completed_models.add(m["name"])
        trunc = sum(1 for r in valid if r.get("truncated"))
        log(f"  {m['name']:<20} — COMPLETE ({len(valid)} records, {trunc} still truncated)", "OK")
    else:
        log(f"  {m['name']:<20} — PARTIAL ({len(valid)}/{TOTAL_PROMPTS}) — will re-run from scratch", "WARN")

log(f"Models already complete: {sorted(completed_models) or 'none'}")
models_to_run = [m for m in PAPER_MODELS if m["name"] not in completed_models]
log(f"Models to run now      : {[m['name'] for m in models_to_run]}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ══════════════════════════════════════════════════════════════════════════════
job_start = time.time()

for model_idx, model_cfg in enumerate(models_to_run, 1):
    model_name = model_cfg["name"]
    model_id   = model_cfg["id"]

    section(f"MODEL {model_idx}/{len(models_to_run)}: {model_name}")
    log(f"HuggingFace ID : {model_id}")
    log(f"GPU memory before load: {gpu_mem()}")

    model = tokenizer = None
    if not args.dry_run:
        if model_cfg["gated"] and not HF_TOKEN:
            log(f"SKIPPING — gated model and HF_TOKEN is not set", "ERR")
            continue

        subsection("Loading tokenizer")
        t_load = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=HF_TOKEN or None, trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                log("  pad_token was None — set to eos_token", "WARN")
            log(f"  Tokenizer loaded | vocab_size={tokenizer.vocab_size}", "OK")
        except Exception as e:
            log(f"  FAILED to load tokenizer: {e}", "ERR")
            continue

        subsection("Loading model (4-bit quantised)")
        load_kwargs = dict(
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN or None,
            trust_remote_code=True,
        )
        if model_cfg.get("eager"):
            load_kwargs["attn_implementation"] = "eager"
            log("  Using eager attention implementation")

        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            model.eval()
        except Exception as e:
            log(f"  FAILED to load model: {str(e)[:300]}", "ERR")
            continue

        t_load = time.time() - t_load
        log(f"  Model loaded in {fmt_elapsed(t_load)}", "OK")
        log(f"  GPU memory after load: {gpu_mem()}")
    else:
        log("  --dry_run: skipping tokenizer/model load entirely", "WARN")

    subsection(f"Running inference — {TOTAL_PROMPTS} prompts")
    log(f"  Checkpoint saved every 10 prompts to {RESULTS_DIR}/checkpoint_{model_name}.json")
    log("")

    model_results      = []
    consecutive_errors = 0
    no_cache            = model_cfg.get("no_cache", False)

    stats = defaultdict(lambda: {"complete": 0, "truncated": 0, "error": 0,
                                  "total_toks": 0, "total_time": 0.0, "attempts_sum": 0})
    model_start = time.time()

    for i, p in enumerate(prompt_list):
        if consecutive_errors >= 10:
            log(f"  10 consecutive errors — aborting this model", "ERR")
            break

        prompt_start = time.time()
        lang         = p["lang"]
        log(f"")
        log(f"  ┌─ Prompt {i+1:03d}/{TOTAL_PROMPTS}  {progress_bar(i+1, TOTAL_PROMPTS, width=30)}")
        log(f"  │  scenario={p['scenario_id']}  version={p['version']}  lang={lang.upper()}  dim={p['dimension']}")

        try:
            if args.dry_run:
                response, n_tokens, n_input, max_toks_used, attempts, truncated = generate_complete_dry(
                    p["prompt"], lang, i
                )
            else:
                response, n_tokens, n_input, max_toks_used, attempts, truncated = generate_complete(
                    model, tokenizer, p["prompt"], lang, i, no_cache=no_cache
                )
            consecutive_errors = 0
        except Exception as e:
            log(f"  │  EXCEPTION: {str(e)[:200]}", "ERR")
            response, n_tokens, n_input, max_toks_used, attempts, truncated = "", 0, 0, 0, 1, False
            consecutive_errors += 1
            if not args.dry_run:
                gc.collect()
                torch.cuda.empty_cache()

        elapsed      = time.time() - prompt_start
        fid_detail   = script_fidelity_detail(response, lang)
        fidelity     = fid_detail["passed"]
        degeneration = detect_degeneration_loop(response) if response else False
        chars_out    = len(response)

        if response.strip():
            if truncated:
                stats[lang]["truncated"] += 1
            else:
                stats[lang]["complete"] += 1
        else:
            stats[lang]["error"] += 1
        stats[lang]["total_toks"]   += n_tokens
        stats[lang]["total_time"]   += elapsed
        stats[lang]["attempts_sum"] += attempts

        done_so_far   = i + 1
        elapsed_total = time.time() - model_start
        rate          = elapsed_total / done_so_far if done_so_far > 0 else 0
        remaining     = (TOTAL_PROMPTS - done_so_far) * rate
        eta_str       = fmt_eta(remaining)

        status_icon = "OK" if (fidelity and not truncated) else ("WARN" if truncated else "ERR")
        log(f"  │  script_ok={fidelity} (ratio={fid_detail['script_ratio']})  chars={chars_out}  "
            f"input_toks={n_input}  out_toks={n_tokens}  attempts={attempts}  {elapsed:.1f}s", status_icon)
        if degeneration:
            log(f"  │  DEGENERATION LOOP DETECTED (repeated n-gram) — flagging record", "WARN")
        log(f"  │  ETA for model: {eta_str}  |  elapsed: {fmt_elapsed(elapsed_total)}  "
            f"|  avg_time/prompt: {rate:.1f}s")
        log(f"  └─ done")

        model_results.append({
            "model":              model_name,
            "scenario_id":        p["scenario_id"],
            "dimension":          p["dimension"],
            "version":            p["version"],
            "lang":               lang,
            "region":             p.get("region"),
            "prompt":             p["prompt"],
            "response":           response,
            "script_ok":          fidelity,
            "script_ratio":       fid_detail["script_ratio"],
            "other_script_hits":  fid_detail["other_script_hits"],
            "degeneration_loop":  degeneration,
            "truncated":          truncated,
            "n_tokens_generated": n_tokens,
            "n_input_tokens":     n_input,
            "max_tokens_used":    max_toks_used,
            "attempts":           attempts,
            "elapsed_s":          round(elapsed, 2),
        })

        if (i + 1) % 10 == 0:
            ckpt_path = os.path.join(RESULTS_DIR, f"checkpoint_{model_name}.json")
            with open(ckpt_path, "w", encoding="utf-8") as cf:
                json.dump(model_results, cf, ensure_ascii=False, indent=2)

            log("")
            log(f"  ══ CHECKPOINT SAVED ({i+1}/{TOTAL_PROMPTS}) ══════════════════════════", "OK")
            log(f"     File: {ckpt_path}")
            log(f"     GPU memory: {gpu_mem()}")
            log(f"     Running stats by language:")
            log(f"     {'Lang':<6} {'Complete':>10} {'Truncated':>10} {'Error':>7} {'AvgToks':>9} {'AvgTime':>9} {'AvgAtt':>8}")
            log(f"     {'-'*60}")
            for lk in LANGS:
                s   = stats[lk]
                tot = s["complete"] + s["truncated"] + s["error"]
                if tot == 0:
                    continue
                avg_toks = s["total_toks"] / tot if tot else 0
                avg_time = s["total_time"] / tot if tot else 0
                avg_att  = s["attempts_sum"] / tot if tot else 0
                log(f"     {lk.upper():<6} {s['complete']:>10} {s['truncated']:>10} {s['error']:>7} "
                    f"{avg_toks:>9.0f} {avg_time:>8.1f}s {avg_att:>8.2f}")
            log(f"  ═══════════════════════════════════════════════════════════")
            log("")

    ckpt_path = os.path.join(RESULTS_DIR, f"checkpoint_{model_name}.json")
    with open(ckpt_path, "w", encoding="utf-8") as cf:
        json.dump(model_results, cf, ensure_ascii=False, indent=2)

    model_elapsed = time.time() - model_start
    valid_count   = sum(1 for r in model_results if r["response"].strip())
    trunc_count   = sum(1 for r in model_results if r.get("truncated"))
    error_count   = sum(1 for r in model_results if not r["response"].strip())
    ok_count      = valid_count - trunc_count

    log("")
    log(f"  ══ MODEL COMPLETE: {model_name} ══════════════════════════════════", "OK")
    log(f"     Total prompts     : {len(model_results)}")
    if model_results:
        log(f"     Fully complete    : {ok_count}  ({100*ok_count/len(model_results):.1f}%)")
        log(f"     Still truncated   : {trunc_count}  ({100*trunc_count/len(model_results):.1f}%)")
        log(f"     Errors/empty      : {error_count}  ({100*error_count/len(model_results):.1f}%)")
    log(f"     Model wall time   : {fmt_elapsed(model_elapsed)}")
    log(f"     Checkpoint        : {ckpt_path}")
    log(f"     GPU memory after  : {gpu_mem()}")
    log(f"  ═══════════════════════════════════════════════════════════════════")

    all_results.extend(model_results)

    if not args.dry_run and model is not None:
        log(f"  Unloading model from GPU...")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        log(f"  GPU memory after unload: {gpu_mem_free()}", "OK")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
section("SAVING FINAL OUTPUT")

# raw_responses.json — matches what 02_judge_responses.py and 03_embed_distance.py
# both expect (filename bug from the reference repo is fixed here).
out_path = os.path.join(RESULTS_DIR, "raw_responses.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
log(f"Saved {len(all_results)} records to: {out_path}", "OK")

valid_total  = sum(1 for r in all_results if r.get("response", "").strip())
trunc_total  = sum(1 for r in all_results if r.get("truncated"))
error_total  = sum(1 for r in all_results if not r.get("response", "").strip())
ok_total     = valid_total - trunc_total
models_done  = sorted(set(r["model"] for r in all_results))
job_elapsed  = time.time() - job_start

section("FINAL SUMMARY")
log(f"Job wall time       : {fmt_elapsed(job_elapsed)}")
log(f"Models completed    : {models_done}")
log(f"Total records       : {len(all_results)}")
if all_results:
    log(f"Fully complete      : {ok_total}  ({100*ok_total/max(len(all_results),1):.1f}%)")
    log(f"Still truncated     : {trunc_total}  ({100*trunc_total/max(len(all_results),1):.1f}%)")
log(f"Errors / empty      : {error_total}")
log("")
log("Per-model per-language x version breakdown (complete/truncated):")
header = f"{'Model':<18}"
for lk in LANGS:
    header += f" {lk.upper()+' gen':>10} {lk.upper()+' loc':>10}"
log(header)
log("-" * len(header))
for mn in models_done:
    mr  = [r for r in all_results if r["model"] == mn]
    row = f"{mn:<18}"
    for lk in LANGS:
        for ver in ("generic", "localized"):
            sub  = [r for r in mr if r["lang"] == lk and r["version"] == ver]
            comp = sum(1 for r in sub if not r.get("truncated") and r.get("response", "").strip())
            row += f" {comp}/{len(sub):>7}" if sub else f" {'—':>10}"
    log(row)
log("")
log("")
log("EN localized anchors by region (new — fixes the embed-distance anchor bug):")
header2 = f"{'Model':<18} {'telugu':>10} {'tamil':>10} {'kannada':>10}"
log(header2)
log("-" * len(header2))
for mn in models_done:
    mr  = [r for r in all_results if r["model"] == mn and r["lang"] == "en" and r["version"] == "localized"]
    row = f"{mn:<18}"
    for region in ("telugu", "tamil", "kannada"):
        sub  = [r for r in mr if r.get("region") == region]
        comp = sum(1 for r in sub if not r.get("truncated") and r.get("response", "").strip())
        row += f" {comp}/{len(sub):>7}" if sub else f" {'—':>10}"
    log(row)
log("")
log(f"Output file: {out_path}", "OK")
log("ALL DONE.", "OK")
