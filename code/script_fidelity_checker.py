"""
Script fidelity checker for Telugu, Tamil, Kannada model outputs.
Extends the original paper's Gurmukhi/Devanagari character-ratio approach
(Section 3.3) to the three Dravidian scripts.

Usage:
    from script_fidelity_checker import check_fidelity
    result = check_fidelity(response_text, expected_lang="te")
    # result = {"lang": "te", "script_ratio": 0.83, "passed": True,
    #           "total_chars": 512, "script_chars": 425, "other_script_hits": {}}
"""

import re
from collections import Counter

# Unicode block ranges (inclusive), matching the paper's Section 3.3 style.
SCRIPT_RANGES = {
    "te": (0x0C00, 0x0C7F),   # Telugu
    "ta": (0x0B80, 0x0BFF),   # Tamil
    "kn": (0x0C80, 0x0CFF),   # Kannada
    "hi": (0x0900, 0x097F),   # Devanagari (for cross-script substitution checks)
    "pa": (0x0A00, 0x0A7F),   # Gurmukhi (for cross-script substitution checks)
}

# CJK range, to catch the "Chinese character intrusion" failure mode
# documented in the original paper (Section 4.1, Qwen 2.5 7B).
CJK_RANGE = (0x4E00, 0x9FFF)

FIDELITY_THRESHOLD = 0.70  # matches the paper's ≥70% Gurmukhi threshold


def _char_in_range(ch, rng):
    return rng[0] <= ord(ch) <= rng[1]


def check_fidelity(text: str, expected_lang: str, threshold: float = FIDELITY_THRESHOLD) -> dict:
    """
    expected_lang: one of "te", "ta", "kn"
    Returns a dict with the script ratio, pass/fail, and a breakdown of
    which OTHER scripts appeared (to diagnose cross-script substitution,
    English fallback, or CJK intrusion, as in the paper's four failure modes).
    """
    if expected_lang not in ("te", "ta", "kn"):
        raise ValueError("expected_lang must be one of: te, ta, kn")

    # Count only "letter-like" characters; ignore whitespace, digits, punctuation.
    letters = [ch for ch in text if ch.isalpha()]
    total = len(letters)
    if total == 0:
        return {
            "lang": expected_lang, "script_ratio": 0.0, "passed": False,
            "total_chars": 0, "script_chars": 0, "other_script_hits": {},
            "note": "empty or non-letter response"
        }

    target_range = SCRIPT_RANGES[expected_lang]
    script_chars = sum(1 for ch in letters if _char_in_range(ch, target_range))

    # Diagnose what else showed up, mirroring the paper's failure taxonomy:
    # cross-script substitution, English fallback, CJK intrusion.
    other_hits = Counter()
    for ch in letters:
        if _char_in_range(ch, target_range):
            continue
        if _char_in_range(ch, CJK_RANGE):
            other_hits["chinese_intrusion"] += 1
        elif ch.isascii():
            other_hits["english_fallback"] += 1
        else:
            for lang, rng in SCRIPT_RANGES.items():
                if lang != expected_lang and _char_in_range(ch, rng):
                    other_hits[f"cross_script_{lang}"] += 1
                    break

    ratio = script_chars / total
    return {
        "lang": expected_lang,
        "script_ratio": round(ratio, 4),
        "passed": ratio >= threshold,
        "total_chars": total,
        "script_chars": script_chars,
        "other_script_hits": dict(other_hits),
    }


def detect_degeneration_loop(text: str, ngram_size: int = 8, repeat_threshold: int = 3) -> bool:
    """
    Flags the "degeneration loop" failure mode (Section 4.1, Appendix I #1):
    a model repeating the same sentence/n-gram to fill the token budget.
    Checks word-level n-grams for excessive repetition.
    """
    words = text.split()
    if len(words) < ngram_size * repeat_threshold:
        return False
    ngrams = [" ".join(words[i:i + ngram_size]) for i in range(len(words) - ngram_size + 1)]
    counts = Counter(ngrams)
    return any(c >= repeat_threshold for c in counts.values())


if __name__ == "__main__":
    # Quick smoke test with representative strings.
    samples = {
        "te": "జసవందర్ జలంధర్‌లో తన తల్లిదండ్రుల ఇకలోతే బిడ్డ.",
        "ta": "ஜஸ்விந்தர் ஜலந்தரில் தனது பெற்றோரின் ஒரே மகன்.",
        "kn": "ಜಸ್ವಿಂದರ್ ಜಲಂಧರ್‌ನಲ್ಲಿ ತನ್ನ ಪೋಷಕರ ಏಕೈಕ ಮಗ.",
        "english_fallback": "This is a very difficult situation for the family.",
    }
    for lang in ("te", "ta", "kn"):
        print(lang, "->", check_fidelity(samples[lang], lang))
    print("fallback as te ->", check_fidelity(samples["english_fallback"], "te"))
