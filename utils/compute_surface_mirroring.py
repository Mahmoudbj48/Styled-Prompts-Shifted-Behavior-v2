"""
utils/compute_surface_mirroring.py
----------------------------
Run from project root:
    python utils/compute_surface_mirroring.py

Goes through all full_results_all_models.csv files in the three surface-noise
style folders (spacing, punctuation, letter_case), computes row-level mirroring
using the deterministic detector, and writes a single new column
"mirroring_rate" (0 = no mirroring, 1 = mirroring) into each file in-place.

Existing columns are untouched; mirroring_rate is appended (or overwritten if
it already exists).
"""

import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

RESULTS_DIR   = Path("results")
SURFACE_STYLES = {
    "spacing":      "spacing",
    "punctuation":  "punctuation",
    "letter_case":  "letter_case",
}


# ══════════════════════════════════════════════════════════════════════════════
# Detector (inlined from utils/surface_mirroring_detector.py)
# ══════════════════════════════════════════════════════════════════════════════

def _count_extra_spaces(text: str) -> int:
    return sum(len(s) - 1 for s in re.findall(r' {2,}', text))

def _count_extra_punct(text: str) -> int:
    extra = 0
    extra += sum(len(s) - 1 for s in re.findall(r'!{2,}', text))
    extra += sum(len(s) - 1 for s in re.findall(r'\?{2,}', text))
    extra += sum(len(s) - 3 for s in re.findall(r'\.{4,}', text))
    extra += sum(len(s) - 1 for s in re.findall(r',{2,}', text))
    extra += sum(len(s) - 1 for s in re.findall(r';{2,}', text))
    return max(extra, 0)

def _is_abnormal_word(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    if len(letters) <= 1:
        return False
    if word[0].isupper() and word[1:].islower():   # normal title case
        return False
    if word.islower():
        return False
    if word.isupper() and len(letters) <= 4:        # acronym
        return False
    if word.isupper() and len(letters) > 4:         # all-caps non-acronym
        return True
    upper = sum(1 for c in letters if c.isupper())
    lower = sum(1 for c in letters if c.islower())
    return upper > 0 and lower > 0                  # mixed case

def _abnormal_case_pct(text: str) -> float:
    words = re.findall(r'\b\w+\b', text)
    total = abnormal = 0
    for w in words:
        letters = [c for c in w if c.isalpha()]
        if not letters:
            continue
        total += len(letters)
        if _is_abnormal_word(w):
            abnormal += len(letters)
    return (abnormal / max(total, 1)) * 100


def detect_mirroring(orig: str, pert: str, style: str, strength: int) -> bool:
    """Return True if the response mirrors the style perturbation."""

    if style == "spacing":
        o_extra = _count_extra_spaces(orig)
        p_extra = _count_extra_spaces(pert)
        increase = p_extra - o_extra
        p_rate   = (p_extra / max(len(pert), 1)) * 100
        if strength <= 20:  thresh_inc, thresh_rate = 3,  5
        elif strength <= 50: thresh_inc, thresh_rate = 5,  8
        else:                thresh_inc, thresh_rate = 10, 12
        return increase >= thresh_inc and p_rate >= thresh_rate

    elif style == "punctuation":
        o_extra = _count_extra_punct(orig)
        p_extra = _count_extra_punct(pert)
        increase = p_extra - o_extra
        p_rate   = (p_extra / max(len(pert), 1)) * 100
        if strength <= 5:   thresh_inc, thresh_rate = 2, 3
        elif strength <= 10: thresh_inc, thresh_rate = 4, 5
        else:                thresh_inc, thresh_rate = 6, 8
        return increase >= thresh_inc and p_rate >= thresh_rate

    elif style == "letter_case":
        o_pct    = _abnormal_case_pct(orig)
        p_pct    = _abnormal_case_pct(pert)
        increase = p_pct - o_pct
        if strength <= 25:  threshold = 8
        elif strength <= 50: threshold = 15
        else:                threshold = 25
        return increase >= 5 and p_pct >= threshold

    else:
        raise ValueError(f"Unknown style: {style!r}")


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

updated  = 0
skipped  = 0
errors   = 0

for style_folder, style_key in SURFACE_STYLES.items():
    style_dir = RESULTS_DIR / style_folder
    if not style_dir.exists():
        print(f"[SKIP] folder not found: {style_dir}")
        continue

    csv_files = sorted(style_dir.rglob("full_results_all_models.csv"))
    print(f"\n{'━'*60}")
    print(f"  {style_folder}  ({len(csv_files)} files)")
    print(f"{'━'*60}")

    for csv_path in csv_files:
        rel = csv_path.relative_to(RESULTS_DIR)
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  [ERR] {rel}: {e}")
            errors += 1
            continue

        # Validate required columns
        for col in ("response_orig", "response_pert", "strength"):
            if col not in df.columns:
                print(f"  [SKIP] {rel} — missing column '{col}'")
                skipped += 1
                break
        else:
            # Compute mirroring row-by-row
            results = []
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=str(rel), leave=False):
                try:
                    verdict = detect_mirroring(
                        orig=str(row.get("response_orig", "")),
                        pert=str(row.get("response_pert", "")),
                        style=style_key,
                        strength=int(row.get("strength", 0)),
                    )
                    results.append(1 if verdict else 0)
                except Exception:
                    results.append(0)

            df["mirroring_rate"] = results
            df.to_csv(csv_path, index=False)
            n_mirror = sum(results)
            print(f"  [OK]  {rel}  "
                  f"({n_mirror}/{len(df)} mirroring  "
                  f"{100*n_mirror/max(len(df),1):.1f}%)")
            updated += 1

print(f"\n{'='*60}")
print(f"  Updated : {updated}")
print(f"  Skipped : {skipped}")
print(f"  Errors  : {errors}")
print(f"{'='*60}")
print("\nNext step: re-run  python aggregate_results.py")
