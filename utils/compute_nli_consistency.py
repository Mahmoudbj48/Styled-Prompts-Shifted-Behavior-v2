"""
utils/compute_nli_consistency.py
---------------------------------
Computes NLI-based factual consistency metrics for all runs in the checklist
and appends them to each full_results_all_models.csv in-place.

New columns added
-----------------
  nli_contradiction        : P(contradiction | response_orig, response_pert) in [0, 1]
  nli_entailment           : P(entailment    | response_orig, response_pert) in [0, 1]
  delta_nli_contradiction  : signed delta relative to within-group baseline
  delta_nli_entailment     : signed delta relative to within-group baseline

Metric justification
--------------------
NLI contradiction between a model's baseline response and its perturbed-prompt
response captures factual instability that BLEU and BERTScore cannot detect:
two responses can be surface-dissimilar but logically consistent, or
surface-similar but factually contradictory. This provides a logit-free proxy
for confidence stability applicable to all models including closed-source ones.

Model  : cross-encoder/nli-deberta-v3-large  (Laurer et al., 2022, EMNLP)
Cited in: Kuhn et al. 2023 (ICLR), Farquhar et al. 2024 (Nature)

Usage
-----
Run from project root:
    # open-source checklist
    python utils/compute_nli_consistency.py --mode open

    # closed-source checklist (run in parallel on a different terminal)
    python utils/compute_nli_consistency.py --mode closed

    --mode       : open | closed
    --force      : recompute even if columns already present
    --batch-size : NLI inference batch size (default 16)
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import openpyxl
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode",       choices=["open", "closed"], required=True,
                    help="open = experiment_checklist_filled.xlsx  |  "
                         "closed = experiment_checklist_closed_filled.xlsx")
parser.add_argument("--force",      action="store_true",
                    help="Recompute even if NLI columns already present.")
parser.add_argument("--batch-size", type=int, default=16,
                    help="Inference batch size (default 16).")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
CHECKLIST_MAP = {
    "open":   Path("experiment_checklist_filled.xlsx"),
    "closed": Path("experiment_checklist_closed_filled.xlsx"),
}
CHECKLIST_PATH = CHECKLIST_MAP[args.mode]

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

NLI_COLS = [
    "nli_contradiction",
    "nli_entailment",
    "delta_nli_contradiction",
    "delta_nli_entailment",
]

ALL_STYLES = {"Polite.", "Punct.", "Spacing", "Casing", "Length", "Form"}

# Baseline strength per style — mirrors compute_delta_metrics.py
BASELINE_STRENGTH = {
    "Polite.":  0,
    "Spacing":  0,
    "Punct.":   0,
    "Casing":   0,
    "Length":   1.0,
    "Form":     None,   # special: 'original' → fallback 'interrogative'
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_baseline_mask(df: pd.DataFrame, style: str) -> pd.Series:
    """Boolean mask for the baseline rows of this style."""
    if style == "Form":
        s = df["strength"].astype(str).str.strip().str.lower()
        if (s == "original").any():
            return s == "original"
        return s == "interrogative"
    target = BASELINE_STRENGTH[style]
    try:
        return df["strength"].astype(float) == float(target)
    except (ValueError, TypeError):
        return df["strength"].astype(str) == str(target)


# ── Read checklist ─────────────────────────────────────────────────────────────
if not CHECKLIST_PATH.exists():
    sys.exit(f"[ERROR] Checklist not found: {CHECKLIST_PATH}\n"
             "        Run fill_checklist.py / fill_checklist_closed.py first.")

print(f"\n[INFO] Mode      : {args.mode}")
print(f"[INFO] Checklist : {CHECKLIST_PATH}")

wb = openpyxl.load_workbook(CHECKLIST_PATH, read_only=True, data_only=True)
ws = wb.active

entries = []
cur_model = cur_dataset = None

for r in range(3, ws.max_row + 1):
    v2 = ws.cell(r, 2).value
    v3 = ws.cell(r, 3).value
    v4 = ws.cell(r, 4).value
    v5 = ws.cell(r, 5).value

    if v2 is not None:
        cur_model = str(v2).strip()
    if v3 is not None:
        v3s = str(v3).strip()
        if v3s.startswith("▶"):
            continue
        cur_dataset = v3s

    if v4 is None or str(v4).strip() not in ALL_STYLES:
        continue
    if cur_model is None or cur_dataset is None:
        continue

    style = str(v4).strip()
    path  = str(v5).strip() if v5 is not None else ""
    if not path or path.upper() == "NOT FOUND":
        continue

    entries.append({"path": Path(path), "style": style,
                    "model": cur_model, "dataset": cur_dataset})

wb.close()

# Deduplicate by run-folder path (one folder may serve multiple checklist rows)
unique_runs = {}          # Path -> style
for e in entries:
    if e["path"] not in unique_runs:
        unique_runs[e["path"]] = e["style"]

print(f"[INFO] {len(entries)} checklist rows → {len(unique_runs)} unique run folders")

# Filter already-computed unless --force
if not args.force:
    to_process = {}
    skipped = 0
    for path, style in unique_runs.items():
        csv_path = path / "full_results_all_models.csv"
        if not csv_path.exists():
            print(f"  [WARN] CSV not found: {csv_path}")
            continue
        try:
            header = pd.read_csv(csv_path, nrows=0).columns.tolist()
        except Exception:
            to_process[path] = style
            continue
        if all(c in header for c in NLI_COLS):
            skipped += 1
        else:
            to_process[path] = style
    if skipped:
        print(f"[INFO] Skipping {skipped} already-computed folders "
              f"(use --force to recompute).")
else:
    to_process = {p: s for p, s in unique_runs.items()
                  if (p / "full_results_all_models.csv").exists()}

if not to_process:
    print("[INFO] Nothing to compute. Done.")
    sys.exit(0)

print(f"[INFO] Folders to process: {len(to_process)}\n")

# ── Load NLI model — mirrors utils/models.py pattern ─────────────────────────
# Use device_map="auto" + float16 exactly as the experiment scripts do.
# This lets accelerate handle device placement, avoiding cudaErrorNoKernelImage.
print(f"[INFO] Loading NLI model: {MODEL_NAME}")
tokenizer_nli = AutoTokenizer.from_pretrained(MODEL_NAME)
model_nli = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model_nli.eval()
print(f"[INFO] NLI model device : {model_nli.device}")

# Resolve label indices from model config
id2label    = model_nli.config.id2label
label_lower = {v.lower(): k for k, v in id2label.items()}
CONTRA_IDX  = label_lower.get("contradiction", 0)
ENTAIL_IDX  = label_lower.get("entailment",    1)
print(f"[INFO] Labels: {id2label}")
print(f"[INFO] contradiction idx={CONTRA_IDX}, entailment idx={ENTAIL_IDX}\n")


# ── Main loop ─────────────────────────────────────────────────────────────────
n_updated = 0
n_errors  = 0

outer_bar = tqdm(
    to_process.items(),
    total=len(to_process),
    desc=f"[{args.mode.upper()}] folders",
    unit="folder",
    ncols=110,
    colour="green",
)

for run_path, style in outer_bar:
    csv_path = run_path / "full_results_all_models.csv"
    outer_bar.set_postfix_str(run_path.name[:45])

    # Load CSV
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as exc:
        tqdm.write(f"  [ERR] Cannot read {csv_path}: {exc}")
        n_errors += 1
        continue

    if df.empty:
        tqdm.write(f"  [SKIP] Empty CSV: {csv_path}")
        continue

    df["response_orig"] = df["response_orig"].fillna("").astype(str)
    df["response_pert"] = df["response_pert"].fillna("").astype(str)

    premises   = df["response_orig"].tolist()
    hypotheses = df["response_pert"].tolist()
    n_rows     = len(df)

    # NLI inference — inputs go to model_nli.device (set by device_map="auto")
    contra_arr = np.zeros(n_rows, dtype=np.float32)
    entail_arr = np.zeros(n_rows, dtype=np.float32)

    inner_bar = tqdm(
        range(0, n_rows, args.batch_size),
        total=(n_rows + args.batch_size - 1) // args.batch_size,
        desc=f"  NLI {run_path.name[:35]}",
        unit="batch",
        leave=False,
        ncols=110,
    )

    for start in inner_bar:
        end     = min(start + args.batch_size, n_rows)
        batch_p = premises[start:end]
        batch_h = hypotheses[start:end]

        enc = tokenizer_nli(
            batch_p, batch_h,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        # Move each tensor to wherever device_map placed the model
        enc = {k: v.to(model_nli.device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model_nli(**enc).logits                       # (B, num_labels)
            probs  = torch.softmax(logits.float(), dim=-1).cpu().numpy()

        contra_arr[start:end] = probs[:, CONTRA_IDX]
        entail_arr[start:end] = probs[:, ENTAIL_IDX]

    inner_bar.close()

    df["nli_contradiction"] = contra_arr
    df["nli_entailment"]    = entail_arr

    # Delta computation — per (model, prompt_id, place) group
    df["delta_nli_contradiction"] = np.nan
    df["delta_nli_entailment"]    = np.nan

    group_cols = [c for c in ["model", "prompt_id", "place"] if c in df.columns]

    if not group_cols:
        # No grouping possible — subtract mean baseline
        base_mask = get_baseline_mask(df, style)
        if base_mask.any():
            df["delta_nli_contradiction"] = (
                df["nli_contradiction"] - df.loc[base_mask, "nli_contradiction"].mean()
            )
            df["delta_nli_entailment"] = (
                df["nli_entailment"] - df.loc[base_mask, "nli_entailment"].mean()
            )
    else:
        for group_key, group_idx in df.groupby(group_cols).groups.items():
            grp       = df.loc[group_idx]
            base_mask = get_baseline_mask(grp, style)
            base_rows = grp[base_mask]

            if base_rows.empty:
                tqdm.write(
                    f"  [WARN] No baseline row for group "
                    f"{dict(zip(group_cols, group_key if isinstance(group_key, tuple) else (group_key,)))} "
                    f"in {run_path.name}"
                )
                continue

            base_contra = float(base_rows["nli_contradiction"].iloc[0])
            base_entail = float(base_rows["nli_entailment"].iloc[0])

            df.loc[group_idx, "delta_nli_contradiction"] = (
                df.loc[group_idx, "nli_contradiction"] - base_contra
            )
            df.loc[group_idx, "delta_nli_entailment"] = (
                df.loc[group_idx, "nli_entailment"] - base_entail
            )

    # Save back in-place
    try:
        df.to_csv(csv_path, index=False)
        n_updated += 1
        tqdm.write(
            f"  [OK]  {run_path.name} | rows={n_rows} | style={style} | "
            f"mean_contra={contra_arr.mean():.4f} | mean_entail={entail_arr.mean():.4f}"
        )
    except Exception as exc:
        tqdm.write(f"  [ERR] Cannot write {csv_path}: {exc}")
        n_errors += 1

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print(f"  Mode    : {args.mode}")
print(f"  Updated : {n_updated} run folders")
print(f"  Errors  : {n_errors}")
print(f"  Columns : {NLI_COLS}")
print(f"  Model   : {MODEL_NAME}")
print(f"{'='*64}\n")