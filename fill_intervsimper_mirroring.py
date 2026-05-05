"""
fill_intervsimper_mirroring.py
------------------------------
Run from project root:
    python fill_intervsimper_mirroring.py

Sets mirroring_rate = 0 for every row in all
results/inter_vs_imper/*/full_results_all_models.csv files.
"""

import pandas as pd
from pathlib import Path

pattern = Path("results/inter_vs_imper").glob("*/full_results_all_models.csv")
files   = sorted(pattern)

if not files:
    print("[WARN] No files found under results/inter_vs_imper/")
else:
    for csv_path in files:
        df = pd.read_csv(csv_path, low_memory=False)
        df["mirroring_rate"] = 0
        df.to_csv(csv_path, index=False)
        print(f"[OK]  {csv_path}  ({len(df):,} rows)")

    print(f"\nDone — updated {len(files)} files.")
