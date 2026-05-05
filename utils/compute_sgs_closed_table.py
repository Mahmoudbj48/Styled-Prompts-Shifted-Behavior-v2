"""
compute_sgs_closed_table.py
---------------------------
DEPRECATED -- this entry point is kept for backwards compatibility.

The SGS pipeline now handles open and closed models in a single pass.
Run ``utils/compute_sgs_table.py`` instead; it produces the closed-only
LaTeX table at ``results/closed_models/sgs_closed_table.tex`` along with
all other outputs.

Run from project root:
    python utils/compute_sgs_table.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

print("[compute_sgs_closed_table.py] DEPRECATED -- delegating to compute_sgs_table.main()")
print("                              Run `python utils/compute_sgs_table.py` directly "
      "to avoid this notice.\n")

from compute_sgs_table import main as _main  # noqa: E402

if __name__ == "__main__":
    _main()
