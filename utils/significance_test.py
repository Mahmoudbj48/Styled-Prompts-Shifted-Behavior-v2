"""
significance_test.py
--------------------
DEPRECATED -- this entry point is kept for backwards compatibility.

The SGS pipeline (including significance tests against fixed
epsilon thresholds {0.01, 0.05, 0.10}) now lives in
``utils.compute_sgs_table``. Running this file simply forwards to that
driver.

Run from project root:
    python utils/compute_sgs_table.py

Outputs are documented in the driver's module docstring.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

print("[significance_test.py] DEPRECATED -- delegating to compute_sgs_table.main()")
print("                       Run `python utils/compute_sgs_table.py` directly "
      "to avoid this notice.\n")

from compute_sgs_table import main as _main  # noqa: E402

if __name__ == "__main__":
    _main()
