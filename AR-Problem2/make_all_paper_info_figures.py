#!/usr/bin/env python3
"""
Regenerate all Problem 2 paper figures and sync them to all-paper-info/.
Runs: problem2a.py, problem2b_controversy.py, viewership_controversy_analysis.py,
then copies Problem 2a/2b figures from figures/ to all-paper-info/.
Viewership analysis already writes to all-paper-info/.
Uses fan-share data from Data/new_estimate_votes.csv (Problem 1 Base).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
PAPER_DIR = BASE / "all-paper-info"

# Figures to copy from figures/ to all-paper-info/
FIGURES_TO_SYNC = [
    "problem2a_part1_displacement.pdf",
    "problem2a_evolution.pdf",
    "problem2a_combined_evolution_bottom2.pdf",
    "problem2b_regime_controversy_by_type.pdf",
    "problem2b_controversy_cdf.pdf",
    "problem2b_controversy_scatter.pdf",
]


def run_script(name: str, script: str) -> bool:
    """Run a Python script from BASE; return True on success."""
    cmd = [sys.executable, "-u", str(BASE / script)]
    print(f"Running {script} ...")
    result = subprocess.run(cmd, cwd=str(BASE), timeout=300)
    if result.returncode != 0:
        print(f"Warning: {script} exited with code {result.returncode}")
        return False
    return True


def main() -> None:
    print("=" * 60)
    print("Problem 2: Regenerating all-paper-info figures")
    print("=" * 60)

    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Problem 2a (rank vs percent; uses load_fan_shares → new_estimate_votes)
    run_script("Problem 2a", "problem2a.py")

    # 2. Problem 2b (controversy; produces outputs + figures)
    run_script("Problem 2b", "problem2b_controversy.py")

    # 3. Viewership vs controversy (writes directly to all-paper-info)
    run_script("Viewership controversy", "viewership_controversy_analysis.py")

    # 4. Copy 2a/2b figures to all-paper-info
    copied = 0
    for fname in FIGURES_TO_SYNC:
        src = FIG_DIR / fname
        if src.is_file():
            dst = PAPER_DIR / fname
            shutil.copy2(src, dst)
            copied += 1
            print(f"  Copied {fname} → all-paper-info/")
        else:
            print(f"  Skip (missing): {fname}")
    print(f"Synced {copied}/{len(FIGURES_TO_SYNC)} figures to all-paper-info/")
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
