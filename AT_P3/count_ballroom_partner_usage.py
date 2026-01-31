"""
Tally ballroom partner usage counts from Data/2026_MCM_Problem_C_Data.csv.

This prints:
  1) Exact-string counts (after whitespace normalization)
  2) Optional "split" counts, where partner strings are split on '/', '&', or ' and '

Run from repo root:
  python3 Archit_Preliminary/count_ballroom_partner_usage.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _clean(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ")  # NBSP -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "Data" / "2026_MCM_Problem_C_Data.csv"
    df = pd.read_csv(path)

    col = "ballroom_partner"
    if col not in df.columns:
        raise SystemExit(f"Column not found: {col}")

    partners = df[col].map(_clean)

    exact = partners.value_counts().reset_index()
    exact.columns = ["ballroom_partner", "count"]

    print("=== Ballroom partner usage (exact string, normalized) ===")
    print(exact.to_string(index=False))

    # Also compute a split-token view (helpful when cells contain multiple partners).
    toks = []
    for v in partners:
        if not v:
            continue
        for p in re.split(r"\s*/\s*|\s*&\s*|\s+and\s+", v):
            p = p.strip()
            if p:
                toks.append(p)

    split = pd.Series(toks).value_counts().reset_index()
    split.columns = ["ballroom_partner_token", "count"]

    print("\n=== Ballroom partner usage (split tokens) ===")
    print(split.to_string(index=False))


if __name__ == "__main__":
    main()


