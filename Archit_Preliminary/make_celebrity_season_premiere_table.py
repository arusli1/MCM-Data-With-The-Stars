"""
Build a simple lookup table of (celebrity_name, season, season_premiere_date)
from the provided COMAP dataset `Data/2026_MCM_Problem_C_Data.csv`.

Input:
  - Data/2026_MCM_Problem_C_Data.csv (already in this repo)

Premiere dates:
  Hard-coded from the user's provided list in the prompt (DWTS US).
  Dates are parsed to ISO YYYY-MM-DD.

Output:
  - Archit_Preliminary/celebrity_season_premiere_table.csv
    columns:
      season
      season_premiere_date
      celebrity_name
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


def _parse_us_date(s: str) -> str:
    """Parse dates like 'June 1, 2005' -> '2005-06-01'."""
    d = dt.datetime.strptime(s.strip(), "%B %d, %Y").date()
    return d.isoformat()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    csv_path = repo / "Data" / "2026_MCM_Problem_C_Data.csv"

    df = pd.read_csv(csv_path)

    # Premiere dates provided by user (note: season 31 not present in list)
    premiere_dates_text = {
        1: "June 1, 2005",
        2: "January 5, 2006",
        3: "September 12, 2006",
        4: "March 19, 2007",
        5: "September 24, 2007",
        6: "March 17, 2008",
        7: "September 22, 2008",
        8: "March 9, 2009",
        9: "September 21, 2009",
        10: "March 22, 2010",
        11: "September 20, 2010",
        12: "March 21, 2011",
        13: "September 19, 2011",
        14: "March 19, 2012",
        15: "September 24, 2012",
        16: "March 18, 2013",
        17: "September 16, 2013",
        18: "March 17, 2014",
        19: "September 15, 2014",
        20: "March 16, 2015",
        21: "September 14, 2015",
        22: "March 21, 2016",
        23: "September 12, 2016",
        24: "March 20, 2017",
        25: "September 18, 2017",
        26: "April 30, 2018",
        27: "September 24, 2018",
        28: "September 16, 2019",
        29: "September 14, 2020",
        30: "September 20, 2021",
        31: "September 19, 2022",
        32: "September 26, 2023",
        33: "September 17, 2024",
        34: "September 16, 2025",
    }
    premiere_dates_iso = {k: _parse_us_date(v) for k, v in premiere_dates_text.items()}

    # Detect relevant columns (robust to minor naming differences)
    cols_lower = {c.lower(): c for c in df.columns}
    season_col = cols_lower.get("season")
    name_col = cols_lower.get("celebrity_name") or cols_lower.get("celebrityname")
    if not season_col or not name_col:
        raise ValueError(
            f"Could not find required columns in {csv_path}. "
            f"Found columns: {list(df.columns)}"
        )

    out = (
        df[[season_col, name_col]]
        .rename(columns={season_col: "season", name_col: "celebrity_name"})
        .copy()
    )
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["celebrity_name"] = out["celebrity_name"].astype(str).str.strip()

    out = out.drop_duplicates().sort_values(["season", "celebrity_name"], kind="stable")

    out["season_premiere_date"] = out["season"].map(premiere_dates_iso)

    # Sanity: report missing premiere dates (if any)
    missing = out[out["season_premiere_date"].isna()]["season"].value_counts().sort_index()
    if len(missing):
        print("WARNING: Missing premiere date for some seasons (count of contestants):")
        print(missing.to_string())

    out = out[["season", "season_premiere_date", "celebrity_name"]]

    out_path = repo / "Archit_Preliminary" / "celebrity_season_premiere_table.csv"
    out.to_csv(out_path, index=False)

    print(f"Wrote {len(out)} rows to: {out_path}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()


