"""
Fetch DWTS viewership (viewers in millions) from Wikipedia season pages.
Each season page has a Ratings section with viewer counts per episode/week.
Outputs:
  - Data/dwts_viewership.csv (season, mean_viewers, n_episodes, note)
  - Data/dwts_viewership_weekly.csv (season, ep_no, viewers_millions, title, show_type)
"""
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

USER_AGENT = "DWTSViewership/1.0 (MCM; educational; https://github.com/)"
BASE_URL = "https://en.wikipedia.org/wiki/Dancing_with_the_Stars_(American_TV_series)_season_{}"

# Seasons 1-34 in our data (Problem C data)
SEASONS = list(range(1, 35))

# Seasons without Wikipedia ratings are left as missing (no fallback).


def extract_viewers_from_text(text: str) -> List[float]:
    """Extract viewer numbers (millions) from text. Strips Wikipedia citations [1], [2]."""
    s = re.sub(r"\[\d+\]", "", str(text))  # strip [1], [2], etc.
    out = []
    # Nielsen format: 7.68, 13.48, 21.341 - decimal with 2–3 places
    for m in re.findall(r"\b(\d{1,2}\.\d{2,3})\b", s):
        try:
            v = float(m)
            if 3 <= v <= 30:
                out.append(v)
        except ValueError:
            pass
    if not out:
        for m in re.findall(r"\b(\d{1,2}\.\d{2})\b", s):
            try:
                v = float(m)
                if 3 <= v <= 30:
                    out.append(v)
            except ValueError:
                pass
    # Integer format: 21, 18 - only if no decimals found
    if not out:
        for m in re.findall(r"\b(\d{2})\b", s):
            try:
                v = float(m)
                if 10 <= v <= 25:
                    out.append(v)
            except ValueError:
                pass
    return out


def _find_viewer_columns(tbl) -> List[int]:
    """
    Find columns that contain viewership data. Handles two formats:
    - New: single "Viewers (millions)" column (exclude DVR, Total)
    - Old: "Performance Show" / "Results Show" under "Viewers (in millions)"
    """
    cols = [str(c).lower() for c in tbl.columns]
    viewer_cols = []
    for i, c in enumerate(cols):
        if "viewers (millions)" in c and "dvr" not in c and "total" not in c:
            viewer_cols.append(i)
            break  # prefer single column format
    if viewer_cols:
        return viewer_cols
    # Fallback: Performance Show / Results Show (older seasons, nested under Viewers in millions)
    if _table_looks_like_ratings(tbl):
        for i, c in enumerate(cols):
            if "performance show" in c or "results show" in c:
                viewer_cols.append(i)
    return viewer_cols


def _table_looks_like_ratings(tbl) -> bool:
    """Heuristic: ratings tables have Week column and 4-25 rows."""
    cols = [str(c).lower() for c in tbl.columns]
    has_week = any("week" in c for c in cols)
    return has_week and 4 <= len(tbl) <= 25


def _get_ratings_table_viewers(tables: list) -> List[float]:
    """
    Find the Ratings table and extract viewer numbers.
    Handles both "Viewers (millions)" and "Performance Show"/"Results Show" formats.
    """
    candidates: List[float] = []
    for tbl in tables:
        viewer_cols = _find_viewer_columns(tbl)
        if not viewer_cols or len(tbl) < 2 or len(tbl) > 25:
            continue
        for vi in viewer_cols:
            for v in tbl.iloc[:, vi].dropna().astype(str):
                nums = extract_viewers_from_text(v)
                candidates.extend(nums)
        if candidates:
            break
    return candidates


def fetch_season_viewership(season: int) -> Tuple[Optional[float], int, str]:
    """
    Fetch viewership for one season. Returns (mean_viewers, n_episodes, note).
    Only uses the Ratings table Viewers column (avoids judge-score contamination).
    """
    url = BASE_URL.format(season)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return None, 0, f"fetch failed: {e}"

    candidates: List[float] = []
    try:
        from io import StringIO
        tables = pd.read_html(StringIO(html), flavor="lxml")
        candidates = _get_ratings_table_viewers(tables)
    except Exception:
        pass

    if not candidates:
        return None, 0, "no viewer numbers found"
    mean_v = sum(candidates) / len(candidates)
    return round(mean_v, 2), len(candidates), "ok"


def fetch_season_viewership_weekly(season: int) -> List[dict]:
    """
    Fetch per-episode viewership in order. Returns list of {season, ep_no, viewers_millions}.
    """
    url = BASE_URL.format(season)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return []

    rows: List[dict] = []
    try:
        from io import StringIO
        tables = pd.read_html(StringIO(html), flavor="lxml")
    except Exception:
        return []

    for tbl in tables:
        viewer_cols = _find_viewer_columns(tbl)
        if not viewer_cols or len(tbl) < 3 or len(tbl) > 25:
            continue
        ep_no = 0
        for idx, row in tbl.iterrows():
            for vi in viewer_cols:
                nums = extract_viewers_from_text(str(row.iloc[vi]))
                for n in nums:
                    if 3 <= n <= 30:
                        ep_no += 1
                        rows.append({"season": season, "ep_no": ep_no, "viewers_millions": round(n, 2)})
        if len(rows) >= 3:
            break

    return rows


def main():
    data_dir = Path(__file__).resolve().parents[1] / "Data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "dwts_viewership.csv"
    weekly_path = data_dir / "dwts_viewership_weekly.csv"

    rows = []
    weekly_rows = []
    for season in SEASONS:
        mean_v, n, note = fetch_season_viewership(season)
        rows.append({"season": season, "mean_viewers": mean_v, "n_episodes": n, "note": note})
        w = fetch_season_viewership_weekly(season)
        weekly_rows.extend(w)
        print(f"  Season {season}: mean={mean_v}, n={n}, weekly_eps={len(w)} ({note})")
        time.sleep(0.5)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    if weekly_rows:
        pd.DataFrame(weekly_rows).to_csv(weekly_path, index=False)
        print(f"\nSaved {out_path}, {weekly_path} ({len(weekly_rows)} weekly rows)")
    else:
        print(f"\nSaved {out_path}")
    valid = [r for r in rows if r["mean_viewers"] is not None]
    print(f"Got viewership for {len(valid)}/{len(rows)} seasons")


if __name__ == "__main__":
    main()
