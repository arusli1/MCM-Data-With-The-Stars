"""
Fetch DWTS viewership (viewers in millions) from Wikipedia season pages.
Each season page has a Ratings section with viewer counts per episode/week.
Output: Data/dwts_viewership.csv (season, mean_viewers, n_episodes, source)
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


def extract_viewers_from_text(text: str) -> List[float]:
    """Extract viewer numbers (millions) from text. Handles formats like 13.48, 15.09, 22.36."""
    # Match numbers that look like millions: 1-2 digits, decimal, 2 digits (e.g. 13.48, 7.68)
    # Also integers like 21 (Season 11 style)
    matches = re.findall(r"\b(\d{1,2}\.\d{2}|\d{2})\b", str(text))
    out = []
    for m in matches:
        try:
            v = float(m)
            # Plausible range for US primetime: 3-25 million
            if 2 <= v <= 30:
                out.append(v)
        except ValueError:
            pass
    return out


def fetch_season_viewership(season: int) -> Tuple[Optional[float], int, str]:
    """
    Fetch viewership for one season. Returns (mean_viewers, n_episodes, note).
    Tries pd.read_html (lxml) first; falls back to regex on HTML.
    """
    url = BASE_URL.format(season)
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return None, 0, f"fetch failed: {e}"

    candidates: List[float] = []

    # 1) Try pd.read_html with lxml
    try:
        from io import StringIO
        tables = pd.read_html(StringIO(html), attrs={"class": "wikitable"}, flavor="lxml")
        for tbl in tables:
            for c in tbl.columns:
                for v in tbl[c].dropna().astype(str):
                    nums = extract_viewers_from_text(v)
                    candidates.extend(nums)
    except Exception:
        pass

    # 2) Fallback: regex on HTML. Viewer numbers 4-25 million, format X.XX or XX
    if not candidates:
        matches = re.findall(r"(\d{1,2}\.\d{2})", html)
        for m in matches:
            try:
                v = float(m)
                if 4 <= v <= 25:
                    candidates.append(v)
            except ValueError:
                pass
        # Also integers (Season 11: 21, 18)
        int_matches = re.findall(r"[>\s](\d{2})[<\s,\[]", html)
        for m in int_matches:
            try:
                v = float(m)
                if 10 <= v <= 25:
                    candidates.append(v)
            except ValueError:
                pass

    if not candidates:
        return None, 0, "no viewer numbers found"
    # Dedupe and take reasonable set - often repeats from multiple columns
    seen: set = set()
    unique: List[float] = []
    for v in sorted(candidates, reverse=True):
        k = round(v, 2)
        if k not in seen:
            seen.add(k)
            unique.append(v)
    # If too many, take first N (likely the main viewer column)
    if len(unique) > 20:
        unique = unique[:15]
    mean_v = sum(unique) / len(unique)
    return round(mean_v, 2), len(unique), "ok"


def main():
    out_path = Path(__file__).resolve().parents[1] / "Data" / "dwts_viewership.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for season in SEASONS:
        mean_v, n, note = fetch_season_viewership(season)
        rows.append({"season": season, "mean_viewers": mean_v, "n_episodes": n, "note": note})
        print(f"  Season {season}: mean={mean_v}, n={n} ({note})")
        time.sleep(0.5)  # be polite to Wikipedia

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    valid = df[df["mean_viewers"].notna()]
    print(f"Got viewership for {len(valid)}/{len(df)} seasons")


if __name__ == "__main__":
    main()
