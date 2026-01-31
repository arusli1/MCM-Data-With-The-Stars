import argparse
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import requests

DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1_Bayes/popularity_wiki_edits_season.csv"

USER_AGENT = "DWTSWikiEdits/1.0 (educational; contact: local)"
SLEEP_SECONDS = 0.5
CHECKPOINT_EVERY = 25


def request_with_backoff(url: str, params: Dict) -> Optional[dict]:
    for attempt in range(6):
        resp = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
        if resp.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        if resp.status_code in (400, 404):
            return None
        resp.raise_for_status()
        return resp.json()
    return None


def wiki_search(name: str) -> Optional[str]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": name,
        "format": "json",
        "srlimit": 1,
    }
    data = request_with_backoff(url, params)
    if not data:
        return None
    results = data.get("query", {}).get("search", [])
    if not results:
        return None
    return results[0]["title"]


def count_revisions_total(title: str, end_time: datetime) -> Optional[int]:
    url = "https://en.wikipedia.org/w/api.php"
    total = 0
    cont = None
    while True:
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "format": "json",
            "rvstart": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rvend": "2001-01-01T00:00:00Z",
            "rvlimit": "max",
            "rvprop": "ids",
            "rvdir": "older",
        }
        if cont:
            params["rvcontinue"] = cont
        data = request_with_backoff(url, params)
        if not data:
            return None
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        revs = page.get("revisions", [])
        total += len(revs)
        cont = data.get("continue", {}).get("rvcontinue")
        if not cont:
            break
        time.sleep(0.1)
    return total


def season_start_dates() -> Dict[int, datetime]:
    return {
        1: datetime(2005, 6, 1),
        2: datetime(2006, 1, 1),
        3: datetime(2006, 9, 1),
        4: datetime(2007, 3, 1),
        5: datetime(2007, 9, 1),
        6: datetime(2008, 3, 1),
        7: datetime(2008, 9, 1),
        8: datetime(2009, 3, 1),
        9: datetime(2009, 9, 1),
        10: datetime(2010, 3, 1),
        11: datetime(2010, 9, 1),
        12: datetime(2011, 3, 1),
        13: datetime(2011, 9, 1),
        14: datetime(2012, 3, 1),
        15: datetime(2012, 9, 1),
        16: datetime(2013, 3, 1),
        17: datetime(2013, 9, 1),
        18: datetime(2014, 3, 1),
        19: datetime(2014, 9, 1),
        20: datetime(2015, 3, 1),
        21: datetime(2015, 9, 1),
        22: datetime(2016, 3, 1),
        23: datetime(2016, 9, 1),
        24: datetime(2017, 3, 1),
        25: datetime(2017, 9, 1),
        26: datetime(2018, 4, 1),
        27: datetime(2018, 9, 1),
        28: datetime(2019, 9, 1),
        29: datetime(2020, 9, 1),
        30: datetime(2021, 9, 1),
        31: datetime(2022, 9, 1),
        32: datetime(2023, 9, 1),
        33: datetime(2024, 9, 1),
        34: datetime(2025, 9, 1),
    }


def season_max_week(df_season: pd.DataFrame) -> int:
    max_week = 0
    for w in range(1, 12):
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
        cols = [c for c in cols if c in df_season.columns]
        if not cols:
            continue
        vals = df_season[cols].replace("N/A", pd.NA)
        numeric = vals.apply(pd.to_numeric, errors="coerce")
        if numeric.notna().any().any():
            max_week = w
    return max_week


def fetch_popularity(df: pd.DataFrame, existing: Dict[Tuple[int, str], Dict]) -> pd.DataFrame:
    rows = []
    season_starts = season_start_dates()
    done = 0
    total = 0
    for season, df_season in df.groupby("season"):
        total += len(df_season["celebrity_name"].unique())
    for season, df_season in df.groupby("season"):
        season = int(season)
        if season not in season_starts:
            continue
        start = season_starts[season]
        max_week = season_max_week(df_season)
        if max_week <= 0:
            max_week = 11
        end = start + timedelta(days=7 * (max_week - 1))
        for name in sorted(df_season["celebrity_name"].unique().tolist()):
            key = (season, name)
            if key in existing:
                rows.append(existing[key])
                continue
            title = wiki_search(name)
            edits_end = None
            edits_start = None
            if title:
                edits_end = count_revisions_total(title, end)
                edits_start = count_revisions_total(title, start)
            edits_end_val = edits_end if edits_end is not None else 0
            edits_start_val = edits_start if edits_start is not None else 0
            rows.append(
                {
                    "season": season,
                    "celebrity_name": name,
                    "wiki_title": title,
                    "pop_score": edits_end_val,
                    "pop_score_end": edits_end_val,
                    "pop_score_start": edits_start_val,
                    "pop_score_delta": max(0, edits_end_val - edits_start_val),
                    "source": "wikipedia_edits_cumulative",
                    "window_start": start.strftime("%Y-%m-%d"),
                    "window_end": end.strftime("%Y-%m-%d"),
                }
            )
            done += 1
            if done % CHECKPOINT_EVERY == 0:
                pd.DataFrame(list(existing.values()) + rows).to_csv(OUT_PATH, index=False)
                print(f"Checkpointed {done}/{total}")
            time.sleep(SLEEP_SECONDS)
    rows = list(existing.values()) + rows
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute all values")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH)
    existing: Dict[Tuple[int, str], Dict] = {}
    if os.path.exists(OUT_PATH) and not args.force:
        prev = pd.read_csv(OUT_PATH)
        for _, row in prev.iterrows():
            existing[(int(row["season"]), row["celebrity_name"])] = row.to_dict()
    pop_df = fetch_popularity(df, existing)
    pop_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
