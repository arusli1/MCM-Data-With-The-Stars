import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from pytrends.request import TrendReq
    from pytrends.exceptions import TooManyRequestsError
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install pytrends") from exc


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1_Bayes/popularity_trends_season.csv"

# Use stable anchor terms to normalize Google Trends scores across requests.
ANCHOR_TERMS = ["Barack Obama", "YouTube", "Facebook"]
TIMEFRAME = "all"  # 2004-present
GEO = "US"
CHECKPOINT_EVERY = 20
SLEEP_SECONDS = 3.0
KW_PER_REQUEST = 1
MIN_SCORE = 1e-4
SEASON_PAUSE_SECONDS = 30
JITTER_SECONDS = 1.0


def trend_timeseries_batch(
    trends: TrendReq, keywords: List[str], timeframe: str, retries: int = 8
) -> Optional[pd.DataFrame]:
    kws = [k for k in keywords if k]
    if not kws:
        return None
    for attempt in range(retries):
        try:
            trends.build_payload(kws + [ANCHOR_TERMS[0]], timeframe=timeframe, geo=GEO)
            df = trends.interest_over_time()
        except TooManyRequestsError:
            wait = 90 * (attempt + 1)
            print(f"Rate limited, sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        except Exception:
            wait = 15 * (attempt + 1)
            print(f"Request error, sleeping {wait}s", flush=True)
            time.sleep(wait)
            continue
        if df is None or df.empty:
            return None
        if ANCHOR_TERMS[0] not in df.columns:
            return None
        return df
    return None


def trend_timeseries_with_anchor(
    trends: TrendReq, keywords: List[str], timeframe: str
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    for anchor in ANCHOR_TERMS:
        df = trend_timeseries_batch(trends, keywords, timeframe)
        if df is not None and anchor in df.columns:
            return df, anchor
    return None, None


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


def parse_week_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("week") and "_judge" in c]


def week_score(df: pd.DataFrame, week: int, cols: List[str]) -> pd.Series:
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def active_by_week(df_season: pd.DataFrame, week_cols: List[str]) -> Dict[int, List[str]]:
    max_week = max(int(c.split("_")[0].replace("week", "")) for c in week_cols)
    active = {}
    for w in range(1, max_week + 1):
        totals = week_score(df_season, w, week_cols)
        names = df_season.loc[totals > 0, "celebrity_name"].tolist()
        if names:
            active[w] = names
    return active


def fetch_popularity(df: pd.DataFrame, existing: Dict[Tuple[int, str], Dict]) -> pd.DataFrame:
    trends = TrendReq(hl="en-US", tz=360, timeout=(5, 20))
    rows = []
    week_cols = parse_week_cols(df)
    season_starts = season_start_dates()
    total_tasks = 0
    for season, df_season in df.groupby("season"):
        season = int(season)
        if season not in season_starts:
            continue
        active = active_by_week(df_season, week_cols)
        total_tasks += len(df_season["celebrity_name"].unique().tolist())
    done = 0
    for season, df_season in df.groupby("season"):
        season = int(season)
        if season not in season_starts:
            continue
        print(f"Season {season}: start", flush=True)
        active = active_by_week(df_season, week_cols)
        start = season_starts[season]
        max_week = max(active.keys())
        season_start = start
        season_end = start + timedelta(days=7 * max_week - 1)
        timeframe = f"{season_start.strftime('%Y-%m-%d')} {season_end.strftime('%Y-%m-%d')}"
        unique_names = df_season["celebrity_name"].unique().tolist()
        total_batches = max(1, (len(unique_names) + KW_PER_REQUEST - 1) // KW_PER_REQUEST)
        for i in range(0, len(unique_names), KW_PER_REQUEST):
            batch = unique_names[i : i + KW_PER_REQUEST]
            print(f"Season {season}: batch {i//KW_PER_REQUEST+1}/{total_batches} start", flush=True)
            df_ts, anchor_term = trend_timeseries_with_anchor(trends, batch, timeframe)
            if df_ts is None or anchor_term is None:
                # Retry as single-keyword requests to reduce 429/missing
                df_ts = None
                single_ok = []
                for name in batch:
                    single, single_anchor = trend_timeseries_with_anchor(trends, [name], timeframe)
                    if single is not None and single_anchor is not None:
                        single_ok.append((name, single, single_anchor))
                if not single_ok:
                    print(f"Season {season}: batch {i//KW_PER_REQUEST+1}/{total_batches} missing data")
                    continue
                # Merge singles into one dataframe with anchor from first
                anchor_term = single_ok[0][2]
                df_ts = single_ok[0][1][[anchor_term]].copy()
                for name, single, _ in single_ok:
                    df_ts[name] = single[name]
            anchor = df_ts[anchor_term].replace(0, np.nan)
            anchor_w = anchor
            if anchor_w.isna().all():
                continue
            for name in unique_names:
                if name not in df_ts.columns:
                    continue
                key = (season, name)
                if key in existing:
                    continue
                series = df_ts[name].replace(0, np.nan)
                if series.isna().all():
                    score = None
                else:
                    ratio = (series / anchor_w).dropna()
                    score = float(ratio.mean()) if not ratio.empty else None
                rows.append(
                    {
                        "season": season,
                        "celebrity_name": name,
                        "pop_score": score if score is not None else np.nan,
                        "source": "google_trends_anchor_season",
                        "anchor_term": anchor_term,
                        "timeframe": timeframe,
                        "geo": GEO,
                    }
                )
                done += 1
                if done % CHECKPOINT_EVERY == 0:
                    pd.DataFrame(list(existing.values()) + rows).to_csv(OUT_PATH, index=False)
                    print(f"Checkpointed {done}/{total_tasks}")
            pd.DataFrame(list(existing.values()) + rows).to_csv(OUT_PATH, index=False)
            print(f"Season {season}: batch {i//KW_PER_REQUEST+1}/{total_batches} saved")
            time.sleep(SLEEP_SECONDS + np.random.rand() * JITTER_SECONDS)
        time.sleep(SEASON_PAUSE_SECONDS)
        # Fill missing per season/celebrity with season median when possible
        if rows:
            tmp = pd.DataFrame(rows)
            for name in tmp["celebrity_name"].unique():
                mask = tmp["celebrity_name"] == name
                med = tmp.loc[mask, "pop_score"].median()
                if np.isnan(med) or med <= 0:
                    med = MIN_SCORE
                tmp.loc[mask, "pop_score"] = tmp.loc[mask, "pop_score"].fillna(med)
                tmp.loc[mask & (tmp["pop_score"] <= 0), "pop_score"] = MIN_SCORE
            rows = tmp.to_dict(orient="records")
    rows = list(existing.values()) + rows
    return pd.DataFrame(rows)


def main():
    warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
    warnings.filterwarnings("ignore", message=".*Downcasting object dtype arrays.*")
    df = pd.read_csv(DATA_PATH)
    existing: Dict[Tuple[int, str], Dict] = {}
    if os.path.exists(OUT_PATH):
        prev = pd.read_csv(OUT_PATH)
        for _, row in prev.iterrows():
            existing[(int(row["season"]), row["celebrity_name"])] = row.to_dict()
    pop_df = fetch_popularity(df, existing)
    pop_df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
