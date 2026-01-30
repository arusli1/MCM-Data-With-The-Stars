import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_week_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df: pd.DataFrame, week: int, cols: List[str]) -> pd.Series:
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def elimination_week(row: pd.Series, last_active_week: int) -> Optional[int]:
    if isinstance(row["results"], str) and "Eliminated Week" in row["results"]:
        return int(row["results"].split("Eliminated Week ")[1])
    if isinstance(row["results"], str) and "Withdrew" in row["results"]:
        return last_active_week
    return None


def regime_for_season(season: int) -> str:
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom"


def build_season_struct(df_season: pd.DataFrame, week_cols: List[str]) -> Dict:
    max_week = max(int(re.search(r"week(\d+)_", c).group(1)) for c in week_cols)
    names = df_season["celebrity_name"].tolist()
    n = len(names)
    J = np.zeros((max_week, n), dtype=float)

    for w in range(1, max_week + 1):
        J[w - 1] = week_score(df_season, w, week_cols).to_numpy()

    week_idx = np.arange(1, max_week + 1)[:, None]
    last_active = (np.where(J > 0, week_idx, 0)).max(axis=0)
    elim_week = []
    for i, row in df_season.iterrows():
        la = int(last_active[df_season.index.get_loc(i)])
        elim_week.append(elimination_week(row, la))

    max_week_active = int(np.where(J.sum(axis=1) > 0)[0].max() + 1)
    return {
        "names": names,
        "J": J[:max_week_active],
        "elim_week": elim_week,
        "max_week": max_week_active,
    }


def popularity_prior(
    df_season: pd.DataFrame, idx: List[int], gamma_pop: float, noise_scale: float, rng: np.random.Generator
) -> Dict[int, float]:
    week_cols = parse_week_cols(df_season)
    weeks = sorted({int(re.search(r"week(\d+)_", c).group(1)) for c in week_cols})
    if not weeks:
        return {i: 1.0 / len(idx) for i in idx}

    judge_matrix = np.vstack(
        [week_score(df_season, w, week_cols).to_numpy() for w in weeks]
    )
    pos_mask = judge_matrix > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_judge = np.where(
            pos_mask.any(axis=0),
            judge_matrix.sum(axis=0) / pos_mask.sum(axis=0),
            np.nan,
        )

    # Actual elimination week = last week with positive judge scores.
    actual_week = np.where(pos_mask.any(axis=0), pos_mask.sum(axis=0), np.nan)

    valid = [
        i for i in idx if not np.isnan(avg_judge[i]) and not np.isnan(actual_week[i])
    ]
    if not valid:
        return {i: 1.0 / len(idx) for i in idx}

    x = avg_judge[valid].astype(float)
    y = actual_week[valid].astype(float)
    if np.unique(x).size < 2:
        pred = np.full_like(y, y.mean(), dtype=float)
    else:
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept

    pop_score = y - pred
    if noise_scale > 0:
        pop_score = pop_score + rng.normal(0.0, noise_scale, size=pop_score.shape[0])
    scale = pop_score.std()
    if scale > 0:
        pop_score = pop_score / scale
    weights = np.exp(gamma_pop * pop_score)

    pop = {i: 0.0 for i in idx}
    for k, i in enumerate(valid):
        pop[i] = weights[k]
    total = sum(pop.values())
    if total > 0:
        pop = {i: pop[i] / total for i in idx}
    else:
        pop = {i: 1.0 / len(idx) for i in idx}
    return pop


def blended_target(
    df_season: pd.DataFrame,
    idx: List[int],
    last_shares: Optional[Dict[int, float]],
    alpha_prev: float,
    gamma_pop: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> Dict[int, float]:
    pop_dist = popularity_prior(df_season, idx, gamma_pop, noise_scale, rng)
    if last_shares is not None:
        target = {
            i: alpha_prev * last_shares.get(i, 0.0) + (1 - alpha_prev) * pop_dist[i]
            for i in idx
        }
    else:
        target = {i: pop_dist[i] for i in idx}
    total = sum(target.values())
    if total > 0:
        return {i: target[i] / total for i in idx}
    return {i: 1.0 / len(idx) for i in idx}
