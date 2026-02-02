"""
Shared utilities for Problem 2: compare rank vs percent combination methods,
and controversy (judge-fan disagreement) analysis.

Forward simulation: phantom survivors use zeros. See simulation_divergence_limitation.md.
"""
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "2026_MCM_Problem_C_Data.csv")
# Prefer new Base model output (Data/new_estimate_votes.csv), then legacy paths.
NEW_ESTIMATE_VOTES_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "new_estimate_votes.csv")
FAN_SHARES_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "estimate_votes.csv")
BASE_SHARES_PATH = os.path.join(os.path.dirname(__file__), "..", "AR-Problem1-Base", "base_results", "base_inferred_shares.csv")


def parse_week_cols(df: pd.DataFrame) -> List[int]:
    weeks = set()
    for c in df.columns:
        m = re.match(r"week(\d+)_judge\d+_score", c)
        if m:
            weeks.add(int(m.group(1)))
    return sorted(weeks)


def season_max_week(df_season: pd.DataFrame, weeks: List[int]) -> int:
    max_week = 0
    for w in weeks:
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
        cols = [c for c in cols if c in df_season.columns]
        vals = df_season[cols].replace("N/A", pd.NA)
        numeric = vals.apply(pd.to_numeric, errors="coerce")
        if numeric.notna().any().any():
            max_week = w
    return max_week


def build_judge_matrix(df_season: pd.DataFrame, weeks: List[int]) -> np.ndarray:
    max_week = season_max_week(df_season, weeks)
    names = df_season["celebrity_name"].tolist()
    W, N = max_week, len(names)
    J = np.zeros((W, N), dtype=float)
    for w in range(1, W + 1):
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
        cols = [c for c in cols if c in df_season.columns]
        vals = df_season[cols].replace("N/A", pd.NA)
        numeric = vals.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        J[w - 1] = numeric.to_numpy().sum(axis=1)
    return J


def rank_order(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    idx = np.where(active)[0]
    order = idx[np.argsort(values[idx])]
    ranks = np.zeros_like(values, dtype=float)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    return ranks


def judge_pct(J_w: np.ndarray, active: np.ndarray) -> np.ndarray:
    s = J_w.copy()
    s[~active] = 0.0
    t = s.sum()
    return s / t if t > 0 else np.zeros_like(s)


def season_regime(season: int) -> str:
    if season <= 2:
        return "rank"
    if 3 <= season <= 27:
        return "percent"
    return "rank_bottom2"


def load_fan_shares_for_season(season: int, names: List[str], W: int) -> Optional[np.ndarray]:
    """Load s_hist (W x N). Prefer Data/new_estimate_votes.csv (latest Base run), then estimate_votes, then base_inferred_shares."""
    for path, col in [(NEW_ESTIMATE_VOTES_PATH, "s_share"), (FAN_SHARES_PATH, "s_share"), (BASE_SHARES_PATH, "s_share")]:
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        df = df[df["season"] == season]
        if df.empty:
            continue
        name_to_i = {n: i for i, n in enumerate(names)}
        s_hist = np.zeros((W, len(names)))
        for _, row in df.iterrows():
            w = int(row["week"])
            name = row["celebrity_name"]
            if w <= W and name in name_to_i:
                s_hist[w - 1, name_to_i[name]] = row[col]
        for w in range(W):
            tot = s_hist[w].sum()
            if tot > 0:
                s_hist[w] /= tot
        return s_hist
    return None


def compute_elim_week_from_judge(J: np.ndarray) -> List[int]:
    """Infer elimination week from judge scores (0 = not competing)."""
    W, N = J.shape
    elim = [W + 1] * N
    for i in range(N):
        for w in range(W):
            if J[w, i] <= 0 and (w == 0 or J[w - 1, i] > 0):
                elim[i] = w + 1
                break
    return elim


def elim_schedule_from_judge(J: np.ndarray) -> List[int]:
    """Number eliminated each week from judge-based elim week."""
    elim_judge = compute_elim_week_from_judge(J)
    W = J.shape[0]
    schedule = [0] * (W + 1)
    for ew in elim_judge:
        if ew <= W:
            schedule[ew] += 1
    return schedule


def forward_simulate_judge_only(J: np.ndarray, schedule: List[int]) -> Tuple[List[int], List[int]]:
    """Simulate elimination by judge total only. Returns (elim_week, placement)."""
    W, N = J.shape
    elim_week = [W + 1] * N
    for w in range(W):
        k = schedule[w + 1] if w + 1 < len(schedule) else 0
        if k <= 0:
            continue
        active = np.array([elim_week[i] > w + 1 for i in range(N)])
        active_idx = np.where(active)[0]
        if len(active_idx) <= k:
            for i in active_idx:
                elim_week[i] = w + 1
            continue
        J_w = J[w]
        order = active_idx[np.argsort(J_w[active_idx])]
        elim_idx = order[:k]
        for i in elim_idx:
            elim_week[i] = w + 1
    remaining = np.array([elim_week[i] == W + 1 for i in range(N)])
    final_J = J[W - 1].copy()
    final_J[~remaining] = -np.inf
    order = np.lexsort((-final_J, -np.array(elim_week)))
    placement = [0] * N
    for r, i in enumerate(order, start=1):
        placement[i] = r
    return elim_week, placement


def forward_simulate_fan_only(s_hist: np.ndarray, schedule: List[int]) -> Tuple[List[int], List[int]]:
    """Simulate elimination by fan share only. Returns (elim_week, placement)."""
    W, N = s_hist.shape
    elim_week = [W + 1] * N
    for w in range(W):
        k = schedule[w + 1] if w + 1 < len(schedule) else 0
        if k <= 0:
            continue
        active = np.array([elim_week[i] > w + 1 for i in range(N)])
        active_idx = np.where(active)[0]
        if len(active_idx) <= k:
            for i in active_idx:
                elim_week[i] = w + 1
            continue
        s_w = s_hist[w]
        order = active_idx[np.argsort(s_w[active_idx])]
        elim_idx = order[:k]
        for i in elim_idx:
            elim_week[i] = w + 1
    remaining = np.array([elim_week[i] == W + 1 for i in range(N)])
    final_s = s_hist[W - 1].copy()
    final_s[~remaining] = -np.inf
    order = np.lexsort((-final_s, -np.array(elim_week)))
    placement = [0] * N
    for r, i in enumerate(order, start=1):
        placement[i] = r
    return elim_week, placement


def forward_simulate_simple(
    J: np.ndarray,
    s_hist: np.ndarray,
    schedule: List[int],
    regime: str = "rank"
) -> Tuple[List[int], List[int]]:
    """Simple forward simulation without bottom-two logic. Phantom survivors use zeros."""
    W, N = J.shape
    elim_week = [W + 1] * N

    for w in range(W):
        k = schedule[w + 1] if w + 1 < len(schedule) else 0
        if k <= 0:
            continue
        active = np.array([elim_week[i] > w + 1 for i in range(N)])
        active_idx = np.where(active)[0]
        if len(active_idx) <= k:
            for i in active_idx:
                elim_week[i] = w + 1
            continue

        J_w = J[w]
        s_w = s_hist[w]

        if regime == "percent":
            jp = judge_pct(J_w, active)
            combined = jp + s_w
            order = active_idx[np.argsort(combined[active_idx])]
            elim_idx = order[:k]
        else:
            rJ = rank_order(-J_w, active)
            rF = rank_order(-s_w, active)
            R = rJ + rF
            order = active_idx[np.argsort(R[active_idx])]
            elim_idx = order[-k:]

        for i in elim_idx:
            elim_week[i] = w + 1

    remaining = np.array([elim_week[i] == W + 1 for i in range(N)])
    final_scores = np.full(N, -np.inf)

    if regime == "percent":
        jp = judge_pct(J[W - 1], remaining)
        final_scores[remaining] = jp[remaining] + s_hist[W - 1][remaining]
    else:
        rJ = rank_order(-J[W - 1], remaining)
        rF = rank_order(-s_hist[W - 1], remaining)
        R = rJ + rF
        final_scores[remaining] = -R[remaining]

    order = np.lexsort((-final_scores, -np.array(elim_week)))
    placement = [0] * N
    for r, i in enumerate(order, start=1):
        placement[i] = r
    return elim_week, placement


def infer_elim_week_from_placement(
    placement: List[Optional[int]], schedule: List[int],
) -> List[int]:
    """Assign elimination weeks from placement order and weekly schedule."""
    W = len(schedule) - 1
    elim = [W + 1] * len(placement)
    ranked = [(p, i) for i, p in enumerate(placement) if p is not None and p > 0]
    ranked.sort(reverse=True)
    idx = 0
    for w in range(1, W + 1):
        for _ in range(schedule[w] if w < len(schedule) else 0):
            if idx >= len(ranked):
                break
            _, i = ranked[idx]
            elim[i] = w
            idx += 1
    return elim


def forward_simulate(
    season: int,
    names: List[str],
    J: np.ndarray,
    s_hist: np.ndarray,
    elim_schedule: List[int],
    regime_override: Optional[str] = None,
    judge_save: bool = True,
    force_bottom2: bool = False,
    force_no_bottom2: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Forward simulation. Phantom survivors use zeros (no data = 0 fan share).
    regime_override: "rank" | "percent" | None.
    Returns (elim_week_pred, placement_pred).
    """
    W, N = J.shape
    regime = regime_override if regime_override else season_regime(season)
    is_bottom2 = ((regime == "rank_bottom2") or force_bottom2) and not force_no_bottom2
    is_bottom2_season = ((season_regime(season) == "rank_bottom2") or force_bottom2) and not force_no_bottom2
    elim_week = [W + 1] * N
    for w in range(W):
        if w == W - 1:
            continue
        k = elim_schedule[w + 1] if w + 1 < len(elim_schedule) else 0
        if k <= 0:
            continue
        active = np.array([elim_week[i] > w + 1 for i in range(N)])
        active_idx = np.where(active)[0]
        if len(active_idx) <= k:
            for i in active_idx:
                elim_week[i] = w + 1
            continue
        J_w = J[w]
        s_w = s_hist[w]
        if regime == "percent":
            jp = judge_pct(J_w, active)
            combined = jp + s_w
            order = active_idx[np.argsort(combined[active_idx])]
            if is_bottom2_season and k == 1 and len(order) >= 2 and judge_save:
                bottom2 = order[:2]
                elim_idx = np.array([bottom2[np.argmin(J_w[bottom2])]])
            elif is_bottom2_season and k == 1 and len(order) >= 2 and not judge_save:
                bottom2 = order[:2]
                elim_idx = np.array([bottom2[np.argmin(s_w[bottom2])]])
            else:
                elim_idx = order[:k]
        else:
            rJ = rank_order(-J_w, active)
            rF = rank_order(-s_w, active)
            R = rJ + rF
            order = active_idx[np.argsort(R[active_idx])]
            if is_bottom2 and k == 1 and len(order) >= 2 and judge_save:
                bottom2 = order[-2:]
                elim_idx = np.array([bottom2[np.argmin(J_w[bottom2])]])
            elif is_bottom2 and k == 1 and len(order) >= 2 and not judge_save:
                bottom2 = order[-2:]
                elim_idx = np.array([bottom2[np.argmin(s_w[bottom2])]])
            else:
                elim_idx = order[-k:]
        for i in elim_idx:
            elim_week[i] = w + 1
    remaining = np.array([elim_week[i] == W + 1 for i in range(N)])
    final_scores = np.full(N, -np.inf)
    if regime == "percent":
        jp = judge_pct(J[W - 1], remaining)
        final_scores[remaining] = jp[remaining] + s_hist[W - 1][remaining]
    else:
        rJ = rank_order(-J[W - 1], remaining)
        rF = rank_order(-s_hist[W - 1], remaining)
        R = rJ + rF
        final_scores[remaining] = -R[remaining]
    order = np.lexsort((-final_scores, -np.array(elim_week)))
    placement = [0] * N
    for r, i in enumerate(order, start=1):
        placement[i] = r
    return elim_week, placement


def kendall_tau(order_a: List[str], order_b: List[str]) -> float:
    """Kendall tau distance (fraction of pairs that are discordant)."""
    if len(order_a) != len(order_b) or len(order_a) < 2:
        return 0.0
    rank_a = {n: i for i, n in enumerate(order_a)}
    rank_b = {n: i for i, n in enumerate(order_b)}
    n = len(order_a)
    pairs = 0
    discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_i, a_j = order_a[i], order_a[j]
            if a_i not in rank_b or a_j not in rank_b:
                continue
            pairs += 1
            if (rank_a[a_i] - rank_a[a_j]) * (rank_b[a_i] - rank_b[a_j]) < 0:
                discord += 1
    return discord / pairs if pairs else 0.0


def mean_displacement(placement_a: List[int], placement_b: List[int]) -> float:
    """Mean absolute difference in placement (1-indexed)."""
    n = min(len(placement_a), len(placement_b))
    if n == 0:
        return 0.0
    return float(np.mean([abs(placement_a[i] - placement_b[i]) for i in range(n)]))


def order_from_placement(names: List[str], placement: List[int]) -> List[str]:
    return [names[i] for i in np.argsort(placement)]
