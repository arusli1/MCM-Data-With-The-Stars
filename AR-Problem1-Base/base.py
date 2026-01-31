import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
RESULTS_DIR = "AR-Problem1-Base/base_results"

EPS = 1e-8
RANK_MSE_WEIGHT = 1.0
WEEKLY_ELIM_WEIGHT = 3.0
ENTROPY_WEIGHT = 0.02
MIN_SHARE = 0.01
MAX_SHARE = 0.8
ALPHA_PENALTY = 0.0
JUDGE_SCALE = 0.5
S0_PRIOR_SCALE = 6.0
S0_PRIOR_CONC = 5.0
N_S0_SAMPLES = 100
REFINE_STEPS = 20
REFINE_SCALE = 0.3
BOOTSTRAP_RUNS = 6
SEED = 42
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"
SENSITIVITY_MODE = os.getenv("SENSITIVITY_MODE", "0") == "1"
OAT_SENSITIVITY_MODE = os.getenv("OAT_SENSITIVITY_MODE", "0") == "1"


@dataclass
class SeasonResult:
    season: int
    names: List[str]
    elim_week_true: List[int]
    elim_week_pred: List[int]
    placement_true: List[Optional[int]]
    placement_pred: List[int]
    mean_abs_rank_diff: float
    mean_sq_rank_diff: float
    weekly_elim_match_rate: float
    finals_order_match: float
    placement_match_rate: float
    elim_week_mae: float
    objective: float
    alpha: float
    s0_entropy: float
    s0: np.ndarray
    s_hist: np.ndarray


def parse_week_cols(df: pd.DataFrame) -> List[int]:
    weeks = set()
    for c in df.columns:
        match = re.match(r"week(\d+)_judge\d+_score", c)
        if match:
            weeks.add(int(match.group(1)))
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
    W = max_week
    N = len(names)
    J = np.zeros((W, N), dtype=float)
    for w in range(1, W + 1):
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
        cols = [c for c in cols if c in df_season.columns]
        vals = df_season[cols].replace("N/A", pd.NA)
        numeric = vals.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        J[w - 1] = numeric.to_numpy().sum(axis=1)
    return J


def compute_elim_week(J: np.ndarray) -> List[int]:
    W, N = J.shape
    elim_week = [W + 1] * N
    for i in range(N):
        for w in range(W):
            if J[w, i] <= 0 and (w == 0 or J[w - 1, i] > 0):
                elim_week[i] = w + 1
                break
    return elim_week


def infer_elim_week_from_placement(
    placement_true: List[Optional[int]],
    elim_week_schedule: List[int],
) -> List[int]:
    # Use placement order to assign eliminations by week schedule.
    # Worst placements are eliminated earliest; finalists get W+1.
    W = len(elim_week_schedule) - 1
    elim_week = [W + 1] * len(placement_true)
    ranked = [
        (p, i)
        for i, p in enumerate(placement_true)
        if p is not None and p > 0
    ]
    ranked.sort(reverse=True)  # worst (largest placement) first
    idx = 0
    for w in range(1, W + 1):
        k = elim_week_schedule[w]
        for _ in range(k):
            if idx >= len(ranked):
                break
            _, i = ranked[idx]
            elim_week[i] = w
            idx += 1
    return elim_week


def compute_judge_pct(J_w: np.ndarray, active: np.ndarray) -> np.ndarray:
    scores = J_w.copy()
    scores[~active] = 0.0
    total = scores.sum()
    if total <= 0:
        return np.zeros_like(scores)
    return scores / total


def zscore_week(J_w: np.ndarray, active: np.ndarray) -> np.ndarray:
    vals = J_w[active]
    if vals.size == 0:
        return np.zeros_like(J_w)
    mean = vals.mean()
    std = vals.std() + EPS
    out = np.zeros_like(J_w)
    out[active] = (vals - mean) / std
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (ex.sum() + EPS)


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + EPS)))


def softmax_logits(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    ex = np.exp(z)
    return ex / (ex.sum() + EPS)


def season_regime(season: int) -> str:
    if season <= 2:
        return "rank"
    if 3 <= season <= 27:
        return "percent"
    return "rank_bottom2"


def rank_order(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    idx = np.where(active)[0]
    order = idx[np.argsort(values[idx])]
    ranks = np.zeros_like(values, dtype=float)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    return ranks


def normalize_with_bounds(values: np.ndarray, min_share: float, max_share: float) -> np.ndarray:
    n = values.size
    if n == 0:
        return values
    if min_share * n >= 1.0:
        min_share = 0.0
    max_share = min(max_share, 1.0)
    x = np.clip(values, 0.0, None)
    if x.sum() <= 0:
        x = np.full(n, 1.0 / n, dtype=float)
    else:
        x = x / x.sum()
    x = np.clip(x, min_share, max_share)
    for _ in range(100):
        total = x.sum()
        if abs(total - 1.0) <= 1e-6:
            break
        if total < 1.0:
            free = x < max_share - EPS
            if not free.any():
                break
            add = (1.0 - total) / free.sum()
            x[free] = np.minimum(x[free] + add, max_share)
        else:
            free = x > min_share + EPS
            if not free.any():
                break
            sub = (total - 1.0) / free.sum()
            x[free] = np.maximum(x[free] - sub, min_share)
    if x.sum() > 0:
        x = x / x.sum()
    return x


def simulate_season(
    season: int,
    names: List[str],
    J: np.ndarray,
    elim_week_true: List[int],
    s0: np.ndarray,
    alpha: float,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    W, N = J.shape
    s = s0.copy()
    elim_week_pred = [W + 1] * N
    s_hist = np.zeros((W, N), dtype=float)
    regime = season_regime(season)
    for w in range(W):
        active = np.array([elim_week_pred[i] > w + 1 for i in range(N)], dtype=bool)
        if active.sum() > 1:
            J_w = J[w]
            Jz = zscore_week(J_w, active)
            s = s * np.exp(alpha * Jz * JUDGE_SCALE)
            s[~active] = 0.0
            s_sum = s[active].sum()
            if s_sum <= 0:
                s[active] = 1.0 / active.sum()
            else:
                s[active] = s[active] / s_sum
            # Keep shares within realistic bounds.
            s[active] = normalize_with_bounds(s[active], MIN_SHARE, MAX_SHARE)
        s_hist[w] = s
        # Finals week: rank remaining using combined scores, no eliminations.
        if w == W - 1:
            continue
        k = sum(1 for e in elim_week_true if e == w + 1)
        if k <= 0:
            continue
        active_idx = np.where(active)[0]
        if active_idx.size == 0:
            continue
        if active_idx.size <= k:
            elim_idx = active_idx
        else:
            if regime == "percent":
                judge_pct = compute_judge_pct(J_w, active)
                combined = judge_pct + s
                order = active_idx[np.argsort(combined[active_idx])]
                elim_idx = order[:k]
            elif regime == "rank":
                rJ = rank_order(-J_w, active)
                rF = rank_order(-s, active)
                R = rJ + rF
                order = active_idx[np.argsort(R[active_idx])]
                elim_idx = order[-k:]
            else:
                rJ = rank_order(-J_w, active)
                rF = rank_order(-s, active)
                R = rJ + rF
                order = active_idx[np.argsort(R[active_idx])]
                if k == 1 and len(order) >= 2:
                    bottom2 = order[-2:]
                    j_scores = J_w[bottom2]
                    elim_idx = np.array([bottom2[np.argmin(j_scores)]])
                else:
                    elim_idx = order[-k:]
        for i in elim_idx:
            elim_week_pred[i] = w + 1
        if active.sum() - k > 0:
            elim_share = s[elim_idx].sum()
            s[elim_idx] = 0.0
            remaining = active_idx[~np.isin(active_idx, elim_idx)]
            s[remaining] += elim_share / len(remaining)
            s[remaining] = normalize_with_bounds(s[remaining], MIN_SHARE, MAX_SHARE)
    # Final-week tie-break scores for finalists (higher = better).
    final_scores = np.full(N, -np.inf, dtype=float)
    remaining = np.array([ew == W + 1 for ew in elim_week_pred], dtype=bool)
    if remaining.any():
        J_w = J[W - 1]
        if regime == "percent":
            judge_pct = compute_judge_pct(J_w, remaining)
            combined = judge_pct + s
            final_scores[remaining] = combined[remaining]
        else:
            rJ = rank_order(-J_w, remaining)
            rF = rank_order(-s, remaining)
            R = rJ + rF
            final_scores[remaining] = -R[remaining]
    # Enforce elimination counts per week to match the true schedule.
    schedule = [0] * (W + 1)
    for ew in elim_week_true:
        if ew <= W:
            schedule[ew] += 1
    pred_schedule = [0] * (W + 1)
    for ew in elim_week_pred:
        if ew <= W:
            pred_schedule[ew] += 1
    if pred_schedule != schedule:
        # Reassign elim weeks to satisfy the schedule using current ordering.
        placement_pred = placement_from_elim(elim_week_pred, final_scores)
        order = np.argsort(-np.array(placement_pred))  # worst to best
        elim_week_fixed = [W + 1] * N
        cursor = 0
        for w in range(1, W + 1):
            k = schedule[w]
            for _ in range(k):
                if cursor >= len(order):
                    break
                elim_week_fixed[order[cursor]] = w
                cursor += 1
        elim_week_pred = elim_week_fixed
    return elim_week_pred, s_hist, final_scores


def placement_from_elim(elim_week: List[int], tie_break: np.ndarray) -> List[int]:
    # Higher elimination week is better (winner has max week),
    # break ties among finalists using tie_break (higher is better).
    order = np.lexsort((-tie_break, -np.array(elim_week)))
    placement = [0] * len(elim_week)
    for rank, idx in enumerate(order, start=1):
        placement[idx] = rank
    return placement


def order_from_placement(names: List[str], placement_true: List[Optional[int]]) -> List[str]:
    vals = []
    for name, p in zip(names, placement_true):
        if p is None:
            continue
        vals.append((p, name))
    if not vals:
        return []
    return [name for _, name in sorted(vals, key=lambda x: x[0])]


def evaluate(
    names: List[str],
    elim_week_true: List[int],
    elim_week_pred: List[int],
    placement_true: List[Optional[int]],
    tie_break: np.ndarray,
    max_week: int,
) -> Tuple[float, float, float, float, float, float]:
    placement_pred = placement_from_elim(elim_week_pred, tie_break)
    diffs = []
    sq_diffs = []
    matches = []
    has_missing = any(p is None for p in placement_true)
    has_ties = len(
        {p for p in placement_true if p is not None}
    ) < len([p for p in placement_true if p is not None])
    order_true = order_from_placement(names, placement_true)
    order_pred = [name for _, name in sorted(zip(placement_pred, names), key=lambda x: x[0])]
    if not has_missing and has_ties:
        rank_pred = {name: r for r, name in enumerate(order_pred, start=1)}
        vals = sorted({p for p in placement_true if p is not None})
        start = 1
        placement_ranges = {}
        for p in vals:
            count = sum(1 for v in placement_true if v == p)
            placement_ranges[p] = (start, start + count - 1)
            start += count
        for name, p_true in zip(names, placement_true):
            r = rank_pred[name]
            lo, hi = placement_ranges[p_true]
            if lo <= r <= hi:
                diffs.append(0.0)
                sq_diffs.append(0.0)
                matches.append(1.0)
            else:
                d = min(abs(r - lo), abs(r - hi))
                diffs.append(d)
                sq_diffs.append(d * d)
                matches.append(0.0)
        mean_abs_rank_diff = float(np.mean(diffs)) if diffs else float("nan")
        mean_sq_rank_diff = float(np.mean(sq_diffs)) if sq_diffs else float("nan")
        placement_match_rate = float(np.mean(matches)) if matches else 0.0
    else:
        for i, p_true in enumerate(placement_true):
            if p_true is None:
                continue
            d = abs(p_true - placement_pred[i])
            diffs.append(d)
            sq_diffs.append(d * d)
            matches.append(1.0 if p_true == placement_pred[i] else 0.0)
        mean_abs_rank_diff = float(np.mean(diffs)) if diffs else float("nan")
        mean_sq_rank_diff = float(np.mean(sq_diffs)) if sq_diffs else float("nan")
        placement_match_rate = float(np.mean(matches)) if matches else 0.0

    weeks = sorted(set(elim_week_true))
    hits = 0
    total = 0
    for w in weeks:
        if w > max_week:
            continue
        true_set = {names[i] for i, ew in enumerate(elim_week_true) if ew == w}
        if len(true_set) == 0:
            continue
        # Treat tied placements as interchangeable for weekly match.
        group_ids = []
        for i, p in enumerate(placement_true):
            if p is None:
                group_ids.append(f"__name__{names[i]}")
            else:
                group_ids.append(f"placement_{p}")
        true_counts = {}
        pred_counts = {}
        for i, ew in enumerate(elim_week_true):
            if ew == w:
                gid = group_ids[i]
                true_counts[gid] = true_counts.get(gid, 0) + 1
        for i, ew in enumerate(elim_week_pred):
            if ew == w:
                gid = group_ids[i]
                pred_counts[gid] = pred_counts.get(gid, 0) + 1
        total += 1
        if true_counts == pred_counts:
            hits += 1
    weekly_elim_match_rate = hits / total if total > 0 else 0.0

    finals_order_match = 0.0
    if order_true:
        if not has_missing and has_ties:
            placements_by_name = dict(zip(names, placement_true))
            finals_order_match = 1.0
            for i in range(len(order_pred) - 1):
                a = placements_by_name[order_pred[i]]
                b = placements_by_name[order_pred[i + 1]]
                if a is None or b is None:
                    continue
                if a > b:
                    finals_order_match = 0.0
                    break
        else:
            finals_order_match = 1.0 if order_true == order_pred else 0.0
    # If placement order is fully correct, weekly eliminations must be correct.
    if order_true and order_pred == order_true:
        placement_match_rate = 1.0
        finals_order_match = 1.0
        weekly_elim_match_rate = 1.0
    elim_week_mae = float(
        np.mean([abs(a - b) for a, b in zip(elim_week_true, elim_week_pred)])
    )
    return (
        mean_abs_rank_diff,
        mean_sq_rank_diff,
        weekly_elim_match_rate,
        elim_week_mae,
        finals_order_match,
        placement_match_rate,
    )
def fit_season(
    season: int,
    df_season: pd.DataFrame,
    weeks: List[int],
    alpha_grid: List[float],
    n_s0_samples: int,
    refine_steps: int,
    refine_scale: float,
    obj_weight: float,
    seed: int,
) -> SeasonResult:
    names = df_season["celebrity_name"].tolist()
    J = build_judge_matrix(df_season, weeks)
    elim_week_judge = compute_elim_week(J)
    placement_true = []
    for v in df_season["placement"].tolist():
        try:
            placement_true.append(int(v))
        except Exception:
            placement_true.append(None)
    elim_week_schedule = [0] * (J.shape[0] + 1)
    for ew in elim_week_judge:
        if ew <= J.shape[0]:
            elim_week_schedule[ew] += 1
    elim_week_true = infer_elim_week_from_placement(placement_true, elim_week_schedule)
    for i, p in enumerate(placement_true):
        if p is None:
            elim_week_true[i] = elim_week_judge[i]

    best: Optional[SeasonResult] = None
    rng = np.random.default_rng(seed + season)
    for alpha in alpha_grid:
        for _ in range(n_s0_samples):
            prior = np.ones(len(names))
            placement_vals = np.array(
                [p if p is not None else len(names) for p in placement_true],
                dtype=float,
            )
            inv_rank = 1.0 / (placement_vals + EPS)
            if inv_rank.sum() > 0:
                prior = prior + S0_PRIOR_SCALE * (inv_rank / inv_rank.sum())
            s0 = rng.dirichlet(prior * S0_PRIOR_CONC)
            elim_week_pred, s_hist, final_scores = simulate_season(
                season, names, J, elim_week_true, s0, alpha
            )
            (
                mean_abs_rank_diff,
                mean_sq_rank_diff,
                weekly_elim_match_rate,
                elim_week_mae,
                finals_order_match,
                placement_match_rate,
            ) = evaluate(
                names, elim_week_true, elim_week_pred, placement_true, final_scores, J.shape[0]
            )
            s0_entropy = entropy(s0)
            objective = (
                RANK_MSE_WEIGHT * mean_sq_rank_diff
                + obj_weight * (1.0 - weekly_elim_match_rate)
                - ENTROPY_WEIGHT * s0_entropy
                + ALPHA_PENALTY * (alpha**2)
            )
            placement_pred = placement_from_elim(elim_week_pred, final_scores)
            res = SeasonResult(
                season=season,
                names=names,
                elim_week_true=elim_week_true,
                elim_week_pred=elim_week_pred,
                placement_true=placement_true,
                placement_pred=placement_pred,
                mean_abs_rank_diff=mean_abs_rank_diff,
                mean_sq_rank_diff=mean_sq_rank_diff,
                weekly_elim_match_rate=weekly_elim_match_rate,
                finals_order_match=finals_order_match,
                placement_match_rate=placement_match_rate,
                elim_week_mae=elim_week_mae,
                objective=objective,
                alpha=alpha,
                s0_entropy=s0_entropy,
                s0=s0,
                s_hist=s_hist,
            )
            if best is None or res.objective < best.objective:
                best = res
        if best is None:
            continue
        # Local refinement around current best s0 (simple coordinate-free search).
        logits = np.log(best.s0 + EPS)
        for _ in range(refine_steps):
            cand = softmax_logits(logits + rng.normal(0.0, refine_scale, size=logits.shape))
            elim_week_pred, s_hist, final_scores = simulate_season(
                season, names, J, elim_week_true, cand, alpha
            )
            (
                mean_abs_rank_diff,
                mean_sq_rank_diff,
                weekly_elim_match_rate,
                elim_week_mae,
                finals_order_match,
                placement_match_rate,
            ) = evaluate(
                names, elim_week_true, elim_week_pred, placement_true, final_scores, J.shape[0]
            )
            cand_entropy = entropy(cand)
            objective = (
                RANK_MSE_WEIGHT * mean_sq_rank_diff
                + obj_weight * (1.0 - weekly_elim_match_rate)
                - ENTROPY_WEIGHT * cand_entropy
                + ALPHA_PENALTY * (alpha**2)
            )
            if objective < best.objective:
                placement_pred = placement_from_elim(elim_week_pred, final_scores)
                best = SeasonResult(
                    season=season,
                    names=names,
                    elim_week_true=elim_week_true,
                    elim_week_pred=elim_week_pred,
                    placement_true=placement_true,
                    placement_pred=placement_pred,
                    mean_abs_rank_diff=mean_abs_rank_diff,
                    mean_sq_rank_diff=mean_sq_rank_diff,
                    weekly_elim_match_rate=weekly_elim_match_rate,
                    finals_order_match=finals_order_match,
                    placement_match_rate=placement_match_rate,
                    elim_week_mae=elim_week_mae,
                    objective=objective,
                    alpha=alpha,
                    s0_entropy=cand_entropy,
                    s0=cand,
                    s_hist=s_hist,
                )
                logits = np.log(best.s0 + EPS)
    if best is None:
        raise RuntimeError("No results produced")
    return best


def run_sensitivity(
    df: pd.DataFrame,
    weeks: List[int],
    seasons: List[int],
    alpha_grid: List[float],
    seed: int,
) -> pd.DataFrame:
    sens_rows = []
    judge_scales = [0.3, 0.5, 0.7]
    alpha_penalties = [0.0, 0.2, 0.5]
    n_s0_samples = 30
    refine_steps = 5
    refine_scale = REFINE_SCALE
    obj_weight = WEEKLY_ELIM_WEIGHT
    for season in seasons:
        df_season = df[df["season"] == season].reset_index(drop=True)
        for js in judge_scales:
            for ap in alpha_penalties:
                global JUDGE_SCALE, ALPHA_PENALTY
                prev_js, prev_ap = JUDGE_SCALE, ALPHA_PENALTY
                JUDGE_SCALE, ALPHA_PENALTY = js, ap
                res = fit_season(
                    season,
                    df_season,
                    weeks,
                    alpha_grid,
                    n_s0_samples,
                    refine_steps,
                    refine_scale,
                    obj_weight,
                    seed,
                )
                sens_rows.append(
                    {
                        "season": season,
                        "judge_scale": js,
                        "alpha_penalty": ap,
                        "alpha": res.alpha,
                        "mean_sq_rank_diff": res.mean_sq_rank_diff,
                        "weekly_elim_match_rate": res.weekly_elim_match_rate,
                        "elim_week_mae": res.elim_week_mae,
                        "objective": res.objective,
                    }
                )
                JUDGE_SCALE, ALPHA_PENALTY = prev_js, prev_ap
    return pd.DataFrame(sens_rows)


def run_oat_sensitivity(
    df: pd.DataFrame,
    weeks: List[int],
    seasons: List[int],
    alpha_grid: List[float],
    seed: int,
    n_s0_samples: int,
    refine_steps: int,
    refine_scale: float,
    obj_weight: float,
) -> pd.DataFrame:
    param_grid = {
        "judge_scale": [0.3, 0.5, 0.7],
        "alpha_penalty": [0.0, 0.2, 0.5],
        "entropy_weight": [0.0, 0.02, 0.05],
        "s0_prior_scale": [0.0, 3.0, 6.0],
        "s0_prior_conc": [1.0, 5.0, 10.0],
    }
    sens_rows = []
    for param, values in param_grid.items():
        for val in values:
            global JUDGE_SCALE, ALPHA_PENALTY, ENTROPY_WEIGHT
            global S0_PRIOR_SCALE, S0_PRIOR_CONC

            prev = {
                "judge_scale": JUDGE_SCALE,
                "alpha_penalty": ALPHA_PENALTY,
                "entropy_weight": ENTROPY_WEIGHT,
                "s0_prior_scale": S0_PRIOR_SCALE,
                "s0_prior_conc": S0_PRIOR_CONC,
            }

            if param == "judge_scale":
                JUDGE_SCALE = val
            elif param == "alpha_penalty":
                ALPHA_PENALTY = val
            elif param == "entropy_weight":
                ENTROPY_WEIGHT = val
            elif param == "s0_prior_scale":
                S0_PRIOR_SCALE = val
            elif param == "s0_prior_conc":
                S0_PRIOR_CONC = val

            for season in seasons:
                df_season = df[df["season"] == season].reset_index(drop=True)
                res = fit_season(
                    int(season),
                    df_season,
                    weeks,
                    alpha_grid,
                    n_s0_samples,
                    refine_steps,
                    refine_scale,
                    obj_weight,
                    seed,
                )
                sens_rows.append(
                    {
                        "param": param,
                        "value": val,
                        "season": int(season),
                        "alpha": res.alpha,
                        "mean_sq_rank_diff": res.mean_sq_rank_diff,
                        "weekly_elim_match_rate": res.weekly_elim_match_rate,
                        "elim_week_mae": res.elim_week_mae,
                        "objective": res.objective,
                    }
                )

            JUDGE_SCALE = prev["judge_scale"]
            ALPHA_PENALTY = prev["alpha_penalty"]
            ENTROPY_WEIGHT = prev["entropy_weight"]
            S0_PRIOR_SCALE = prev["s0_prior_scale"]
            S0_PRIOR_CONC = prev["s0_prior_conc"]
            obj_weight = WEEKLY_ELIM_WEIGHT

    return pd.DataFrame(sens_rows)


def main():
    df = pd.read_csv(DATA_PATH)
    weeks = parse_week_cols(df)
    alpha_grid = list(np.linspace(-0.2, 1.5, 25))
    n_s0_samples = N_S0_SAMPLES
    refine_steps = REFINE_STEPS
    refine_scale = REFINE_SCALE
    obj_weight = WEEKLY_ELIM_WEIGHT
    seed = SEED
    bootstrap_runs = BOOTSTRAP_RUNS
    if FAST_MODE:
        alpha_grid = list(np.linspace(-0.2, 1.5, 13))
        n_s0_samples = 60
        refine_steps = 15
        bootstrap_runs = 4
    if SENSITIVITY_MODE or OAT_SENSITIVITY_MODE:
        n_s0_samples = min(n_s0_samples, 30)
        refine_steps = min(refine_steps, 5)
        bootstrap_runs = min(bootstrap_runs, 2)

    if OAT_SENSITIVITY_MODE:
        seasons = sorted(df["season"].unique(), reverse=True)
        oat = run_oat_sensitivity(
            df,
            weeks,
            seasons,
            alpha_grid,
            seed,
            n_s0_samples,
            refine_steps,
            refine_scale,
            obj_weight,
        )
        oat_path = os.path.join(RESULTS_DIR, "base_sensitivity_oat.csv")
        oat.to_csv(oat_path, index=False)
        print(f"Wrote {oat_path}")
        return

    results = []
    boot_rows = []
    placement_rows = []
    share_rows = []
    share_uncertainty_rows = []
    s0_rows = []
    truth_rows = []
    season_weights = []
    for season in sorted(df["season"].unique(), reverse=True):
        df_season = df[df["season"] == season]
        season = int(season)
        boot = []
        for b in range(bootstrap_runs):
            res = fit_season(
                season,
                df_season.reset_index(drop=True),
                weeks,
                alpha_grid,
                n_s0_samples,
                refine_steps,
                refine_scale,
                obj_weight,
                seed + 1000 * b,
            )
            boot.append(res)
            boot_rows.append(
                {
                    "season": season,
                    "bootstrap": b,
                    "alpha": res.alpha,
                    "s0_entropy": res.s0_entropy,
                    "mean_abs_rank_diff": res.mean_abs_rank_diff,
                    "mean_sq_rank_diff": res.mean_sq_rank_diff,
                    "weekly_elim_match_rate": res.weekly_elim_match_rate,
                    "finals_order_match": res.finals_order_match,
                    "placement_match_rate": res.placement_match_rate,
                    "elim_week_mae": res.elim_week_mae,
                    "objective": res.objective,
                }
            )
        best = min(boot, key=lambda r: r.objective)
        boot_hist = np.stack([r.s_hist for r in boot], axis=0)
        rank_diffs = np.array([r.mean_abs_rank_diff for r in boot], dtype=float)
        rank_sq_diffs = np.array([r.mean_sq_rank_diff for r in boot], dtype=float)
        elim_rates = np.array([r.weekly_elim_match_rate for r in boot], dtype=float)
        finals_match = np.array([r.finals_order_match for r in boot], dtype=float)
        placement_match = np.array([r.placement_match_rate for r in boot], dtype=float)
        elim_week_mae = np.array([r.elim_week_mae for r in boot], dtype=float)
        alpha_vals = np.array([r.alpha for r in boot], dtype=float)
        s0_entropy = np.array([r.s0_entropy for r in boot], dtype=float)
        results.append(
            {
                "season": season,
                "alpha": best.alpha,
                "s0_entropy": best.s0_entropy,
                "mean_abs_rank_diff": best.mean_abs_rank_diff,
                "mean_sq_rank_diff": best.mean_sq_rank_diff,
                "weekly_elim_match_rate": best.weekly_elim_match_rate,
                "finals_order_match": best.finals_order_match,
                "placement_match_rate": best.placement_match_rate,
                "elim_week_mae": best.elim_week_mae,
                "objective": best.objective,
                "mean_abs_rank_diff_mean": float(np.mean(rank_diffs)),
                "mean_abs_rank_diff_std": float(np.std(rank_diffs)),
                "mean_sq_rank_diff_mean": float(np.mean(rank_sq_diffs)),
                "mean_sq_rank_diff_std": float(np.std(rank_sq_diffs)),
                "weekly_elim_match_rate_mean": float(np.mean(elim_rates)),
                "weekly_elim_match_rate_std": float(np.std(elim_rates)),
                "finals_order_match_mean": float(np.mean(finals_match)),
                "finals_order_match_std": float(np.std(finals_match)),
                "placement_match_rate_mean": float(np.mean(placement_match)),
                "placement_match_rate_std": float(np.std(placement_match)),
                "elim_week_mae_mean": float(np.mean(elim_week_mae)),
                "elim_week_mae_std": float(np.std(elim_week_mae)),
                "alpha_p10": float(np.percentile(alpha_vals, 10)),
                "alpha_p50": float(np.percentile(alpha_vals, 50)),
                "alpha_p90": float(np.percentile(alpha_vals, 90)),
                "s0_entropy_p10": float(np.percentile(s0_entropy, 10)),
                "s0_entropy_p50": float(np.percentile(s0_entropy, 50)),
                "s0_entropy_p90": float(np.percentile(s0_entropy, 90)),
                "mean_sq_rank_diff_p10": float(np.percentile(rank_sq_diffs, 10)),
                "mean_sq_rank_diff_p50": float(np.percentile(rank_sq_diffs, 50)),
                "mean_sq_rank_diff_p90": float(np.percentile(rank_sq_diffs, 90)),
                "weekly_elim_match_rate_p10": float(np.percentile(elim_rates, 10)),
                "weekly_elim_match_rate_p50": float(np.percentile(elim_rates, 50)),
                "weekly_elim_match_rate_p90": float(np.percentile(elim_rates, 90)),
            }
        )
        season_weights.append(len(best.names))
        print(
            f"Season {season}: alpha={best.alpha:.2f} s0_entropy={best.s0_entropy:.3f} "
            f"rank_diff={best.mean_abs_rank_diff:.3f} "
            f"weekly_elim_match={best.weekly_elim_match_rate:.3f} "
            f"finals_match={best.finals_order_match:.3f} "
            f"placement_match={best.placement_match_rate:.3f} "
            f"elim_week_mae={best.elim_week_mae:.3f}"
        )
        # Terminal details: placement order + top s0
        order_pred = [
            name
            for _, name in sorted(
                zip(best.placement_pred, best.names), key=lambda x: x[0]
            )
        ]
        order_true = order_from_placement(best.names, best.placement_true)
        if not order_true:
            order_true = [best.names[i] for i in np.argsort(best.elim_week_true)]
        print(f"  Pred placement order: {', '.join(order_pred)}")
        print(f"  True placement order: {', '.join(order_true)}")
        top_idx = np.argsort(best.s0)[::-1][:5]
        top_s0 = ", ".join([f"{best.names[i]}={best.s0[i]:.3f}" for i in top_idx])
        print(f"  Top s0: {top_s0}")
        # Finalist shares after all judge effects (last-week s).
        final_week = best.s_hist.shape[0] - 1
        finalist_idx = [
            i for i, ew in enumerate(best.elim_week_pred) if ew == best.s_hist.shape[0] + 1
        ]
        if finalist_idx:
            finalist_shares = sorted(
                [(best.s_hist[final_week, i], best.names[i]) for i in finalist_idx],
                reverse=True,
            )
            top_finalists = ", ".join(
                [f"{name}={share:.3f}" for share, name in finalist_shares]
            )
            print(f"  Finalist s*: {top_finalists}")
        for i, name in enumerate(best.names):
            truth_rows.append(
                {
                    "season": season,
                    "celebrity_name": name,
                    "placement_true": best.placement_true[i],
                    "elim_week_true": best.elim_week_true[i],
                }
            )
            placement_rows.append(
                {
                    "season": season,
                    "celebrity_name": name,
                    "elim_week_true": best.elim_week_true[i],
                    "elim_week_pred": best.elim_week_pred[i],
                    "placement_true": best.placement_true[i],
                    "placement_pred": best.placement_pred[i],
                }
            )
            s0_rows.append(
                {
                    "season": season,
                    "celebrity_name": name,
                    "s0": best.s0[i],
                }
            )
        for w in range(best.s_hist.shape[0]):
            for i, name in enumerate(best.names):
                share_rows.append(
                    {
                        "season": season,
                        "week": w + 1,
                        "celebrity_name": name,
                        "s_share": best.s_hist[w, i],
                    }
                )
                boot_vals = boot_hist[:, w, i]
                share_uncertainty_rows.append(
                    {
                        "season": season,
                        "week": w + 1,
                        "celebrity_name": name,
                        "s_share_mean": float(np.mean(boot_vals)),
                        "s_share_std": float(np.std(boot_vals)),
                        "s_share_p10": float(np.percentile(boot_vals, 10)),
                        "s_share_p50": float(np.percentile(boot_vals, 50)),
                        "s_share_p90": float(np.percentile(boot_vals, 90)),
                    }
                )
    out = pd.DataFrame(results).sort_values("season")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "base_metrics.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    # Overall (single-number) evaluation summary across seasons.
    weights = np.array(season_weights, dtype=float)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(results)) / max(1, len(results))
    overall = {
        "overall_objective_mean": float(np.average([r["objective"] for r in results], weights=weights)),
        "overall_mean_sq_rank_diff": float(np.average([r["mean_sq_rank_diff"] for r in results], weights=weights)),
        "overall_weekly_elim_match_rate": float(
            np.average([r["weekly_elim_match_rate"] for r in results], weights=weights)
        ),
        "overall_elim_week_mae": float(np.average([r["elim_week_mae"] for r in results], weights=weights)),
        "overall_alpha_mean": float(np.average([r["alpha"] for r in results], weights=weights)),
    }
    overall_path = os.path.join(RESULTS_DIR, "base_overall_metrics.csv")
    pd.DataFrame([overall]).to_csv(overall_path, index=False)
    print(f"Wrote {overall_path}")
    boot_path = os.path.join(RESULTS_DIR, "base_bootstrap_results.csv")
    pd.DataFrame(boot_rows).to_csv(boot_path, index=False)
    print(f"Wrote {boot_path}")
    placement_path = os.path.join(RESULTS_DIR, "base_placement_orders.csv")
    pd.DataFrame(placement_rows).to_csv(placement_path, index=False)
    print(f"Wrote {placement_path}")
    share_path = os.path.join(RESULTS_DIR, "base_inferred_shares.csv")
    pd.DataFrame(share_rows).to_csv(share_path, index=False)
    print(f"Wrote {share_path}")
    share_uncertainty_path = os.path.join(
        RESULTS_DIR, "base_inferred_shares_uncertainty.csv"
    )
    pd.DataFrame(share_uncertainty_rows).to_csv(share_uncertainty_path, index=False)
    print(f"Wrote {share_uncertainty_path}")
    # Overall summary of inferred shares (all seasons/weeks).
    share_vals = np.array([r["s_share"] for r in share_rows], dtype=float)
    share_summary = {
        "s_share_mean": float(np.mean(share_vals)),
        "s_share_std": float(np.std(share_vals)),
        "s_share_p10": float(np.percentile(share_vals, 10)),
        "s_share_p50": float(np.percentile(share_vals, 50)),
        "s_share_p90": float(np.percentile(share_vals, 90)),
    }
    share_summary_path = os.path.join(RESULTS_DIR, "base_inferred_shares_summary.csv")
    pd.DataFrame([share_summary]).to_csv(share_summary_path, index=False)
    print(f"Wrote {share_summary_path}")
    s0_path = os.path.join(RESULTS_DIR, "base_s0.csv")
    pd.DataFrame(s0_rows).to_csv(s0_path, index=False)
    print(f"Wrote {s0_path}")
    truth_path = os.path.join(RESULTS_DIR, "base_truth_elim_weeks.csv")
    pd.DataFrame(truth_rows).to_csv(truth_path, index=False)
    print(f"Wrote {truth_path}")
    hyper_path = os.path.join(RESULTS_DIR, "base_hyperparams.json")
    pd.Series(
        {
            "alpha_grid": alpha_grid,
            "n_s0_samples": n_s0_samples,
            "refine_steps": refine_steps,
            "refine_scale": refine_scale,
            "weekly_elim_weight": obj_weight,
            "rank_mse_weight": RANK_MSE_WEIGHT,
            "entropy_weight": ENTROPY_WEIGHT,
            "min_share": MIN_SHARE,
            "max_share": MAX_SHARE,
            "alpha_penalty": ALPHA_PENALTY,
            "judge_scale": JUDGE_SCALE,
            "s0_prior_scale": S0_PRIOR_SCALE,
            "s0_prior_conc": S0_PRIOR_CONC,
            "bootstrap_runs": bootstrap_runs,
            "fast_mode": FAST_MODE,
            "seed": seed,
            "regimes": {"1-2": "rank", "3-27": "percent", "28+": "rank_bottom2"},
        }
    ).to_json(hyper_path)
    print(f"Wrote {hyper_path}")
    if SENSITIVITY_MODE:
        seasons = sorted(df["season"].unique(), reverse=True)
        sens = run_sensitivity(
            df,
            weeks,
            seasons,
            alpha_grid,
            seed,
        )
        sens_path = os.path.join(RESULTS_DIR, "base_sensitivity.csv")
        sens.to_csv(sens_path, index=False)
        print(f"Wrote {sens_path}")


if __name__ == "__main__":
    main()
