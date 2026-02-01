#!/usr/bin/env python3
"""
MCM 2026 Problem C Problem 1: Fan Vote Share Estimation (DWTS)
Placement-Inverse Prior with Constrained Optimization

Self-contained script: loads data, preprocesses, computes blended prior q_iw,
optimizes vote shares per week, handles all voting regimes (rank/percent/bottom-two),
runs bootstrap for uncertainty, evaluates consistency, generates graphs.

PAPER NOTES:
- Model: minimize sum((s - q_iw)^2) subject to s>=0, sum(s)=1, and elimination
  constraints ensuring eliminated contestants have worse combined (judge + fan)
  scores than survivors.
- Regimes: rank (seasons 1-2, 28-34), percent (3-27), bottom-two (28-34).
- Blended prior q_iw combines placement-based prior (1/placement) with
  weekly survival/deficit proxy for robustness.
- Bootstrap (JUDGE_PERTURB_SD) quantifies uncertainty in fan-share estimates.
"""

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import minimize, Bounds

# =============================================================================
# Tunable parameters (paper sensitivity)
# =============================================================================
BETA = 0.7           # Blended prior: beta * placement_boost + (1-beta) * weekly_proxy
K = 0.5              # Weekly proxy deficit coefficient
BOOT_N = 200         # Bootstrap iterations per week
JUDGE_PERTURB_SD = 0.5  # Bootstrap: N(0, sd) perturbation on judge_total
SEED = 42
EPS = 1e-10
MARGIN = 1e-6  # Strict separation: min_surv - max_elim >= MARGIN (avoids tie-breaking)
# Optional soft cap: penalize shares > MAX_SHARE_CAP to avoid unrealistic dominance
MAX_SHARE_CAP = 0.55  # Soft penalty when s_i > 0.55; set None to disable
MAX_SHARE_PENALTY = 2.0  # Weight for (s - cap)^2 when s > cap
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_SCRIPT_DIR, "..", "Data", "2026_MCM_Problem_C_Data.csv")
OUT_DIR = os.path.join(_SCRIPT_DIR, "outputs")
FIG_DIR = os.path.join(_SCRIPT_DIR, "figures")


# =============================================================================
# Preprocessing
# =============================================================================

def parse_week_cols(df: pd.DataFrame) -> List[int]:
    """Extract week numbers from judge score columns."""
    weeks = set()
    for c in df.columns:
        m = re.match(r"week(\d+)_judge\d+_score", c)
        if m:
            weeks.add(int(m.group(1)))
    return sorted(weeks)


def load_and_preprocess(path: str) -> Tuple[pd.DataFrame, List[int]]:
    """
    Load CSV, clean names, compute judge_total and judge_rank per week,
    infer survived, num_eliminated, active contestants, regime.
    Returns wide-format df with added columns and list of weeks.
    """
    if not os.path.isfile(path):
        # Sample data fallback
        print(f"CSV not found at {path}; using sample fallback.")
        return _sample_fallback(), list(range(1, 6))

    df = pd.read_csv(path)
    df["celebrity_name"] = df["celebrity_name"].astype(str).str.strip()

    weeks = parse_week_cols(df)
    if not weeks:
        raise ValueError("No week columns found")

    # Judge total per week
    for w in weeks:
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
        cols = [c for c in cols if c in df.columns]
        vals = df[cols].replace("N/A", pd.NA)
        numeric = vals.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        df[f"judge_total_w{w}"] = numeric.sum(axis=1).clip(lower=0)

    # Judge rank per season/week (ascending = better; lower rank = better)
    for w in weeks:
        col = f"judge_total_w{w}"
        df[f"judge_rank_w{w}"] = df.groupby("season")[col].rank(
            ascending=False, method="min"
        ).astype(int)

    # Infer elim_week from results
    def parse_elim_week(res: str, max_w: int) -> int:
        if pd.isna(res):
            return max_w + 1
        s = str(res).strip()
        if "Eliminated Week" in s:
            m = re.search(r"Eliminated Week\s*(\d+)", s, re.I)
            if m:
                return int(m.group(1))
        if "Withdrew" in s or "1st" in s or "2nd" in s or "3rd" in s or "4th" in s or "5th" in s or "Place" in s:
            return max_w + 1
        return max_w + 1

    max_week = max(weeks)
    df["elim_week"] = df["results"].apply(
        lambda r: parse_elim_week(r, max_week)
    )
    # Override for Withdrew: first week judge_total=0
    df["placement"] = pd.to_numeric(df["placement"], errors="coerce")
    withdrew = df["results"].astype(str).str.contains("Withdrew", na=False)
    for idx in df.index[withdrew]:
        for w in weeks:
            col = f"judge_total_w{w}"
            if col in df.columns and df.at[idx, col] <= 0:
                prev_ok = True
                if w > 1:
                    prev_col = f"judge_total_w{w-1}"
                    if prev_col in df.columns:
                        prev_ok = df.at[idx, prev_col] > 0
                if prev_ok:
                    df.at[idx, "elim_week"] = w
                    break

    # Survived at week w: 1 if elim_week > w
    for w in weeks:
        df[f"survived_w{w}"] = (df["elim_week"] > w).astype(int)

    # Regime
    def regime(season: int) -> str:
        if season <= 2:
            return "rank"
        if 3 <= season <= 27:
            return "percent"
        return "rank_bottom2"

    df["regime"] = df["season"].apply(regime)
    return df, weeks


def _sample_fallback() -> pd.DataFrame:
    """Generate minimal sample data when CSV missing."""
    rows = [
        {"celebrity_name": "A", "season": 1, "results": "1st Place", "placement": 1},
        {"celebrity_name": "B", "season": 1, "results": "Eliminated Week 2", "placement": 2},
        {"celebrity_name": "C", "season": 1, "results": "Eliminated Week 1", "placement": 3},
    ]
    df = pd.DataFrame(rows)
    for w in range(1, 6):
        for j in range(1, 4):
            df[f"week{w}_judge{j}_score"] = 7.0 if w <= 2 else 0.0
    df.loc[df["celebrity_name"] == "C", "week2_judge1_score"] = 0
    df.loc[df["celebrity_name"] == "C", "week2_judge2_score"] = 0
    df.loc[df["celebrity_name"] == "B", "week3_judge1_score"] = 0
    df.loc[df["celebrity_name"] == "B", "week3_judge2_score"] = 0
    return df


# =============================================================================
# Long-format per-week data
# =============================================================================

def build_long_format(df: pd.DataFrame, weeks: List[int]) -> pd.DataFrame:
    """
    Build long-format DataFrame: one row per (season, week, contestant).
    Columns: season, week, celebrity_name, judge_total, judge_rank, survived,
    elim_this_week, regime, placement, elim_week.
    """
    rows = []
    for _, r in df.iterrows():
        season = r["season"]
        name = r["celebrity_name"]
        elim_week = int(r["elim_week"])
        placement = r.get("placement")
        regime_val = r["regime"]
        max_w = max(weeks)
        for w in weeks:
            jtot = r.get(f"judge_total_w{w}", 0)
            jrank = r.get(f"judge_rank_w{w}", np.nan)
            surv = 1 if elim_week > w else 0
            elim_this = 1 if elim_week == w else 0
            rows.append({
                "season": season,
                "week": w,
                "celebrity_name": name,
                "judge_total": float(jtot),
                "judge_rank": jrank if pd.notna(jrank) else np.nan,
                "survived": surv,
                "elim_this_week": elim_this,
                "regime": regime_val,
                "placement": placement,
                "elim_week": elim_week,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Blended Prior q_iw
# =============================================================================

def compute_placement_boost(df: pd.DataFrame, long_df: pd.DataFrame, weeks: List[int]) -> pd.Series:
    """
    placement_boost = (1 / placement.clip(1)) normalized per season.
    """
    placement_inv = 1.0 / df["placement"].fillna(len(df)).clip(lower=1)
    # Normalize per season
    boost = pd.Series(index=long_df.index, dtype=float)
    for (season, week), grp in long_df.groupby(["season", "week"]):
        idxs = grp.index
        names = grp["celebrity_name"]
        vals = []
        for n in names:
            row = df[(df["season"] == season) & (df["celebrity_name"] == n)]
            if len(row) > 0:
                p = row["placement"].iloc[0]
                v = 1.0 / max(1, float(p)) if pd.notna(p) else 1.0
            else:
                v = 1.0
            vals.append(v)
        arr = np.array(vals, dtype=float)
        s = arr.sum()
        if s > 0:
            arr = arr / s
        boost.loc[idxs] = arr
    return boost


def compute_weekly_proxy(
    df: pd.DataFrame, long_df: pd.DataFrame, weeks: List[int], k: float = K
) -> pd.Series:
    """
    weekly_proxy:
    - survival_lag = survived.shift(1) per contestant (fillna 1 for week 1)
    - judge_lag = judge_total.shift(1)
    - median_judge_lag = groupby(season, week) median of judge_lag
    - deficit_lag = max(0, median_judge_lag - judge_lag)
    - weekly_proxy = survival_lag * (1 + k * deficit_lag), normalized per season/week
    - Week 1: weekly_proxy = 1 / num_active
    """
    ld = long_df.copy().sort_values(["season", "celebrity_name", "week"])
    ld["judge_lag"] = ld.groupby(["season", "celebrity_name"])["judge_total"].shift(1).fillna(0)
    ld["survival_lag"] = ld.groupby(["season", "celebrity_name"])["survived"].shift(1).fillna(1)
    ld.loc[ld["week"] == 1, "survival_lag"] = 1

    proxy = pd.Series(index=long_df.index, dtype=float)
    for (season, week), grp in ld.groupby(["season", "week"]):
        idxs = grp.index
        n_active = int(grp["survived"].sum())
        if week == 1 or n_active <= 0:
            proxy.loc[idxs] = 1.0 / max(1, len(idxs))
            continue
        median_jl = grp["judge_lag"].median()
        surv_lag = grp["survival_lag"].values
        jl = grp["judge_lag"].values
        deficit = np.maximum(0, median_jl - jl)
        raw = surv_lag * (1 + k * deficit)
        s = raw.sum()
        proxy.loc[idxs] = (raw / s) if s > 0 else (1.0 / len(idxs))
    return proxy


def compute_q_iw(
    df: pd.DataFrame, long_df: pd.DataFrame, weeks: List[int],
    beta: float = BETA, k: float = K
) -> pd.Series:
    """q_iw = beta * placement_boost + (1-beta) * weekly_proxy, normalized."""
    pb = compute_placement_boost(df, long_df, weeks)
    wp = compute_weekly_proxy(df, long_df, weeks, k=k)
    q = beta * pb + (1 - beta) * wp
    for (season, week), grp in long_df.groupby(["season", "week"]):
        idxs = grp.index
        vals = q.loc[idxs].values
        s = vals.sum()
        if s > 0:
            q.loc[idxs] = vals / s
    return q


# =============================================================================
# Per-week optimization
# =============================================================================

def judge_pct(J_w: np.ndarray, active: np.ndarray) -> np.ndarray:
    s = J_w.copy()
    s[~active] = 0
    t = s.sum()
    return (s / t) if t > 0 else np.zeros_like(s)


def rank_order_asc(values: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Rank ascending (lower value = rank 1)."""
    idx = np.where(active)[0]
    order = idx[np.argsort(values[idx])]
    ranks = np.full(values.shape, np.nan)
    for r, i in enumerate(order, start=1):
        ranks[i] = r
    return ranks


def soft_rank(s: np.ndarray, active: np.ndarray, tau: float = 0.03) -> np.ndarray:
    """Differentiable approx of fan_rank: 1=best (highest s), N=worst. rank_i = 1 + #{j: s_j > s_i}."""
    out = np.zeros_like(s)
    idx = np.where(active)[0]
    if len(idx) < 2:
        out[idx] = 1.0
        return out
    s_a = s[idx]
    for i, ii in enumerate(idx):
        diff = s_a - s_a[i]  # s_j - s_i
        diff[i] = 0
        out[ii] = 1.0 + np.sum(1.0 / (1.0 + np.exp(-diff / tau)))
    return out


def optimize_week(
    q: np.ndarray,
    J_w: np.ndarray,
    judge_rank: np.ndarray,
    active: np.ndarray,
    elim_idx: np.ndarray,
    surv_idx: np.ndarray,
    regime: str,
    season: int,
    max_share_cap: Optional[float] = MAX_SHARE_CAP,
    week: Optional[int] = None,  # For diagnostics
) -> Tuple[np.ndarray, bool]:
    """
    MCM Model: Find vote shares s CLOSEST to prior q_iw that satisfy elimination rules.

    Objective: minimize sum((s - q)^2)  [loss = deviation from prior]
    Constraints:
      - s >= 0, sum(s) = 1
      - Elimination: eliminated contestants have worse combined score than survivors
        * Percent regime (seasons 3-27): combined = judge_% + fan_%; lowest eliminated
        * Rank regime (1-2, 28-34): combined = judge_rank + fan_rank; highest eliminated
          (We use surrogate judge_rank - s b/c true fan_rank = rank(s) is non-differentiable.
           Dominant fan favorites (e.g. Bobby Bones) can have large s; constraints allow this.)

    Stays as true to prior q_iw as possible while fitting regime math.

    When opt succeeds: percent regime yields 100% elim match (constraint is exact).
    Rank regime uses soft_rank; small mismatch vs discrete rank can cause <100%.
    When opt fails (infeasible or no converge): fallback to prior → wrong elims.
    Returns (s_opt, success).
    """
    n = len(q)
    q_active = q[active]
    n_active = int(active.sum())

    def obj(s_active: np.ndarray) -> float:
        s = np.zeros(n)
        s[active] = s_active
        loss = float(np.sum((s - q) ** 2))
        if max_share_cap is not None and max_share_cap > 0:
            excess = np.maximum(0, s - max_share_cap)
            loss += MAX_SHARE_PENALTY * float(np.sum(excess ** 2))
        return loss

    # Bounds: s >= 0
    bounds = Bounds(lb=np.zeros(n_active), ub=np.ones(n_active))

    # sum(s) = 1
    A = np.ones((1, n_active))
    lb_eq = ub_eq = 1.0
    constraints = [{"type": "eq", "fun": lambda s: np.sum(s) - 1}]

    # Elimination constraints: eliminated have worse combined than all survivors
    if len(elim_idx) > 0 and len(surv_idx) > 0:
        if regime == "percent":
            def elim_constraint(s_active: np.ndarray) -> float:
                s = np.zeros(n)
                s[active] = s_active
                jp = judge_pct(J_w, active)
                combined = jp + s
                min_surv = np.min(combined[surv_idx])
                max_elim = np.max(combined[elim_idx])
                return float(min_surv - max_elim - MARGIN)  # Strict: ensures correct prediction
            constraints.append({"type": "ineq", "fun": elim_constraint})
        else:
            # Rank regime: EXACT combined_rank = judge_rank + fan_rank. Eliminated have highest.
            # Use soft_rank (differentiable) to match evaluation.
            def rank_constraint(s_active: np.ndarray) -> float:
                s = np.zeros(n)
                s[active] = s_active
                jr = np.nan_to_num(judge_rank, nan=1e6)
                fr = soft_rank(s, active)
                combined_rank = jr + fr
                min_elim = np.min(combined_rank[elim_idx])
                max_surv = np.max(combined_rank[surv_idx])
                return float(min_elim - max_surv - MARGIN)
            constraints.append({"type": "ineq", "fun": rank_constraint})

    x0 = q_active.copy()
    x0 = x0 / (x0.sum() + EPS)

    try:
        res = minimize(
            obj,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 3000, "ftol": 1e-6},
        )
        s_opt = np.zeros(n)
        s_opt[active] = res.x
        success = res.success
        if not success:
            wstr = f" week {week}" if week is not None else ""
            msg = getattr(res, "message", str(res))[:60]
            warnings.warn(f"Week opt failed (season {season}{wstr}): {msg}")
            s_opt = q.copy()
            s_opt = s_opt / (s_opt.sum() + EPS)
        return s_opt, success
    except Exception as e:
        warnings.warn(f"Week opt error: {e}, using prior q.")
        s_fallback = q.copy()
        s_fallback = s_fallback / (s_fallback.sum() + EPS)
        return s_fallback, False


# =============================================================================
# Season-level processing
# =============================================================================

def process_season(
    df: pd.DataFrame,
    long_df: pd.DataFrame,
    weeks: List[int],
    q_iw: pd.Series,
    judge_matrix: np.ndarray,
    beta: float = BETA,
    k: float = K,
    max_share_cap: Optional[float] = MAX_SHARE_CAP,
) -> Tuple[Dict, np.ndarray, List[bool]]:
    """
    Process one season: optimize each week, return results and s_hist.
    """
    season = int(df["season"].iloc[0])
    names = df["celebrity_name"].tolist()
    elim_week_true = df["elim_week"].astype(int).tolist()
    regime_val = df["regime"].iloc[0]
    W, N = judge_matrix.shape

    s_hist = np.zeros((W, N))
    success_log = []

    for w in range(W):
        week = w + 1
        # Active = competed in week w (including those eliminated at end of week w)
        active = np.array([elim_week_true[i] >= week for i in range(N)], dtype=bool)
        elim_idx = np.array([i for i in range(N) if elim_week_true[i] == week], dtype=int)
        surv_idx = np.array([i for i in range(N) if elim_week_true[i] > week], dtype=int)

        long_week = long_df[(long_df["season"] == season) & (long_df["week"] == week)]
        q = np.zeros(N)
        for i, name in enumerate(names):
            r = long_week[long_week["celebrity_name"] == name]
            if len(r) > 0:
                q[i] = q_iw.loc[r.index[0]]
            else:
                q[i] = 1.0 / max(1, active.sum())
        q[~active] = 0
        if active.sum() > 0:
            q[active] = q[active] / q[active].sum()

        J_w = judge_matrix[w]
        judge_rank_w = np.full(N, np.nan)
        for i, name in enumerate(names):
            r = long_df[(long_df["season"] == season) & (long_df["week"] == week) & (long_df["celebrity_name"] == name)]
            if len(r) > 0 and pd.notna(r["judge_rank"].iloc[0]):
                judge_rank_w[i] = float(r["judge_rank"].iloc[0])
        judge_rank_w = np.nan_to_num(judge_rank_w, nan=1e6)  # 1=best, higher=worse

        s_opt, ok = optimize_week(
            q, J_w, judge_rank_w, active, elim_idx, surv_idx, regime_val, season,
            max_share_cap=max_share_cap, week=week
        )
        s_hist[w] = s_opt
        success_log.append(ok)

    return {"season": season, "names": names, "elim_week_true": elim_week_true, "regime": regime_val}, s_hist, success_log


def build_judge_matrix(df_season: pd.DataFrame, weeks: List[int]) -> np.ndarray:
    max_week = 0
    for w in weeks:
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
        cols = [c for c in cols if c in df_season.columns]
        if cols:
            vals = df_season[cols].replace("N/A", pd.NA)
            numeric = vals.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            if numeric.sum().sum() > 0:
                max_week = w
    W = max_week
    N = len(df_season)
    J = np.zeros((W, N))
    for w in range(1, W + 1):
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
        cols = [c for c in cols if c in df_season.columns]
        vals = df_season[cols].replace("N/A", pd.NA)
        numeric = vals.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        J[w - 1] = numeric.sum(axis=1).clip(lower=0)
    return J


# =============================================================================
# Bootstrap
# =============================================================================

def bootstrap_week(
    q: np.ndarray,
    J_w: np.ndarray,
    judge_rank: np.ndarray,
    active: np.ndarray,
    elim_idx: np.ndarray,
    surv_idx: np.ndarray,
    regime: str,
    season: int,
    rng: np.random.Generator,
    max_share_cap: Optional[float] = MAX_SHARE_CAP,
) -> np.ndarray:
    """Perturb judge_total, recompute judge_rank, re-optimize."""
    J_pert = J_w + rng.normal(0, JUDGE_PERTURB_SD, size=J_w.shape)
    J_pert = np.clip(J_pert, 0, None)
    # Recompute judge rank from perturbed J (1=best, higher=worse)
    jr = np.full_like(J_pert, 1e6)
    jtot = J_pert.copy()
    jtot[~active] = -1e9
    order = np.argsort(-jtot)  # best first
    for rk, idx in enumerate(order):
        if active[idx]:
            jr[idx] = rk + 1
    s_opt, _ = optimize_week(
        q, J_pert, jr, active, elim_idx, surv_idx, regime, season,
        max_share_cap=max_share_cap
    )
    return s_opt


# =============================================================================
# Evaluation
# =============================================================================

def elimination_match_rate(
    df: pd.DataFrame,
    s_hist: np.ndarray,
    judge_matrix: np.ndarray,
    elim_week_true: List[int],
    names: List[str],
    regime_val: str,
) -> float:
    """Fraction of weeks where combined (recomputed) correctly eliminates actual."""
    W, N = s_hist.shape
    hits = 0
    total = 0
    for w in range(W):
        m_w = sum(1 for e in elim_week_true if e == w + 1)
        if m_w <= 0:
            continue
        active = np.array([elim_week_true[i] >= w + 1 for i in range(N)], dtype=bool)
        if active.sum() < 2:
            continue
        J_w = judge_matrix[w]
        s_w = s_hist[w]
        elim_true = np.array([i for i in range(N) if elim_week_true[i] == w + 1])
        surv_true = np.array([i for i in range(N) if elim_week_true[i] > w + 1])

        if regime_val == "percent":
            jp = judge_pct(J_w, active)
            combined = jp + s_w
            order = np.argsort(combined[active])
            active_idx = np.where(active)[0]
            worst_m = active_idx[order[:m_w]]
        else:
            jr = rank_order_asc(-J_w, active)
            jr = np.nan_to_num(jr, nan=0)
            fr = rank_order_asc(-s_w, active)
            fr = np.nan_to_num(fr, nan=0)
            R = jr + fr
            order = np.argsort(-R[active])
            active_idx = np.where(active)[0]
            worst_m = active_idx[order[:m_w]]
            if regime_val == "rank_bottom2" and m_w == 1 and len(worst_m) >= 2:
                bottom2 = active_idx[order[:2]]
                j_scores = J_w[bottom2]
                worst_m = np.array([bottom2[np.argmin(j_scores)]])

        pred_set = set(worst_m)
        true_set = set(elim_true)
        if pred_set == true_set:
            hits += 1
        total += 1
    return hits / total if total > 0 else 0.0


def placement_mse(
    elim_week_true: List[int],
    placement_true: List[Optional[float]],
    s_hist: np.ndarray,
    judge_matrix: np.ndarray,
    regime_val: str,
) -> float:
    """MSE between predicted final ranks and actual placement."""
    W, N = s_hist.shape
    # Simulate: use s_hist and judge to get elim order
    elim_pred = [W + 1] * N
    for w in range(W - 1):
        m_w = sum(1 for e in elim_week_true if e == w + 1)
        if m_w <= 0:
            continue
        active = np.array([elim_pred[i] > w + 1 for i in range(N)], dtype=bool)
        J_w = judge_matrix[w]
        s_w = s_hist[w]
        if regime_val == "percent":
            jp = judge_pct(J_w, active)
            combined = jp + s_w
            order = np.argsort(combined[active])
        else:
            jr = rank_order_asc(-J_w, active)
            fr = rank_order_asc(-s_w, active)
            R = np.nan_to_num(jr, nan=0) + np.nan_to_num(fr, nan=0)
            order = np.argsort(-R[active])
        active_idx = np.where(active)[0]
        elim_idx = active_idx[order[:m_w]]
        if regime_val == "rank_bottom2" and m_w == 1 and len(elim_idx) >= 2:
            bottom2 = active_idx[order[:2]]
            elim_idx = np.array([bottom2[np.argmin(J_w[bottom2])]])
        for i in elim_idx:
            elim_pred[i] = w + 1

    # Placement from elim_week (higher elim_week = better)
    tie_break = s_hist[-1]
    order = np.lexsort((-tie_break, -np.array(elim_pred)))
    placement_pred = np.zeros(N)
    for rank, idx in enumerate(order, start=1):
        placement_pred[idx] = rank

    sq_err = 0
    n = 0
    for i, p_true in enumerate(placement_true):
        if p_true is None or (isinstance(p_true, float) and np.isnan(p_true)):
            continue
        sq_err += (placement_pred[i] - float(p_true)) ** 2
        n += 1
    return sq_err / n if n > 0 else np.nan


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(
    data_path: str = DATA_PATH,
    beta: float = BETA,
    k: float = K,
    boot_n: int = BOOT_N,
    seed: int = SEED,
    max_share_cap: Optional[float] = MAX_SHARE_CAP,
    seasons_limit: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, Dict]:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df, weeks = load_and_preprocess(data_path)
    long_df = build_long_format(df, weeks)
    q_iw = compute_q_iw(df, long_df, weeks, beta=beta, k=k)

    rng = np.random.default_rng(seed)
    all_rows = []
    season_metrics = []
    all_s_hist = {}
    all_boot_p10 = {}
    all_boot_p50 = {}
    all_boot_p90 = {}

    seasons_to_run = sorted(df["season"].unique())
    if seasons_limit is not None:
        seasons_to_run = [s for s in seasons_to_run if s in seasons_limit]
    for season in seasons_to_run:
        df_season = df[df["season"] == season].reset_index(drop=True)
        judge_matrix = build_judge_matrix(df_season, weeks)
        long_season = long_df[long_df["season"] == season]
        q_season = q_iw.loc[long_season.index]

        meta, s_hist, _ = process_season(
            df_season, long_df, weeks, q_iw, judge_matrix, beta=beta, k=k,
            max_share_cap=max_share_cap if max_share_cap and max_share_cap > 0 else None
        )
        all_s_hist[season] = s_hist

        # Bootstrap
        W, N = s_hist.shape
        boot_s = np.zeros((boot_n, W, N))
        for b in range(boot_n):
            for w in range(W):
                week = w + 1
                active = np.array([meta["elim_week_true"][i] >= week for i in range(N)], dtype=bool)
                elim_idx = np.array([i for i in range(N) if meta["elim_week_true"][i] == week])
                surv_idx = np.array([i for i in range(N) if meta["elim_week_true"][i] > week], dtype=int)
                q = np.zeros(N)
                for i, name in enumerate(meta["names"]):
                    r = long_df[(long_df["season"] == season) & (long_df["week"] == week) & (long_df["celebrity_name"] == name)]
                    if len(r) > 0:
                        q[i] = q_iw.loc[r.index[0]]
                    else:
                        q[i] = 1.0 / max(1, active.sum())
                q[~active] = 0
                if active.sum() > 0:
                    q[active] = q[active] / q[active].sum()
                J_w = judge_matrix[w]
                jr = np.zeros(N)
                jtot = J_w.copy()
                jtot[~active] = -1e9
                for i in np.argsort(-jtot):
                    if active[i]:
                        jr[i] = np.sum(active)
                boot_s[b, w] = bootstrap_week(
                    q, J_w, jr, active, elim_idx, surv_idx, meta["regime"], season, rng,
                    max_share_cap=max_share_cap if max_share_cap and max_share_cap > 0 else None
                )
        all_boot_p10[season] = np.percentile(boot_s, 10, axis=0)
        all_boot_p50[season] = np.percentile(boot_s, 50, axis=0)
        all_boot_p90[season] = np.percentile(boot_s, 90, axis=0)

        # Evaluation
        placement_true = df_season["placement"].tolist()
        match_rate = elimination_match_rate(
            df_season, s_hist, judge_matrix, meta["elim_week_true"], meta["names"], meta["regime"]
        )
        mse = placement_mse(
            meta["elim_week_true"], placement_true, s_hist, judge_matrix, meta["regime"]
        )
        season_metrics.append({
            "season": season,
            "match_rate": match_rate,
            "placement_MSE": mse,
        })

        for w in range(W):
            for i, name in enumerate(meta["names"]):
                r = long_df[(long_df["season"] == season) & (long_df["week"] == w + 1) & (long_df["celebrity_name"] == name)]
                q_val = float(q_iw.loc[r.index[0]]) if len(r) > 0 else np.nan
                p10 = float(all_boot_p10[season][w, i])
                p90 = float(all_boot_p90[season][w, i])
                p_true = placement_true[i] if placement_true[i] is not None and not (isinstance(placement_true[i], float) and np.isnan(placement_true[i])) else 999
                all_rows.append({
                    "season": season,
                    "week": w + 1,
                    "celebrity_name": name,
                    "s_hat": float(s_hist[w, i]),
                    "p10": p10,
                    "p50": float(all_boot_p50[season][w, i]),
                    "p90": p90,
                    "certainty_interval_width": p90 - p10,
                    "q_iw": q_val,
                    "elim_week": meta["elim_week_true"][i],
                    "placement": p_true,
                })

    fan_est = pd.DataFrame(all_rows)
    metrics_df = pd.DataFrame(season_metrics)

    # Overall
    match_rates = [m["match_rate"] for m in season_metrics]
    mses = [m["placement_MSE"] for m in season_metrics if not np.isnan(m["placement_MSE"])]
    overall_match = np.mean(match_rates) if match_rates else 0
    overall_mse = np.mean(mses) if mses else np.nan

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "fan_vote_estimates.csv")
    fan_est.to_csv(csv_path, index=False)

    # Log
    log_path = os.path.join(OUT_DIR, "run_log.txt")
    with open(log_path, "w") as f:
        f.write(f"match_rate (overall): {overall_match:.4f}\n")
        f.write(f"placement_MSE (overall): {overall_mse:.4f}\n")
        f.write(f"params: beta={beta}, k={k}, boot_n={boot_n}\n")
    # MCM-aligned summary: answers Problem C questions
    mcm_path = os.path.join(OUT_DIR, "mcm_answers_summary.md")
    ci_widths = (fan_est["p90"] - fan_est["p10"]).dropna()
    ci_mean, ci_std = float(ci_widths.mean()), float(ci_widths.std()) if len(ci_widths) > 0 else (0.0, 0.0)
    ci_min, ci_max = float(ci_widths.min()), float(ci_widths.max()) if len(ci_widths) > 0 else (0.0, 0.0)
    with open(mcm_path, "w") as f:
        f.write("# MCM 2026 Problem C: Fan Vote Estimation – Model Answers\n\n")
        f.write("## 1. Estimated fan votes consistent with eliminations?\n")
        f.write(f"- **Elimination match rate**: {overall_match:.2%} of weeks where predicted eliminations match actual.\n")
        f.write(f"- **Placement MSE**: {overall_mse:.4f} (mean squared error of predicted vs actual final placement).\n\n")
        f.write("## 2. Certainty in fan vote totals – same for each contestant/week?\n")
        f.write(f"- **No.** Certainty varies. Bootstrap 80% interval width (p90−p10): mean={ci_mean:.4f}, std={ci_std:.4f}, min={ci_min:.4f}, max={ci_max:.4f}.\n")
        f.write("- Contestants with tighter constraints (e.g. clear elim) have narrower intervals; marginal cases wider.\n\n")
        f.write("## 3. Model logic\n")
        f.write("- Objective: minimize sum((s − q_iw)²) to stay closest to prior.\n")
        f.write("- Constraints: s ≥ 0, sum(s)=1, and regime elimination rules.\n")
    print(f"match_rate: {overall_match:.4f}")
    print(f"placement_MSE: {overall_mse:.4f}")
    print(f"Saved {csv_path}")

    return fan_est, metrics_df, overall_match, overall_mse, {
        "s_hist": all_s_hist,
        "p10": all_boot_p10,
        "p50": all_boot_p50,
        "p90": all_boot_p90,
        "metrics": season_metrics,
    }


# =============================================================================
# Graphs
# =============================================================================

def plot_stacked_bar(season: int, fan_est: pd.DataFrame, out_dir: str = FIG_DIR) -> None:
    """Stacked area plot: eliminated first (bottom), finalists on top. Matches AR-Problem1 style."""
    sub = fan_est[fan_est["season"] == season].copy()
    if sub.empty:
        return
    # Need elim_week and placement for ordering (like AR-Problem1)
    if "elim_week" not in sub.columns or "placement" not in sub.columns:
        return
    weeks = sorted(sub["week"].unique())
    # One row per contestant with elim_week, placement
    place = sub.drop_duplicates("celebrity_name")[["celebrity_name", "elim_week", "placement"]]
    order = place.sort_values(["elim_week", "placement"], ascending=[True, False])["celebrity_name"].tolist()
    names = order
    data = np.zeros((len(names), len(weeks)))
    for j, w in enumerate(weeks):
        rw = sub[sub["week"] == w].set_index("celebrity_name")
        for i, n in enumerate(names):
            data[i, j] = rw.loc[n, "s_hat"] if n in rw.index else 0.0
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(weeks))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(names)))
    for i in range(len(names)):
        ax.fill_between(weeks, bottom, bottom + data[i], color=colors[i], alpha=0.85, linewidth=0)
        bottom = bottom + data[i]
    ax.set_xlim(weeks[0] - 0.3, weeks[-1] + 0.3)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative fan share")
    ax.set_title(f"Season {season}: Estimated fan vote share (stacked)")
    leg_ord = place.sort_values("placement")["celebrity_name"].tolist()
    leg_colors = [colors[names.index(n)] for n in leg_ord]
    handles = [Patch(color=c, label=n[:20]) for c, n in zip(leg_colors, leg_ord)]
    ax.legend(handles=handles, fontsize=6, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()
    path = os.path.join(out_dir, f"vote_shares_season_{season:02d}_stacked.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_consistency(metrics_df: pd.DataFrame, out_dir: str = FIG_DIR) -> None:
    fig, ax1 = plt.subplots(figsize=(8, 4))
    x = metrics_df["season"].values
    ax1.bar(x - 0.2, metrics_df["match_rate"], 0.4, label="Elim match rate", color="steelblue")
    ax1.set_ylabel("Match rate")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, metrics_df["placement_MSE"], 0.4, label="Placement MSE", color="coral", alpha=0.7)
    ax2.set_ylabel("Placement MSE")
    ax1.set_xlabel("Season")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_title("Consistency by season")
    plt.tight_layout()
    # PNG for preview; PDF for LaTeX (font embedding for clean rendering)
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42}):
        fig.savefig(os.path.join(out_dir, "consistency_by_season.png"), bbox_inches="tight", dpi=150)
        fig.savefig(os.path.join(out_dir, "consistency_by_season.pdf"), bbox_inches="tight", format="pdf")
    plt.close(fig)


def plot_uncertainty(
    season: int,
    fan_est: pd.DataFrame,
    out_dir: str = FIG_DIR,
) -> None:
    sub = fan_est[fan_est["season"] == season].copy()
    if sub.empty:
        return
    names = sub["celebrity_name"].unique().tolist()
    weeks = sorted(sub["week"].unique())
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in names[:10]:
        r = sub[sub["celebrity_name"] == name].sort_values("week")
        ax.fill_between(r["week"], r["p10"], r["p90"], alpha=0.2)
        ax.plot(r["week"], r["p50"], label=name[:20], linewidth=1.5)
    ax.set_xlabel("Week")
    ax.set_ylabel("Fan vote share")
    ax.set_title(f"Uncertainty (p10–p90) Season {season}")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(out_dir, f"uncertainty_p90_season_{season}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="MCM 2026 P1: Fan vote share estimation (constrained opt)")
    parser.add_argument("--beta", type=float, default=BETA, help="Blended prior weight for placement_boost")
    parser.add_argument("--k", type=float, default=K, help="Weekly proxy deficit coefficient")
    parser.add_argument("--max-share", type=float, default=MAX_SHARE_CAP, help="Soft cap on max share (0.55); use 0 to disable")
    parser.add_argument("--boot-n", type=int, default=BOOT_N, help="Bootstrap iterations per week")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--data", type=str, default=DATA_PATH, help="Path to 2026_MCM_Problem_C_Data.csv")
    parser.add_argument("--seasons", type=int, nargs="+", default=None, help="Limit to these seasons (default: all)")
    args = parser.parse_args()

    max_cap = args.max_share if args.max_share and args.max_share > 0 else None
    fan_est, metrics_df, match_rate, mse, data = run_pipeline(
        data_path=args.data,
        beta=args.beta, k=args.k, boot_n=args.boot_n, seed=args.seed,
        max_share_cap=max_cap,
        seasons_limit=args.seasons,
    )
    print("\nSample DF head:")
    print(fan_est.head(10).to_string())

    # Graphs: plot first 5 + key seasons (27 Bobby Bones, 34)
    all_seasons = sorted(fan_est["season"].unique())
    seasons_plot = list(all_seasons[:5])
    for s in [27, 34]:
        if s in all_seasons and s not in seasons_plot:
            seasons_plot.append(s)
    for season in seasons_plot:
        plot_stacked_bar(int(season), fan_est)
    plot_consistency(metrics_df)
    for s in [1, 27, 34]:
        if s in fan_est["season"].values:
            plot_uncertainty(int(s), fan_est)
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
