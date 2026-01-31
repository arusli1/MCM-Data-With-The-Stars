"""
Inverse Optimization (InvOpt) reconstruction of DWTS fan vote shares.

High-level goal
---------------
We observe:
  - weekly judge scores for each contestant (from `Data/2026_MCM_Problem_C_Data.csv`)
  - which contestant is eliminated each week (parsed from the `results` column)

We do NOT observe:
  - weekly fan vote shares.

This script infers (reconstructs) weekly fan vote shares s[i,w] that are consistent with the
show’s elimination rules under multiple eras ("regimes"). Where the constraints do not uniquely
identify shares, we add priors/regularizers to select a plausible solution.

Important interpretation note
-----------------------------
The "popularity prior" used here is computed from realized elimination timing (future information).
That makes this approach appropriate for *reconstruction* / *inverse inference*, NOT for
out-of-sample prediction. High elimination "match rates" therefore indicate constraint satisfaction,
not predictive power.

Regimes (by season)
-------------------
  - seasons 1–2: "rank"      (combined judge-rank + fan-rank)
  - seasons 3–27: "percent"  (combined percent score j_pct + fan share s)
  - seasons 28+: "bottom"    (eliminated must be in bottom-two by combined ranks; details simplified)

Outputs
-------
Writes two CSVs:
  - OUT_PATH: inferred weekly shares (long format): season, week, celebrity_name, s_map
  - MATCH_PATH: season-level diagnostics: match_rate, kendall_tau, mean_abs_rank_diff, ...

File structure note (very important)
------------------------------------
This file contains multiple historical/experimental variants appended AFTER the first `main()`
and an explicit `raise SystemExit(0)`. Only the first block (above the first `raise SystemExit(0)`)
will execute when you run the script.
"""

import re

import cvxpy as cp
import numpy as np
import pandas as pd

try:
    import pulp as pl
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install pulp") from exc


# -----------------------------
# Configuration / file paths
# -----------------------------
DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1/inferred_shares.csv"
MATCH_PATH = "AR-Problem1/elimination_match.csv"

# -----------------------------
# Numerical tolerances / big-M
# -----------------------------
# EPS_SHARE enforces a strict inequality (eliminated must be *strictly* worse) in percent seasons.
EPS_SHARE = 1e-6
# EPS_RANK is used in MILP rank comparisons; set to 0.0 => allow ties.
EPS_RANK = 0.0
# Big-M constant for MILP formulations (must be "large enough" to relax constraints when binary flips).
M_BIG = 50.0

# -----------------------------
# Hyperparameters / priors
# -----------------------------
# Convex weights (percent seasons / QP)
LAM_SMOOTH = 1.0
LAM_POP = 1.0
# FAST_BOTTOM: try heuristic first for bottom-two era before falling back to a full MILP
FAST_BOTTOM = True
# ALPHA_PREV controls temporal persistence in rank/bottom seasons (target blends previous-week shares)
ALPHA_PREV = 0.85
# GAMMA_POP controls sharpness of the "popularity prior" (softmax temperature)
GAMMA_POP = 0.7


def parse_week_cols(df):
    """
    Identify all judge-score columns in the wide DWTS CSV.

    Expected format: week{w}_judge{j}_score
    """
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df, week, cols):
    """
    Compute each contestant's total judge score for a given week by summing all judge columns for that week.
    Missing values (N/A) are treated as NaN by the CSV reader and are skipped by sum().
    """
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def elimination_week(row, last_active_week):
    """
    Parse elimination timing from the 'results' column.

    - If row['results'] contains 'Eliminated Week k', returns k.
    - If 'Withdrew', treat as eliminated at last_active_week (last week with positive judge score).
    - Otherwise (finalists/winner/etc.), returns None.
    """
    if isinstance(row["results"], str) and "Eliminated Week" in row["results"]:
        return int(row["results"].split("Eliminated Week ")[1])
    if isinstance(row["results"], str) and "Withdrew" in row["results"]:
        return last_active_week
    return None


def regime_for_season(season):
    """
    Determine the scoring/elimination regime for the season.

    This drives:
      - which elimination constraints we impose,
      - which optimizer type we use (QP vs MILP),
      - how we evaluate weekly elimination consistency.
    """
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom"


def build_season_struct(df_season, week_cols):
    """
    Convert a season's wide-format dataframe into a structured representation:

    - names: contestant names in row order
    - J:     week x contestant matrix of judge totals (only up to last active week)
    - elim_week: list of elimination week indices per contestant (None for finalists)
    - max_week: number of active weeks in this season

    Key detail:
      We truncate J to `max_week_active`, the last week where anyone has a positive total.
    """
    max_week = max(int(re.search(r"week(\d+)_", c).group(1)) for c in week_cols)
    names = df_season["celebrity_name"].tolist()
    n = len(names)
    J = np.zeros((max_week, n), dtype=float)

    for w in range(1, max_week + 1):
        J[w - 1] = week_score(df_season, w, week_cols).to_numpy()

    # last_active[w,i] (in week numbers) is the last week where contestant i has J>0.
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


def popularity_prior(df_season, idx):
    """
    Build a "popularity" distribution over the currently-active contestants `idx`.

    Steps (per season):
      1) Compute each contestant's average judge score across weeks where they have positive scores.
      2) Compute each contestant's actual elimination timing (last week with positive scores).
      3) Fit a per-season linear model: expected_elim_week ~ avg_judge_score
      4) Define pop_score = actual_elim_week - expected_elim_week  (survival beyond judge expectation)
      5) Softmax(pop_score) with temperature GAMMA_POP to get a probability distribution.

    Critical caveat:
      Steps (2)–(4) use realized elimination timing (future outcomes),
      so this prior is reconstruction-biased by design.
    """
    week_cols = [c for c in df_season.columns if re.match(r"week\d+_judge\d+_score", c)]
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

    # Actual elimination week proxy = last week with positive judge scores.
    actual_week = np.where(pos_mask.any(axis=0), pos_mask.sum(axis=0), np.nan)

    valid = [
        i for i in idx if not np.isnan(avg_judge[i]) and not np.isnan(actual_week[i])
    ]
    if not valid:
        return {i: 1.0 / len(idx) for i in idx}

    # Judge-only expected elimination week via per-season linear fit.
    # This is an intentionally simplistic "baseline" model for reconstruction.
    x = avg_judge[valid].astype(float)
    y = actual_week[valid].astype(float)
    if np.unique(x).size < 2:
        pred = np.full_like(y, y.mean(), dtype=float)
    else:
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept

    # Fan support proxy = survival beyond judge-only expectation.
    # Positive => survived longer than expected from judges alone.
    pop_score = y - pred
    scale = pop_score.std()
    if scale > 0:
        pop_score = pop_score / scale
    weights = np.exp(GAMMA_POP * pop_score)
    pop = {i: 0.0 for i in idx}
    for k, i in enumerate(valid):
        pop[i] = weights[k]
    total = sum(pop.values())
    if total > 0:
        pop = {i: pop[i] / total for i in idx}
    else:
        pop = {i: 1.0 / len(idx) for i in idx}
    return pop


def blended_target(df_season, idx, last_shares, use_prev):
    """
    Construct the per-week target share distribution that the optimizer prefers.

    - Always starts from the popularity_prior distribution.
    - If use_prev=True and last_shares is available:
        target = ALPHA_PREV * last_week_shares + (1 - ALPHA_PREV) * popularity_prior
      This induces temporal continuity for rank/bottom seasons where we solve week-by-week.

    Returns a normalized distribution over idx.
    """
    pop_dist = popularity_prior(df_season, idx)
    if use_prev and last_shares is not None:
        prev = {i: last_shares.get(i, 0.0) for i in idx}
        target = {
            i: ALPHA_PREV * prev[i] + (1 - ALPHA_PREV) * pop_dist[i]
            for i in idx
        }
    else:
        target = {i: pop_dist[i] for i in idx}
    total = sum(target.values())
    if total > 0:
        return {i: target[i] / total for i in idx}
    return {i: 1.0 / len(idx) for i in idx}


def solve_percent_season(df_season, week_cols):
    """
    Solve percent-regime seasons (S3–S27) as a single convex Quadratic Program (QP).

    Variables:
      s[w,i] = fan vote share for contestant i in week w.

    Constraints per week:
      - s[w,i] >= 0
      - sum_{i in active_w} s[w,i] = 1
      - s[w,i] = 0 for non-active contestants
      - elimination hard constraints:
           for each eliminated e and each non-eliminated j:
               j_pct[e] + s[w,e] <= j_pct[j] + s[w,j] - EPS_SHARE

    Objective:
      - LAM_POP   * || s[w,active] - popularity_prior(active) ||^2
      - LAM_SMOOTH* || s[w,shared] - s[w-1,shared] ||^2
    """
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape

    active = []
    for w in range(W):
        if J[w].sum() == 0:
            active.append([])
            continue
        # Active set in week w: contestants not yet eliminated (or finalists).
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        active.append(idx)

    # Decision variables: s is a W x N matrix of weekly fan shares.
    s = cp.Variable((W, N))
    constraints = [s >= 0]
    obj_terms = []

    last_shares = None
    for w in range(W):
        idx = active[w]
        if not idx:
            constraints.append(s[w, :] == 0)
            continue
        # Shares must form a simplex over active contestants.
        constraints.append(cp.sum(s[w, idx]) == 1)
        if len(idx) < N:
            constraints.append(s[w, [i for i in range(N) if i not in idx]] == 0)

        # "Popularity" target for this week (use_prev=False in percent regime to avoid double-counting smoothness).
        target = blended_target(df_season, idx, last_shares, use_prev=False)
        target_vec = np.array([target[i] for i in idx])
        obj_terms.append(LAM_POP * cp.sum_squares(s[w, idx] - target_vec))

        # Elimination constraint indices: contestants eliminated at end of this week.
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        if not elim:
            continue
        non_elim = [i for i in idx if i not in elim]
        if not non_elim:
            continue

        # j_pct is the judge percent contribution for each active contestant this week.
        j_pct = J[w, idx] / J[w, idx].sum()
        j_map = {i: j_pct[k] for k, i in enumerate(idx)}
        for e in elim:
            for j in non_elim:
                # Enforce eliminated has strictly lower combined score (judge percent + fan share).
                constraints.append(
                    j_map[e] + s[w, e] <= j_map[j] + s[w, j] - EPS_SHARE
                )

    # Smoothness: penalize week-to-week changes among contestants present in both weeks.
    for w in range(1, W):
        prev = set(active[w - 1])
        curr = set(active[w])
        shared = list(prev & curr)
        if shared:
            obj_terms.append(LAM_SMOOTH * cp.sum_squares(s[w, shared] - s[w - 1, shared]))

    # Solve convex QP with OSQP (fast, global optimum for convex problems).
    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Season {df_season['season'].iloc[0]}: {prob.status}")
    s_val = np.clip(s.value, 0, 1)
    return s_val, struct


def solve_rank_week(Jw, elim_idx, idx, target=None):
    """
    Solve ONE week for a rank-regime season (S1–S2) as a MILP.

    Inputs
    ------
    Jw: full-length vector of judge totals for the week (indexed by contestant id)
    elim_idx: list of eliminated contestant ids for this week (usually length 1)
    idx: list of active contestant ids (those still in competition this week)
    target: desired fan-share vector (dict over idx) we want to stay close to

    Decision variables
    ------------------
    - s_i in [0,1]: fan share for each active contestant i
    - u_i >= 0: L1 slack to measure |s_i - target_i|
    - y_{i,j} in {0,1}: pairwise ordering indicators used to compute fan ranks
        y_{i,j} = 1 implies s_i >= s_j (within EPS_RANK), else s_i <= s_j

    Constraints
    -----------
    - Simplex: sum_i s_i = 1
    - L1 distance encoding: u_i >= (s_i - target_i) and u_i >= -(s_i - target_i)
    - Pairwise rank encoding using y_{i,j}
    - Combined rank: R_i = rJ_i + rF_i, where:
        rJ_i is the ordinal rank of judge totals among idx (1=best)
        rF_i is implied from y_{i,j} (higher rank number = worse fan support)
    - Elimination constraint: eliminated contestants must be at least as "bad" as any survivor:
        R_e >= R_j for all e in elim_idx and j in non_elim

    Objective
    ---------
    Minimize sum_i u_i (L1 distance to the target distribution).
    """
    n = len(idx)
    prob = pl.LpProblem("rank_week", pl.LpMinimize)
    # Fan share decision variables for active contestants this week.
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    # L1 deviation variables (|s_i - target_i|).
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

    # Default target is uniform if none provided.
    if target is None:
        target = {i: 1.0 / n for i in idx}
    for i in idx:
        prob += u[i] >= s[i] - target[i]
        prob += u[i] >= -(s[i] - target[i])

    # Pairwise ordering binaries used to compute fan ranks.
    y = {}
    for i in idx:
        for j in idx:
            if i == j:
                continue
            y[i, j] = pl.LpVariable(f"y_{i}_{j}", cat="Binary")
            prob += s[i] - s[j] >= EPS_RANK - (1 - y[i, j])
            prob += s[i] - s[j] <= -EPS_RANK + y[i, j]

    # Judge rank (rJ): 1=best judge score among idx.
    rJ = {
        i: int(pd.Series(Jw).rank(ascending=False, method="first").to_numpy()[i])
        for i in idx
    }
    # Fan rank (rF): derived from pairwise wins (higher rF = worse fan share rank).
    rF = {i: n - pl.lpSum(y[i, j] for j in idx if j != i) for i in idx}
    # Combined rank: higher is worse.
    R = {i: rJ[i] + rF[i] for i in idx}

    non_elim = [i for i in idx if i not in elim_idx]
    for e in elim_idx:
        for j in non_elim:
            prob += R[e] >= R[j]

    # Objective: minimize L1 deviation from target shares.
    prob += pl.lpSum(u.values())
    # Solve with CBC (default open-source MILP solver shipped with PuLP).
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    return {i: pl.value(s[i]) for i in idx}


def solve_bottom_week(Jw, elim_idx, idx, target=None):
    """
    Solve ONE week for a bottom-two regime (S28+) as a MILP.

    Goal:
      Find fan shares s_i close to target such that the eliminated contestant(s) are constrained
      to be in the bottom set by combined rank R = rJ + rF.

    Key difference vs rank regime:
      - In bottom-two seasons (with judges' save etc.), the observed eliminated must be among
        the bottom-two by combined rank (not necessarily the single worst).
      - This script approximates the era by enforcing bottom-two membership (and bottom-k for double elims).

    Implementation:
      - We reuse the same rank construction as solve_rank_week.
      - Introduce binary indicators z_i for whether i is in the bottom set.
      - Enforce sum_i z_i = bottom_k.
      - Force eliminated e to have z_e = 1.
    """
    n = len(idx)
    prob = pl.LpProblem("bottom_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

    # Default target is uniform if none provided.
    if target is None:
        target = {i: 1.0 / n for i in idx}
    for i in idx:
        prob += u[i] >= s[i] - target[i]
        prob += u[i] >= -(s[i] - target[i])

    # Pairwise ordering binaries for fan rank.
    y = {}
    for i in idx:
        for j in idx:
            if i == j:
                continue
            y[i, j] = pl.LpVariable(f"y_{i}_{j}", cat="Binary")
            prob += s[i] - s[j] >= EPS_RANK - (1 - y[i, j])
            prob += s[i] - s[j] <= -EPS_RANK + y[i, j]

    rJ = {
        i: int(pd.Series(Jw).rank(ascending=False, method="first").to_numpy()[i])
        for i in idx
    }
    rF = {i: n - pl.lpSum(y[i, j] for j in idx if j != i) for i in idx}
    R = {i: rJ[i] + rF[i] for i in idx}

    k_elim = len(elim_idx)
    bottom_k = 2 if k_elim == 1 else k_elim
    # z_i = 1 if contestant i is in the "bottom_k" set by rank R.
    z = {i: pl.LpVariable(f"z_{i}", cat="Binary") for i in idx}
    # t is a threshold used with big-M constraints to approximate selecting bottom_k items.
    t = pl.LpVariable("t", lowBound=0)
    prob += pl.lpSum(z[i] for i in idx) == bottom_k
    for i in idx:
        prob += R[i] <= t + M_BIG * z[i]
        prob += R[i] >= t - M_BIG * (1 - z[i])
    for e in elim_idx:
        prob += z[e] == 1

    # Objective: minimize L1 deviation from target shares.
    prob += pl.lpSum(u.values())
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    return {i: pl.value(s[i]) for i in idx}


def solve_bottom_week_fast(Jw, elim_idx, idx, target=None):
    """
    Heuristic solver for bottom-two seasons (S28+), used as a fast approximation.

    This avoids solving a full MILP by:
      - Enumerating a choice for the second bottom contestant (the first is the observed eliminated),
      - Constructing a deterministic fan-rank assignment consistent with that choice,
      - Converting ranks into a simple share vector (higher weight for worse fan rank),
      - Selecting the candidate that both:
           (a) includes the eliminated in the bottom-two by combined rank R = rJ + rF, and
           (b) minimizes L1 distance to the provided target distribution (if target exists).

    This is not guaranteed optimal; it’s a speed/robustness tradeoff.
    """
    n = len(idx)
    if n < 2:
        return None
    rJ = {
        i: int(pd.Series(Jw).rank(ascending=False, method="first").to_numpy()[i])
        for i in idx
    }

    def assign_fan_ranks(second_idx):
        remaining = [i for i in idx if i not in {elim_idx[0], second_idx}]
        remaining.sort(key=lambda i: rJ[i], reverse=True)
        ranks = {}
        ranks[elim_idx[0]] = n
        ranks[second_idx] = n - 1
        for r, i in enumerate(remaining, start=1):
            ranks[i] = r
        return ranks

    if not elim_idx:
        return None

    candidates = [i for i in idx if i not in elim_idx]
    if not candidates:
        return None

    if target is not None:
        total = sum(target.values())
        target = {i: (target[i] / total if total > 0 else 1.0 / n) for i in idx}

    best = None
    best_dist = None
    for second in candidates:
        rF = assign_fan_ranks(second)
        R = {i: rJ[i] + rF[i] for i in idx}
        worst_two = sorted(idx, key=lambda i: R[i], reverse=True)[:2]
        if elim_idx[0] in worst_two:
            weights = {i: (n - rF[i] + 1) for i in idx}
            total = sum(weights.values())
            shares = {i: weights[i] / total for i in idx}
            if target is None:
                return shares
            dist = sum(abs(shares[i] - target[i]) for i in idx)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = shares
    if best is not None:
        return best
    return None


def solve_rank_bottom_season(df_season, week_cols, regime):
    """
    Solve rank-regime or bottom-two-regime seasons (S1–S2 or S28+) week-by-week.

    Unlike percent seasons (which are solved in one global QP across all weeks),
    rank/bottom seasons are solved sequentially per week because rank constraints are discrete.

    Workflow per week:
      - Determine active contestants (not yet eliminated).
      - Build a target distribution (blend of last week + popularity prior).
      - If the week has eliminations:
          solve a MILP (rank) or heuristic+MILP fallback (bottom).
        else:
          copy the target distribution into shares for that week.
      - Store last_shares for temporal continuity.
    """
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    shares = np.zeros((W, N), dtype=float)
    last_shares = None

    for w in range(W):
        if J[w].sum() == 0:
            continue
        # Active contestants are those whose elimination week is >= current week (or None for finalists).
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        if len(idx) < 2:
            continue
        # Contestants eliminated at end of this week.
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        # Target shares incorporate temporal continuity for these regimes.
        target = blended_target(df_season, idx, last_shares, use_prev=True)
        if not elim:
            # No elimination: carry forward the target distribution (constraints provide no new info).
            for i in idx:
                shares[w, i] = target[i]
            continue

        if regime == "rank":
            sol = solve_rank_week(J[w], elim, idx, target=target)
        else:
            # Bottom-two: try heuristic first (faster); if it fails, solve the MILP.
            sol = (
                solve_bottom_week_fast(J[w], elim, idx, target=target)
                if FAST_BOTTOM
                else None
            )
            if sol is None:
                sol = solve_bottom_week(J[w], elim, idx, target=target)
        if sol is None:
            # Solver infeasible: fall back to target distribution (never uniform in this variant).
            for i in idx:
                shares[w, i] = target[i]
        else:
            for i, val in sol.items():
                shares[w, i] = val
            # Update last_shares only over active indices (used in next week's target blend).
            last_shares = {i: shares[w, i] for i in idx}
    return shares, struct


def elimination_match_rate(shares, struct, regime):
    """
    Compute a week-weighted "match" rate for elimination consistency.

    For each week with >=2 active contestants:
      - If no eliminations are recorded, treat as a "hit" (no constraint to violate).
      - Otherwise compute the model's implied risk ordering and check whether the
        eliminated set is a subset of the predicted bottom set.

    Percent regime:
      - Compute C_i = j_pct_i + s_i and predict the bottom-k by smallest C.

    Rank regime:
      - Compute combined rank R_i = rJ_i + rF_i and predict worst by largest R.

    Bottom-two regime:
      - For single-elim weeks, predict bottom-two membership (worst 2 by R).
    """
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    hits = 0
    total = 0

    for w in range(W):
        idx = np.where(J[w] > 0)[0]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        total += 1
        if not elim:
            hits += 1
            continue

        k_elim = len(elim)

        if regime == "percent":
            j_pct = J[w, idx] / J[w, idx].sum()
            C = np.full(N, np.inf)
            for k, i in enumerate(idx):
                C[i] = j_pct[k] + shares[w, i]
            pred = set(np.argsort(C)[:k_elim])
        else:
            # Rank-based evaluation (rank & bottom regimes)
            rF = np.argsort(-shares[w, idx]).argsort() + 1
            rJ = (
                pd.Series(J[w, idx])
                .rank(ascending=False, method="first")
                .to_numpy()
            )
            R = rJ + rF
            if regime == "bottom" and k_elim == 1:
                worst = np.argsort(-R)[:2]
                pred = set(idx[worst])
            else:
                worst = np.argsort(-R)[:k_elim]
                pred = set(idx[worst])

        if set(elim).issubset(pred):
            hits += 1
    return hits, total


def kendall_tau(order_a, order_b):
    """
    Compute Kendall-like tau between two full orderings (no ties).

    This implementation counts concordant vs discordant pairs by comparing relative order
    between list `order_a` and list `order_b`.
    """
    pos_b = {k: i for i, k in enumerate(order_b)}
    concordant = 0
    discordant = 0
    items = order_a
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a_i, a_j = items[i], items[j]
            if (i - j) * (pos_b[a_i] - pos_b[a_j]) > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return (concordant - discordant) / denom if denom else 0.0


def season_rank_metrics(shares, struct, df_season, regime):
    """
    Season-level ranking diagnostics (not week-by-week elimination).

    We compress weekly information into a per-contestant "average score" and compare:
      - Model-implied ordering vs actual placement ordering.

    Percent regime:
      - avg_score_i = average over active weeks of (j_pct_i + s_i)
      - Higher avg_score => better, so predicted order sorts descending avg_score.

    Rank/bottom regimes:
      - avg_score_i = average over active weeks of combined rank R_i (lower is better)
      - Predicted order sorts ascending avg_score.

    Outputs:
      - kendall_tau: agreement between predicted vs actual placement ordering
      - mean_abs_rank_diff: mean |pred_rank - actual_placement|
    """
    J = struct["J"]
    W, N = J.shape
    placements = (
        pd.to_numeric(df_season["placement"], errors="coerce").to_numpy()
    )
    valid_ids = [i for i in range(N) if not np.isnan(placements[i])]
    if len(valid_ids) < 2:
        return 0.0, 0.0

    scores = np.zeros(N)
    counts = np.zeros(N)
    for w in range(W):
        idx = np.where(J[w] > 0)[0]
        if len(idx) < 2:
            continue
        if regime == "percent":
            j_pct = J[w, idx] / J[w, idx].sum()
            for k, i in enumerate(idx):
                scores[i] += j_pct[k] + shares[w, i]
                counts[i] += 1
        else:
            rF = np.argsort(-shares[w, idx]).argsort() + 1
            rJ = (
                pd.Series(J[w, idx])
                .rank(ascending=False, method="first")
                .to_numpy()
            )
            R = rJ + rF
            for k, i in enumerate(idx):
                scores[i] += R[k]
                counts[i] += 1

    avg_score = np.where(counts > 0, scores / counts, np.nan)
    valid_ids = [i for i in valid_ids if not np.isnan(avg_score[i])]
    if len(valid_ids) < 2:
        return 0.0, 0.0

    if regime == "percent":
        pred_order = sorted(valid_ids, key=lambda i: -avg_score[i])
    else:
        pred_order = sorted(valid_ids, key=lambda i: avg_score[i])
    place_order = sorted(valid_ids, key=lambda i: placements[i])

    tau = kendall_tau(pred_order, place_order)
    pred_rank = {k: i + 1 for i, k in enumerate(pred_order)}
    mean_abs = np.mean([abs(pred_rank[i] - placements[i]) for i in valid_ids])
    return tau, mean_abs


def main():
    """
    Entry point: solve all seasons, write outputs, and compute season diagnostics.

    Steps:
      1) Load the DWTS wide CSV.
      2) Identify week score columns.
      3) For each season:
          - choose regime
          - solve shares (QP for percent; week-by-week MILP/heuristic for rank/bottom)
          - compute elimination match diagnostics + season rank diagnostics
          - store weekly share records (long format)
      4) Write inferred shares + diagnostics to CSV.
    """
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    week_cols = parse_week_cols(df)

    records = []
    match_rows = []
    for season in sorted(df["season"].unique()):
        df_season = df[df["season"] == season].reset_index(drop=True)
        regime = regime_for_season(season)
        print(f"Solving season {season} ({regime})...")

        if regime == "percent":
            shares, struct = solve_percent_season(df_season, week_cols)
        else:
            shares, struct = solve_rank_bottom_season(df_season, week_cols, regime)

        hits, total = elimination_match_rate(shares, struct, regime)
        tau, mean_abs = season_rank_metrics(shares, struct, df_season, regime)
        match_rows.append(
            {
                "season": season,
                "weeks": total,
                "matched": hits,
                "match_rate": hits / total if total else 1.0,
                "kendall_tau": tau,
                "mean_abs_rank_diff": mean_abs,
            }
        )

        for w in range(struct["max_week"]):
            for i, name in enumerate(struct["names"]):
                if shares[w, i] > 0:
                    records.append(
                        {
                            "season": season,
                            "week": w + 1,
                            "celebrity_name": name,
                            "s_map": shares[w, i],
                        }
                    )

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    pd.DataFrame(match_rows).to_csv(MATCH_PATH, index=False)


if __name__ == "__main__":
    main()
    raise SystemExit(0)

# =====================================================================
# ARCHIVED / EXPERIMENTAL VARIANTS BELOW THIS POINT
# ---------------------------------------------------------------------
# Everything below is *not executed* because the script exits above.
# This file includes multiple pasted variants of the model (uniform priors,
# percent-only runs, hard-constraint MILPs, etc.). Keep them here for
# reference in paper-writing, but do not expect them to run unless you
# remove the SystemExit above and clean up duplicate definitions.
# =====================================================================
import re

import cvxpy as cp
import numpy as np
import pandas as pd

try:
    import pulp as pl
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install pulp") from exc


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1/inferred_shares.csv"
MATCH_PATH = "AR-Problem1/elimination_match.csv"

EPS_SHARE = 1e-6
EPS_RANK = 0.0
M_BIG = 50.0

# Convex weights (percent seasons)
LAM_SMOOTH = 1.0
LAM_UNI = 1.0


def parse_week_cols(df):
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df, week, cols):
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def elimination_week(row, last_active_week):
    if isinstance(row["results"], str) and "Eliminated Week" in row["results"]:
        return int(row["results"].split("Eliminated Week ")[1])
    if isinstance(row["results"], str) and "Withdrew" in row["results"]:
        return last_active_week
    return None


def regime_for_season(season):
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom"


def build_season_struct(df_season, week_cols):
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


def solve_percent_season(df_season, week_cols):
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape

    active = []
    for w in range(W):
        if J[w].sum() == 0:
            active.append([])
            continue
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        active.append(idx)

    s = cp.Variable((W, N))
    constraints = [s >= 0]
    obj_terms = []

    for w in range(W):
        idx = active[w]
        if not idx:
            constraints.append(s[w, :] == 0)
            continue
        constraints.append(cp.sum(s[w, idx]) == 1)
        if len(idx) < N:
            constraints.append(s[w, [i for i in range(N) if i not in idx]] == 0)

        uniform = 1.0 / len(idx)
        obj_terms.append(cp.sum_squares(s[w, idx] - uniform))

        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        if not elim:
            continue
        non_elim = [i for i in idx if i not in elim]
        if not non_elim:
            continue

        j_pct = J[w, idx] / J[w, idx].sum()
        j_map = {i: j_pct[k] for k, i in enumerate(idx)}
        for e in elim:
            for j in non_elim:
                constraints.append(
                    j_map[e] + s[w, e] <= j_map[j] + s[w, j] - EPS_SHARE
                )

    for w in range(1, W):
        prev = set(active[w - 1])
        curr = set(active[w])
        shared = list(prev & curr)
        if shared:
            obj_terms.append(cp.sum_squares(s[w, shared] - s[w - 1, shared]))

    prob = cp.Problem(cp.Minimize(LAM_UNI * cp.sum(obj_terms)), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Season {df_season['season'].iloc[0]}: {prob.status}")
    return np.clip(s.value, 0, 1), struct


def solve_rank_week(Jw, elim_idx, idx):
    n = len(idx)
    prob = pl.LpProblem("rank_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

    uniform = 1.0 / n
    for i in idx:
        prob += u[i] >= s[i] - uniform
        prob += u[i] >= -(s[i] - uniform)

    y = {}
    for i in idx:
        for j in idx:
            if i == j:
                continue
            y[i, j] = pl.LpVariable(f"y_{i}_{j}", cat="Binary")
            prob += s[i] - s[j] >= EPS_RANK - (1 - y[i, j])
            prob += s[i] - s[j] <= -EPS_RANK + y[i, j]

    rJ = {i: int(pd.Series(Jw).rank(ascending=False, method="first").to_numpy()[i]) for i in idx}
    rF = {i: n - pl.lpSum(y[i, j] for j in idx if j != i) for i in idx}
    R = {i: rJ[i] + rF[i] for i in idx}

    non_elim = [i for i in idx if i not in elim_idx]
    for e in elim_idx:
        for j in non_elim:
            prob += R[e] >= R[j]

    prob += pl.lpSum(u.values())
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    return {i: pl.value(s[i]) for i in idx}


def solve_bottom_week(Jw, elim_idx, idx):
    n = len(idx)
    prob = pl.LpProblem("bottom_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

    uniform = 1.0 / n
    for i in idx:
        prob += u[i] >= s[i] - uniform
        prob += u[i] >= -(s[i] - uniform)

    y = {}
    for i in idx:
        for j in idx:
            if i == j:
                continue
            y[i, j] = pl.LpVariable(f"y_{i}_{j}", cat="Binary")
            prob += s[i] - s[j] >= EPS_RANK - (1 - y[i, j])
            prob += s[i] - s[j] <= -EPS_RANK + y[i, j]

    rJ = {i: int(pd.Series(Jw).rank(ascending=False, method="first").to_numpy()[i]) for i in idx}
    rF = {i: n - pl.lpSum(y[i, j] for j in idx if j != i) for i in idx}
    R = {i: rJ[i] + rF[i] for i in idx}

    k_elim = len(elim_idx)
    bottom_k = 2 if k_elim == 1 else k_elim
    z = {i: pl.LpVariable(f"z_{i}", cat="Binary") for i in idx}
    t = pl.LpVariable("t", lowBound=0)
    prob += pl.lpSum(z[i] for i in idx) == bottom_k
    for i in idx:
        prob += R[i] <= t + M_BIG * z[i]
        prob += R[i] >= t - M_BIG * (1 - z[i])
    for e in elim_idx:
        prob += z[e] == 1

    prob += pl.lpSum(u.values())
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    return {i: pl.value(s[i]) for i in idx}


def solve_rank_bottom_season(df_season, week_cols, regime):
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    shares = np.zeros((W, N), dtype=float)

    for w in range(W):
        if J[w].sum() == 0:
            continue
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        if not elim:
            continue

        sol = solve_rank_week(J[w], elim, idx) if regime == "rank" else solve_bottom_week(J[w], elim, idx)
        if sol is None:
            for i in idx:
                shares[w, i] = 1.0 / len(idx)
        else:
            for i, val in sol.items():
                shares[w, i] = val
    return shares, struct


def elimination_match_rate(shares, struct):
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    hits = 0
    total = 0

    for w in range(W):
        idx = np.where(J[w] > 0)[0]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        total += 1
        j_pct = J[w, idx] / J[w, idx].sum()
        C = np.full(N, np.inf)
        for k, i in enumerate(idx):
            C[i] = j_pct[k] + shares[w, i]

        if not elim:
            hits += 1
            continue

        k_elim = len(elim)
        pred = set(np.argsort(C)[:k_elim])
        if pred == set(elim):
            hits += 1
    return hits, total


def main():
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    week_cols = parse_week_cols(df)

    records = []
    match_rows = []
    for season in sorted(df["season"].unique()):
        df_season = df[df["season"] == season].reset_index(drop=True)
        regime = regime_for_season(season)
        print(f"Solving season {season} ({regime})...")

        if regime == "percent":
            shares, struct = solve_percent_season(df_season, week_cols)
        else:
            shares, struct = solve_rank_bottom_season(df_season, week_cols, regime)

        hits, total = elimination_match_rate(shares, struct)
        match_rows.append(
            {
                "season": season,
                "weeks": total,
                "matched": hits,
                "match_rate": hits / total if total else 1.0,
            }
        )

        for w in range(struct["max_week"]):
            for i, name in enumerate(struct["names"]):
                if shares[w, i] > 0:
                    records.append(
                        {
                            "season": season,
                            "week": w + 1,
                            "celebrity_name": name,
                            "s_map": shares[w, i],
                        }
                    )

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    pd.DataFrame(match_rows).to_csv(MATCH_PATH, index=False)


if __name__ == "__main__":
    main()
import re

import cvxpy as cp
import numpy as np
import pandas as pd

try:
    import pulp as pl
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install pulp") from exc


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1/inferred_shares.csv"
MATCH_PATH = "AR-Problem1/elimination_match.csv"

EPS_SHARE = 1e-6
EPS_RANK = 0.0
M_BIG = 50.0

# Convex weights (percent seasons)
LAM_SMOOTH = 1.0
LAM_UNI = 1.0


def parse_week_cols(df):
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df, week, cols):
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def elimination_week(row, last_active_week):
    if isinstance(row["results"], str) and "Eliminated Week" in row["results"]:
        return int(row["results"].split("Eliminated Week ")[1])
    if isinstance(row["results"], str) and "Withdrew" in row["results"]:
        return last_active_week
    return None


def regime_for_season(season):
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom"


def build_season_struct(df_season, week_cols):
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


def solve_percent_season(df_season, week_cols):
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape

    active = []
    for w in range(W):
        if J[w].sum() == 0:
            active.append([])
            continue
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        active.append(idx)

    s = cp.Variable((W, N))
    constraints = [s >= 0]
    obj_terms = []

    for w in range(W):
        idx = active[w]
        if not idx:
            constraints.append(s[w, :] == 0)
            continue
        constraints.append(cp.sum(s[w, idx]) == 1)
        if len(idx) < N:
            constraints.append(s[w, [i for i in range(N) if i not in idx]] == 0)

        uniform = 1.0 / len(idx)
        obj_terms.append(cp.sum_squares(s[w, idx] - uniform))

        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        if not elim:
            continue
        non_elim = [i for i in idx if i not in elim]
        if not non_elim:
            continue

        j_pct = J[w, idx] / J[w, idx].sum()
        j_map = {i: j_pct[k] for k, i in enumerate(idx)}
        for e in elim:
            for j in non_elim:
                constraints.append(
                    j_map[e] + s[w, e] <= j_map[j] + s[w, j] - EPS_SHARE
                )

    for w in range(1, W):
        prev = set(active[w - 1])
        curr = set(active[w])
        shared = list(prev & curr)
        if shared:
            obj_terms.append(cp.sum_squares(s[w, shared] - s[w - 1, shared]))

    prob = cp.Problem(cp.Minimize(LAM_UNI * cp.sum(obj_terms)), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Season {df_season['season'].iloc[0]}: {prob.status}")
    return np.clip(s.value, 0, 1), struct


def solve_rank_week(Jw, elim_idx, idx):
    n = len(idx)
    prob = pl.LpProblem("rank_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

    uniform = 1.0 / n
    for i in idx:
        prob += u[i] >= s[i] - uniform
        prob += u[i] >= -(s[i] - uniform)

    y = {}
    for i in idx:
        for j in idx:
            if i == j:
                continue
            y[i, j] = pl.LpVariable(f"y_{i}_{j}", cat="Binary")
            prob += s[i] - s[j] >= EPS_RANK - (1 - y[i, j])
            prob += s[i] - s[j] <= -EPS_RANK + y[i, j]

    rJ = {i: int(pd.Series(Jw).rank(ascending=False, method="first").to_numpy()[i]) for i in idx}
    rF = {i: n - pl.lpSum(y[i, j] for j in idx if j != i) for i in idx}
    R = {i: rJ[i] + rF[i] for i in idx}

    non_elim = [i for i in idx if i not in elim_idx]
    for e in elim_idx:
        for j in non_elim:
            prob += R[e] >= R[j]

    prob += pl.lpSum(u.values())
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    return {i: pl.value(s[i]) for i in idx}


def solve_bottom_week(Jw, elim_idx, idx):
    n = len(idx)
    prob = pl.LpProblem("bottom_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

    uniform = 1.0 / n
    for i in idx:
        prob += u[i] >= s[i] - uniform
        prob += u[i] >= -(s[i] - uniform)

    y = {}
    for i in idx:
        for j in idx:
            if i == j:
                continue
            y[i, j] = pl.LpVariable(f"y_{i}_{j}", cat="Binary")
            prob += s[i] - s[j] >= EPS_RANK - (1 - y[i, j])
            prob += s[i] - s[j] <= -EPS_RANK + y[i, j]

    rJ = {i: int(pd.Series(Jw).rank(ascending=False, method="first").to_numpy()[i]) for i in idx}
    rF = {i: n - pl.lpSum(y[i, j] for j in idx if j != i) for i in idx}
    R = {i: rJ[i] + rF[i] for i in idx}

    k_elim = len(elim_idx)
    bottom_k = 2 if k_elim == 1 else k_elim
    z = {i: pl.LpVariable(f"z_{i}", cat="Binary") for i in idx}
    t = pl.LpVariable("t", lowBound=0)
    prob += pl.lpSum(z[i] for i in idx) == bottom_k
    for i in idx:
        prob += R[i] <= t + M_BIG * z[i]
        prob += R[i] >= t - M_BIG * (1 - z[i])
    for e in elim_idx:
        prob += z[e] == 1

    prob += pl.lpSum(u.values())
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    return {i: pl.value(s[i]) for i in idx}


def solve_rank_bottom_season(df_season, week_cols, regime):
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    shares = np.zeros((W, N), dtype=float)

    for w in range(W):
        if J[w].sum() == 0:
            continue
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        if not elim:
            continue

        sol = solve_rank_week(J[w], elim, idx) if regime == "rank" else solve_bottom_week(J[w], elim, idx)
        if sol is None:
            for i in idx:
                shares[w, i] = 1.0 / len(idx)
        else:
            for i, val in sol.items():
                shares[w, i] = val
    return shares, struct


def elimination_match_rate(shares, struct):
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    hits = 0
    total = 0

    for w in range(W):
        idx = np.where(J[w] > 0)[0]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        total += 1
        j_pct = J[w, idx] / J[w, idx].sum()
        C = np.full(N, np.inf)
        for k, i in enumerate(idx):
            C[i] = j_pct[k] + shares[w, i]

        if not elim:
            hits += 1
            continue

        k_elim = len(elim)
        pred = set(np.argsort(C)[:k_elim])
        if pred == set(elim):
            hits += 1
    return hits, total


def main():
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    week_cols = parse_week_cols(df)

    records = []
    match_rows = []
    for season in sorted(df["season"].unique()):
        df_season = df[df["season"] == season].reset_index(drop=True)
        regime = regime_for_season(season)
        print(f"Solving season {season} ({regime})...")

        if regime == "percent":
            shares, struct = solve_percent_season(df_season, week_cols)
        else:
            shares, struct = solve_rank_bottom_season(df_season, week_cols, regime)

        hits, total = elimination_match_rate(shares, struct)
        match_rows.append(
            {
                "season": season,
                "weeks": total,
                "matched": hits,
                "match_rate": hits / total if total else 1.0,
            }
        )

        for w in range(struct["max_week"]):
            for i, name in enumerate(struct["names"]):
                if shares[w, i] > 0:
                    records.append(
                        {
                            "season": season,
                            "week": w + 1,
                            "celebrity_name": name,
                            "s_map": shares[w, i],
                        }
                    )

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    pd.DataFrame(match_rows).to_csv(MATCH_PATH, index=False)


if __name__ == "__main__":
    main()
import re

import cvxpy as cp
import numpy as np
import pandas as pd

try:
    import pulp as pl
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install pulp") from exc


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1/inferred_shares.csv"
MATCH_PATH = "AR-Problem1/elimination_match.csv"

EPS_SHARE = 1e-6
EPS_RANK = 1e-6
M_BIG = 100.0

# Convex weights (percent seasons)
LAM_SMOOTH = 1.0
LAM_UNI = 1.0


def parse_week_cols(df):
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df, week, cols):
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def elimination_week(row, last_active_week):
    if isinstance(row["results"], str) and "Eliminated Week" in row["results"]:
        return int(row["results"].split("Eliminated Week ")[1])
    if isinstance(row["results"], str) and "Withdrew" in row["results"]:
        return last_active_week
    return None


def regime_for_season(season):
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom"


def build_season_struct(df_season, week_cols):
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


def solve_percent_season(df_season, week_cols):
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape

    active = []
    for w in range(W):
        if J[w].sum() == 0:
            active.append([])
            continue
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        active.append(idx)

    s = cp.Variable((W, N))
    constraints = [s >= 0]
    obj_terms = []

    for w in range(W):
        idx = active[w]
        if not idx:
            constraints.append(s[w, :] == 0)
            continue
        constraints.append(cp.sum(s[w, idx]) == 1)
        if len(idx) < N:
            constraints.append(s[w, [i for i in range(N) if i not in idx]] == 0)

        uniform = 1.0 / len(idx)
        obj_terms.append(cp.sum_squares(s[w, idx] - uniform))

        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        if not elim:
            continue
        non_elim = [i for i in idx if i not in elim]
        if not non_elim:
            continue

        j_pct = J[w, idx] / J[w, idx].sum()
        j_map = {i: j_pct[k] for k, i in enumerate(idx)}
        for e in elim:
            for j in non_elim:
                constraints.append(
                    j_map[e] + s[w, e] <= j_map[j] + s[w, j] - EPS_SHARE
                )

    for w in range(1, W):
        prev = set(active[w - 1])
        curr = set(active[w])
        shared = list(prev & curr)
        if shared:
            obj_terms.append(cp.sum_squares(s[w, shared] - s[w - 1, shared]))

    prob = cp.Problem(cp.Minimize(LAM_UNI * cp.sum(obj_terms)), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Season {df_season['season'].iloc[0]}: {prob.status}")
    return np.clip(s.value, 0, 1), struct


def solve_rank_week(Jw, elim_idx, idx):
    n = len(idx)
    prob = pl.LpProblem("rank_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

    uniform = 1.0 / n
    for i in idx:
        prob += u[i] >= s[i] - uniform
        prob += u[i] >= -(s[i] - uniform)

    # Fan rank binaries
    y = {}
    for i in idx:
        for j in idx:
            if i == j:
                continue
            y[i, j] = pl.LpVariable(f"y_{i}_{j}", cat="Binary")
            prob += s[i] - s[j] >= EPS_RANK - (1 - y[i, j])
            prob += s[i] - s[j] <= -EPS_RANK + y[i, j]

    rJ = {i: int(pd.Series(Jw).rank(ascending=False, method="first").to_numpy()[i]) for i in idx}
    rF = {i: n - pl.lpSum(y[i, j] for j in idx if j != i) for i in idx}
    R = {i: rJ[i] + rF[i] for i in idx}

    non_elim = [i for i in idx if i not in elim_idx]
    for e in elim_idx:
        for j in non_elim:
            prob += R[e] >= R[j] + EPS_RANK

    prob += pl.lpSum(u.values())
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    return {i: pl.value(s[i]) for i in idx}


def solve_bottom_week(Jw, elim_idx, idx):
    n = len(idx)
    prob = pl.LpProblem("bottom_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

    uniform = 1.0 / n
    for i in idx:
        prob += u[i] >= s[i] - uniform
        prob += u[i] >= -(s[i] - uniform)

    # Fan rank binaries
    y = {}
    for i in idx:
        for j in idx:
            if i == j:
                continue
            y[i, j] = pl.LpVariable(f"y_{i}_{j}", cat="Binary")
            prob += s[i] - s[j] >= EPS_RANK - (1 - y[i, j])
            prob += s[i] - s[j] <= -EPS_RANK + y[i, j]

    rJ = {i: int(pd.Series(Jw).rank(ascending=False, method="first").to_numpy()[i]) for i in idx}
    rF = {i: n - pl.lpSum(y[i, j] for j in idx if j != i) for i in idx}
    R = {i: rJ[i] + rF[i] for i in idx}

    k_elim = len(elim_idx)
    bottom_k = 2 if k_elim == 1 else k_elim
    z = {i: pl.LpVariable(f"z_{i}", cat="Binary") for i in idx}
    t = pl.LpVariable("t", lowBound=0)
    prob += pl.lpSum(z[i] for i in idx) == bottom_k
    for i in idx:
        prob += R[i] <= t + M_BIG * z[i]
        prob += R[i] >= t + EPS_RANK - M_BIG * (1 - z[i])
    for e in elim_idx:
        prob += z[e] == 1

    prob += pl.lpSum(u.values())
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    return {i: pl.value(s[i]) for i in idx}


def solve_rank_bottom_season(df_season, week_cols, regime):
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    shares = np.zeros((W, N), dtype=float)

    for w in range(W):
        if J[w].sum() == 0:
            continue
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        if not elim:
            continue

        if regime == "rank":
            sol = solve_rank_week(J[w], elim, idx)
        else:
            sol = solve_bottom_week(J[w], elim, idx)
        if sol is None:
            # fallback: uniform among active
            for i in idx:
                shares[w, i] = 1.0 / len(idx)
        else:
            for i, val in sol.items():
                shares[w, i] = val
    return shares, struct


def elimination_match_rate(shares, struct):
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    hits = 0
    total = 0

    for w in range(W):
        idx = np.where(J[w] > 0)[0]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        total += 1
        j_pct = J[w, idx] / J[w, idx].sum()
        C = np.full(N, np.inf)
        for k, i in enumerate(idx):
            C[i] = j_pct[k] + shares[w, i]

        if not elim:
            hits += 1
            continue

        k_elim = len(elim)
        pred = set(np.argsort(C)[:k_elim])
        if pred == set(elim):
            hits += 1
    return hits, total


def main():
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    week_cols = parse_week_cols(df)

    records = []
    match_rows = []
    for season in sorted(df["season"].unique()):
        df_season = df[df["season"] == season].reset_index(drop=True)
        regime = regime_for_season(season)
        print(f"Solving season {season} ({regime})...")

        if regime == "percent":
            shares, struct = solve_percent_season(df_season, week_cols)
        else:
            shares, struct = solve_rank_bottom_season(df_season, week_cols, regime)

        hits, total = elimination_match_rate(shares, struct)
        match_rows.append(
            {
                "season": season,
                "weeks": total,
                "matched": hits,
                "match_rate": hits / total if total else 1.0,
            }
        )

        for w in range(struct["max_week"]):
            for i, name in enumerate(struct["names"]):
                if shares[w, i] > 0:
                    records.append(
                        {
                            "season": season,
                            "week": w + 1,
                            "celebrity_name": name,
                            "s_map": shares[w, i],
                        }
                    )

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    pd.DataFrame(match_rows).to_csv(MATCH_PATH, index=False)


if __name__ == "__main__":
    main()
import re

import cvxpy as cp
import numpy as np
import pandas as pd


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1/inferred_shares.csv"
MATCH_PATH = "AR-Problem1/elimination_match.csv"

# Convex percent-only settings (S3–S27)
START_SEASON = 3
END_SEASON = 27
EPS_SHARE = 1e-6
LAM_SMOOTH = 1.0  # L2 smoothness
LAM_UNI = 1.0  # L2 to uniform


def parse_week_cols(df):
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df, week, cols):
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def elimination_week(row, last_active_week):
    if isinstance(row["results"], str) and "Eliminated Week" in row["results"]:
        return int(row["results"].split("Eliminated Week ")[1])
    if isinstance(row["results"], str) and "Withdrew" in row["results"]:
        return last_active_week
    return None


def build_season_struct(df_season, week_cols):
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


def solve_season_percent(df_season, week_cols):
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape

    active = []
    for w in range(W):
        if J[w].sum() == 0:
            active.append([])
            continue
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        active.append(idx)

    s = cp.Variable((W, N))
    constraints = [s >= 0]
    obj_terms = []

    for w in range(W):
        idx = active[w]
        if not idx:
            constraints.append(s[w, :] == 0)
            continue
        constraints.append(cp.sum(s[w, idx]) == 1)
        if len(idx) < N:
            constraints.append(s[w, [i for i in range(N) if i not in idx]] == 0)

        uniform = 1.0 / len(idx)
        obj_terms.append(cp.sum_squares(s[w, idx] - uniform))

        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        if not elim:
            continue
        non_elim = [i for i in idx if i not in elim]
        if not non_elim:
            continue

        j_pct = J[w, idx] / J[w, idx].sum()
        j_map = {i: j_pct[k] for k, i in enumerate(idx)}
        for e in elim:
            for j in non_elim:
                constraints.append(
                    j_map[e] + s[w, e] <= j_map[j] + s[w, j] - EPS_SHARE
                )

    for w in range(1, W):
        prev = set(active[w - 1])
        curr = set(active[w])
        shared = list(prev & curr)
        if shared:
            obj_terms.append(cp.sum_squares(s[w, shared] - s[w - 1, shared]))

    objective = cp.Minimize(LAM_UNI * cp.sum(obj_terms))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Season {df_season['season'].iloc[0]}: {prob.status}")

    shares = np.clip(s.value, 0, 1)
    return shares, struct


def elimination_match_rate_percent(shares, struct):
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    hits = 0
    total = 0

    for w in range(W):
        idx = np.where(J[w] > 0)[0]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        total += 1
        j_pct = J[w, idx] / J[w, idx].sum()
        C = np.full(N, np.inf)
        for k, i in enumerate(idx):
            C[i] = j_pct[k] + shares[w, i]

        if not elim:
            hits += 1
            continue

        k_elim = len(elim)
        pred = set(np.argsort(C)[:k_elim])
        if pred == set(elim):
            hits += 1
    return hits, total


def main():
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    week_cols = parse_week_cols(df)

    records = []
    match_rows = []
    for season in sorted(df["season"].unique()):
        if season < START_SEASON or season > END_SEASON:
            continue
        print(f"Solving percent season {season}...")
        df_season = df[df["season"] == season].reset_index(drop=True)
        shares, struct = solve_season_percent(df_season, week_cols)
        hits, total = elimination_match_rate_percent(shares, struct)
        print(f"Season {season}: elimination match {hits}/{total}")
        match_rows.append(
            {
                "season": season,
                "weeks": total,
                "matched": hits,
                "match_rate": hits / total if total else 1.0,
            }
        )

        for w in range(struct["max_week"]):
            for i, name in enumerate(struct["names"]):
                if shares[w, i] > 0:
                    records.append(
                        {
                            "season": season,
                            "week": w + 1,
                            "celebrity_name": name,
                            "s_map": shares[w, i],
                        }
                    )

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    pd.DataFrame(match_rows).to_csv(MATCH_PATH, index=False)


if __name__ == "__main__":
    main()
import re

import cvxpy as cp
import numpy as np
import pandas as pd


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1/inferred_shares.csv"

# Convex percent-only settings (S3–S27)
START_SEASON = 3
END_SEASON = 27
EPS_SHARE = 1e-6
LAM_SMOOTH = 1.0  # L2 smoothness
LAM_UNI = 1.0  # L2 to uniform


def parse_week_cols(df):
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df, week, cols):
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def elimination_week(row, last_active_week):
    if isinstance(row["results"], str) and "Eliminated Week" in row["results"]:
        return int(row["results"].split("Eliminated Week ")[1])
    if isinstance(row["results"], str) and "Withdrew" in row["results"]:
        return last_active_week
    return None


def build_season_struct(df_season, week_cols):
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


def solve_season_percent(df_season, week_cols):
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape

    active = []
    for w in range(W):
        if J[w].sum() == 0:
            active.append([])
            continue
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        active.append(idx)

    s = cp.Variable((W, N))
    constraints = [s >= 0]
    obj_terms = []

    for w in range(W):
        idx = active[w]
        if not idx:
            constraints.append(s[w, :] == 0)
            continue
        constraints.append(cp.sum(s[w, idx]) == 1)
        if len(idx) < N:
            constraints.append(s[w, [i for i in range(N) if i not in idx]] == 0)

        uniform = 1.0 / len(idx)
        obj_terms.append(cp.sum_squares(s[w, idx] - uniform))

        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        if not elim:
            continue
        non_elim = [i for i in idx if i not in elim]
        if not non_elim:
            continue

        j_pct = J[w, idx] / J[w, idx].sum()
        j_map = {i: j_pct[k] for k, i in enumerate(idx)}
        for e in elim:
            for j in non_elim:
                constraints.append(
                    j_map[e] + s[w, e] <= j_map[j] + s[w, j] - EPS_SHARE
                )

    for w in range(1, W):
        prev = set(active[w - 1])
        curr = set(active[w])
        shared = list(prev & curr)
        if shared:
            obj_terms.append(cp.sum_squares(s[w, shared] - s[w - 1, shared]))

    objective = cp.Minimize(LAM_UNI * cp.sum(obj_terms))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, verbose=False)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Season {df_season['season'].iloc[0]}: {prob.status}")

    shares = np.clip(s.value, 0, 1)
    return shares, struct


def elimination_match_rate_percent(shares, struct):
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    hits = 0
    total = 0

    for w in range(W):
        idx = np.where(J[w] > 0)[0]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        total += 1
        j_pct = J[w, idx] / J[w, idx].sum()
        C = np.full(N, np.inf)
        for k, i in enumerate(idx):
            C[i] = j_pct[k] + shares[w, i]

        if not elim:
            # No-elimination week: considered matched if we don't force eliminations.
            hits += 1
            continue

        k_elim = len(elim)
        pred = set(np.argsort(C)[:k_elim])
        if pred == set(elim):
            hits += 1
    return hits, total


def main():
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    week_cols = parse_week_cols(df)

    records = []
    for season in sorted(df["season"].unique()):
        if season < START_SEASON or season > END_SEASON:
            continue
        print(f"Solving percent season {season}...")
        df_season = df[df["season"] == season].reset_index(drop=True)
        shares, struct = solve_season_percent(df_season, week_cols)
        hits, total = elimination_match_rate_percent(shares, struct)
        print(f"Season {season}: elimination match {hits}/{total}")

        for w in range(struct["max_week"]):
            for i, name in enumerate(struct["names"]):
                if shares[w, i] > 0:
                    records.append(
                        {
                            "season": season,
                            "week": w + 1,
                            "celebrity_name": name,
                            "s_map": shares[w, i],
                        }
                    )

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
import re

import numpy as np
import pandas as pd

try:
    import pulp as pl
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install pulp") from exc


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1/inferred_shares.csv"
START_SEASON = 28
FAST_MODE = True  # speed: no smoothness, percent-combined for S28+

# Hard-constraint MILP settings
EPS_SHARE = 1e-6
EPS_RANK = 1e-6
EPS_JUDGE = 0.0
M_SHARE = 1.0
M_BIG = 100.0
LAM_SMOOTH = 10.0
LAM_UNI = 1.0


def parse_week_cols(df):
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df, week, cols):
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def elimination_week(row, last_active_week):
    if isinstance(row["results"], str) and "Eliminated Week" in row["results"]:
        return int(row["results"].split("Eliminated Week ")[1])
    if isinstance(row["results"], str) and "Withdrew" in row["results"]:
        return last_active_week
    return None


def regime_for_season(season):
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom"


def build_season_struct(df_season, week_cols):
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


def judge_rank(Jw):
    jitter = np.arange(len(Jw)) * 1e-6
    return (
        pd.Series(Jw + jitter).rank(ascending=False, method="first").to_numpy().astype(int)
    )


def solve_season_hard(season, df_season, week_cols):
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape

    active = []
    for w in range(W):
        if J[w].sum() == 0:
            active.append([])
            continue
        idx = [
            int(i)
            for i, ew in enumerate(elim_week)
            if ew is None or ew >= (w + 1)
        ]
        active.append(idx)
    j_pct = []
    for w in range(W):
        idx = active[w]
        if not idx:
            j_pct.append({})
            continue
        denom = J[w, idx].sum()
        j_pct.append({i: J[w, i] / denom for i in idx})

    prob = pl.LpProblem(f"invopt_s{season}", pl.LpMinimize)
    s = {}
    u = {}
    d = {}

    for w in range(W):
        idx = active[w]
        if not idx:
            continue
        n = len(idx)
        for i in idx:
            s[w, i] = pl.LpVariable(f"s_{w}_{i}", lowBound=0, upBound=1)
        prob += pl.lpSum(s[w, i] for i in idx) == 1

        # L1 to uniform (max-entropy proxy)
        for i in idx:
            u[w, i] = pl.LpVariable(f"u_{w}_{i}", lowBound=0)
            prob += u[w, i] >= s[w, i] - 1 / n
            prob += u[w, i] >= -(s[w, i] - 1 / n)

    # Smoothness across weeks (L1)
    if not FAST_MODE and LAM_SMOOTH > 0:
        for w in range(1, W):
            prev = set(active[w - 1])
            curr = set(active[w])
            for i in prev & curr:
                d[w, i] = pl.LpVariable(f"d_{w}_{i}", lowBound=0)
                prob += d[w, i] >= s[w, i] - s[w - 1, i]
                prob += d[w, i] >= -(s[w, i] - s[w - 1, i])

    # Elimination constraints
    for w in range(W):
        idx = active[w]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        if not elim:
            continue
        non_elim = [i for i in idx if i not in elim]
        k_elim = len(elim)

        if regime_for_season(season) == "percent":
            # Bottom-k: every eliminated is no higher than any survivor.
            for e in elim:
                for j in non_elim:
                    prob += (
                        j_pct[w][e] + s[w, e]
                        <= j_pct[w][j] + s[w, j] - EPS_SHARE
                    )

        else:
            if FAST_MODE and regime_for_season(season) == "bottom":
                # Fast approximation: use percent-combined score for bottom-k.
                C = {i: j_pct[w][i] + s[w, i] for i in idx}
                bottom_k = 2 if k_elim == 1 else k_elim
                z = {i: pl.LpVariable(f"z_{w}_{i}", cat="Binary") for i in idx}
                t = pl.LpVariable(f"t_{w}", lowBound=0)
                prob += pl.lpSum(z[i] for i in idx) == bottom_k
                for i in idx:
                    prob += C[i] <= t + M_BIG * z[i]
                    prob += C[i] >= t + EPS_RANK - M_BIG * (1 - z[i])
                for e in elim:
                    prob += z[e] == 1
                continue

            # Fan rank binaries
            y = {}
            for i in idx:
                for j in idx:
                    if i == j:
                        continue
                    y[i, j] = pl.LpVariable(f"y_{w}_{i}_{j}", cat="Binary")
                    prob += s[w, i] - s[w, j] >= EPS_RANK - M_SHARE * (1 - y[i, j])
                    prob += s[w, i] - s[w, j] <= -EPS_RANK + M_SHARE * y[i, j]

            rJ = {i: int(judge_rank(J[w])[i]) for i in idx}
            rF = {i: len(idx) - pl.lpSum(y[i, j] for j in idx if j != i) for i in idx}
            R = {i: rJ[i] + rF[i] for i in idx}

            if regime_for_season(season) == "rank":
                # Bottom-k by combined rank.
                for e in elim:
                    for j in non_elim:
                        prob += R[e] >= R[j] + EPS_RANK

            else:
                # Bottom-two (or bottom-k) by rank; only enforce membership.
                bottom_k = 2 if k_elim == 1 else k_elim
                z = {i: pl.LpVariable(f"z_{w}_{i}", cat="Binary") for i in idx}
                t = pl.LpVariable(f"t_{w}", lowBound=0)
                prob += pl.lpSum(z[i] for i in idx) == bottom_k
                for i in idx:
                    prob += R[i] <= t + M_BIG * z[i]
                    prob += R[i] >= t + EPS_RANK - M_BIG * (1 - z[i])
                for e in elim:
                    prob += z[e] == 1

    obj = LAM_UNI * pl.lpSum(u.values()) + LAM_SMOOTH * pl.lpSum(d.values())
    if isinstance(obj, (int, float)):
        obj = pl.LpAffineExpression()
    prob += obj

    prob.solve(pl.PULP_CBC_CMD(msg=True))
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Season {season}: solver status {pl.LpStatus[prob.status]}")

    shares = np.zeros((W, N), dtype=float)
    for w in range(W):
        for i in active[w]:
            shares[w, i] = pl.value(s[w, i])
    return shares, struct


def elimination_match_rate(season, shares, struct):
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    hits = 0
    total = 0

    for w in range(W):
        mask = J[w] > 0
        if mask.sum() < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1)]
        if not elim:
            continue
        total += 1

        if regime_for_season(season) == "percent":
            C = J[w] / (J[w][mask].sum() + 1e-12) + shares[w]
            pred = np.argmin(np.where(mask, C, np.inf))
        elif regime_for_season(season) == "rank":
            rF = np.argsort(-shares[w][mask]).argsort() + 1
            rJ = pd.Series(J[w][mask]).rank(ascending=False, method="average").to_numpy()
            R = rJ + rF
            pred_idx = np.argmax(R)
            pred = np.where(mask)[0][pred_idx]
        else:
            rF = np.argsort(-shares[w][mask]).argsort() + 1
            rJ = pd.Series(J[w][mask]).rank(ascending=False, method="average").to_numpy()
            C = rJ + rF
            worst_two = np.argsort(-C)[:2]
            worst_two_idx = np.where(mask)[0][worst_two]
            j_scores = J[w][worst_two_idx]
            pred = worst_two_idx[np.argmin(j_scores)]

        if pred in elim:
            hits += 1

    return hits, total


def main():
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    week_cols = parse_week_cols(df)

    records = []
    for season in sorted(df["season"].unique()):
        if season < START_SEASON:
            continue
        print(f"Solving season {season}...")
        df_season = df[df["season"] == season].reset_index(drop=True)
        shares, struct = solve_season_hard(season, df_season, week_cols)
        hits, total = elimination_match_rate(season, shares, struct)
        print(f"Season {season}: elimination match {hits}/{total}")

        for w in range(struct["max_week"]):
            for i, name in enumerate(struct["names"]):
                if shares[w, i] > 0:
                    records.append(
                        {
                            "season": season,
                            "week": w + 1,
                            "celebrity_name": name,
                            "s_map": shares[w, i],
                        }
                    )

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)


if __name__ == "__main__":
    main()
