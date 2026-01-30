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
LAM_POP = 1.0
FAST_BOTTOM = True
ALPHA_PREV = 0.85
GAMMA_POP = 0.7


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


def popularity_prior(df_season, idx):
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

    # Actual elimination week = last week with positive judge scores.
    actual_week = np.where(pos_mask.any(axis=0), pos_mask.sum(axis=0), np.nan)

    valid = [
        i for i in idx if not np.isnan(avg_judge[i]) and not np.isnan(actual_week[i])
    ]
    if not valid:
        return {i: 1.0 / len(idx) for i in idx}

    # Judge-only expected elimination week via per-season linear fit.
    x = avg_judge[valid].astype(float)
    y = actual_week[valid].astype(float)
    if np.unique(x).size < 2:
        pred = np.full_like(y, y.mean(), dtype=float)
    else:
        slope, intercept = np.polyfit(x, y, 1)
        pred = slope * x + intercept

    # Fan support proxy = survival beyond judge-only expectation.
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

    last_shares = None
    for w in range(W):
        idx = active[w]
        if not idx:
            constraints.append(s[w, :] == 0)
            continue
        constraints.append(cp.sum(s[w, idx]) == 1)
        if len(idx) < N:
            constraints.append(s[w, [i for i in range(N) if i not in idx]] == 0)

        target = blended_target(df_season, idx, last_shares, use_prev=False)
        target_vec = np.array([target[i] for i in idx])
        obj_terms.append(LAM_POP * cp.sum_squares(s[w, idx] - target_vec))

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
            obj_terms.append(LAM_SMOOTH * cp.sum_squares(s[w, shared] - s[w - 1, shared]))

    prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Season {df_season['season'].iloc[0]}: {prob.status}")
    s_val = np.clip(s.value, 0, 1)
    return s_val, struct


def solve_rank_week(Jw, elim_idx, idx, target=None):
    n = len(idx)
    prob = pl.LpProblem("rank_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

    if target is None:
        target = {i: 1.0 / n for i in idx}
    for i in idx:
        prob += u[i] >= s[i] - target[i]
        prob += u[i] >= -(s[i] - target[i])

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

    non_elim = [i for i in idx if i not in elim_idx]
    for e in elim_idx:
        for j in non_elim:
            prob += R[e] >= R[j]

    prob += pl.lpSum(u.values())
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        return None
    return {i: pl.value(s[i]) for i in idx}


def solve_bottom_week(Jw, elim_idx, idx, target=None):
    n = len(idx)
    prob = pl.LpProblem("bottom_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

    if target is None:
        target = {i: 1.0 / n for i in idx}
    for i in idx:
        prob += u[i] >= s[i] - target[i]
        prob += u[i] >= -(s[i] - target[i])

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


def solve_bottom_week_fast(Jw, elim_idx, idx, target=None):
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
    struct = build_season_struct(df_season, week_cols)
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    shares = np.zeros((W, N), dtype=float)
    last_shares = None

    for w in range(W):
        if J[w].sum() == 0:
            continue
        idx = [i for i, ew in enumerate(elim_week) if ew is None or ew >= (w + 1)]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        target = blended_target(df_season, idx, last_shares, use_prev=True)
        if not elim:
            for i in idx:
                shares[w, i] = target[i]
            continue

        if regime == "rank":
            sol = solve_rank_week(J[w], elim, idx, target=target)
        else:
            sol = (
                solve_bottom_week_fast(J[w], elim, idx, target=target)
                if FAST_BOTTOM
                else None
            )
            if sol is None:
                sol = solve_bottom_week(J[w], elim, idx, target=target)
        if sol is None:
            for i in idx:
                shares[w, i] = target[i]
        else:
            for i, val in sol.items():
                shares[w, i] = val
            last_shares = {i: shares[w, i] for i in idx}
    return shares, struct


def elimination_match_rate(shares, struct, regime):
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
