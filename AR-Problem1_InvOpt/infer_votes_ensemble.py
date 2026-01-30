import numpy as np
import pandas as pd
import cvxpy as cp

try:
    import pulp as pl
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install pulp") from exc

from model_utils import (
    build_season_struct,
    blended_target,
    parse_week_cols,
    regime_for_season,
)


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1/inferred_shares_ensemble.csv"
MATCH_PATH = "AR-Problem1/elimination_match_ensemble.csv"

EPS_SHARE = 1e-6
EPS_RANK = 0.0
M_BIG = 50.0

LAM_SMOOTH = 1.0
ALPHA_PREV = 0.85
GAMMA_POP = 0.7
NOISE_SCALE = 0.35
N_SAMPLES = 30


def solve_percent_week(Jw, elim_idx, idx, target):
    n = len(idx)
    if n < 2:
        return None
    s = cp.Variable(n)
    constraints = [s >= 0, cp.sum(s) == 1]
    if elim_idx:
        j_pct = Jw[idx] / Jw[idx].sum()
        idx_map = {i: k for k, i in enumerate(idx)}
        for e in elim_idx:
            for j in idx:
                if j == e:
                    continue
                constraints.append(
                    j_pct[idx_map[e]] + s[idx_map[e]]
                    <= j_pct[idx_map[j]] + s[idx_map[j]] - EPS_SHARE
                )
    target_vec = np.array([target[i] for i in idx])
    obj = cp.sum_squares(s - target_vec)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.OSQP, verbose=False)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return {i: max(0.0, min(1.0, s.value[k])) for k, i in enumerate(idx)}


def solve_rank_week(Jw, elim_idx, idx, target):
    n = len(idx)
    prob = pl.LpProblem("rank_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

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


def solve_bottom_week(Jw, elim_idx, idx, target):
    n = len(idx)
    prob = pl.LpProblem("bottom_week", pl.LpMinimize)
    s = {i: pl.LpVariable(f"s_{i}", lowBound=0, upBound=1) for i in idx}
    u = {i: pl.LpVariable(f"u_{i}", lowBound=0) for i in idx}
    prob += pl.lpSum(s[i] for i in idx) == 1

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


def solve_season(df_season, week_cols, regime, rng, noise_scale):
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
        target = blended_target(
            df_season, idx, last_shares, ALPHA_PREV, GAMMA_POP, noise_scale, rng
        )

        if not elim:
            for i in idx:
                shares[w, i] = target[i]
            continue

        if regime == "percent":
            sol = solve_percent_week(J[w], elim, idx, target)
        elif regime == "rank":
            sol = solve_rank_week(J[w], elim, idx, target)
        else:
            sol = solve_bottom_week(J[w], elim, idx, target)

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


def main() -> None:
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    week_cols = parse_week_cols(df)

    records = []
    match_rows = []
    rng = np.random.default_rng(7)

    for season in sorted(df["season"].unique()):
        df_season = df[df["season"] == season].reset_index(drop=True)
        regime = regime_for_season(season)
        print(f"Solving season {season} ({regime}) with ensemble...")

        samples = []
        match_rates = []
        for s in range(N_SAMPLES):
            shares, struct = solve_season(df_season, week_cols, regime, rng, NOISE_SCALE)
            samples.append(shares)
            hits, total = elimination_match_rate(shares, struct, regime)
            match_rates.append(hits / total if total else 1.0)

        arr = np.stack(samples, axis=0)  # S x W x N
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        p10 = np.quantile(arr, 0.1, axis=0)
        p90 = np.quantile(arr, 0.9, axis=0)

        match_rows.append(
            {
                "season": season,
                "match_rate_mean": float(np.mean(match_rates)),
                "match_rate_std": float(np.std(match_rates)),
            }
        )

        for w in range(struct["max_week"]):
            for i, name in enumerate(struct["names"]):
                if mean[w, i] > 0:
                    records.append(
                        {
                            "season": season,
                            "week": w + 1,
                            "celebrity_name": name,
                            "s_mean": mean[w, i],
                            "s_std": std[w, i],
                            "s_p10": p10[w, i],
                            "s_p90": p90[w, i],
                        }
                    )

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    pd.DataFrame(match_rows).to_csv(MATCH_PATH, index=False)
    print(f"Wrote {OUT_PATH} and {MATCH_PATH}")


if __name__ == "__main__":
    main()
