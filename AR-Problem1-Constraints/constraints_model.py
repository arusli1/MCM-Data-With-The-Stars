import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_DIR = "AR-Problem1-Constraints/outputs"
FIG_DIR = "AR-Problem1-Constraints/figures"

EPS = 1e-8


@dataclass
class WeekSamples:
    season: int
    week: int
    regime: str
    names: List[str]
    shares: np.ndarray  # shape (n_samples, n_active)
    margins: np.ndarray  # shape (n_samples,)
    n_elim: int
    n_trials: int


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
        if not cols:
            continue
        vals = df_season[cols].replace("N/A", pd.NA)
        numeric = vals.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        totals = numeric.sum(axis=1)
        if (totals > 0).any():
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
    W = len(elim_week_schedule) - 1
    elim_week = [W + 1] * len(placement_true)
    ranked = [
        (p, i)
        for i, p in enumerate(placement_true)
        if p is not None and p > 0
    ]
    ranked.sort(reverse=True)
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


def compute_judge_pct(J_w: np.ndarray, active: np.ndarray) -> np.ndarray:
    scores = J_w.copy()
    scores[~active] = 0.0
    total = scores.sum()
    if total <= 0:
        return np.zeros_like(scores)
    return scores / total


def margin_for_true_set(
    J_w: np.ndarray,
    s: np.ndarray,
    active: np.ndarray,
    elim_set: np.ndarray,
    regime: str,
) -> float:
    active_idx = np.where(active)[0]
    if active_idx.size == 0:
        return 0.0
    if elim_set.size == 0 or elim_set.size == active_idx.size:
        return 0.0

    if regime == "percent":
        judge_pct = compute_judge_pct(J_w, active)
        combined = judge_pct + s
        max_elim = np.max(combined[elim_set])
        min_surv = np.min(combined[active_idx[~np.isin(active_idx, elim_set)]])
        return float(min_surv - max_elim)

    rJ = rank_order(-J_w, active)
    rF = rank_order(-s, active)
    R = rJ + rF
    min_elim = np.min(R[elim_set])
    max_surv = np.max(R[active_idx[~np.isin(active_idx, elim_set)]])
    return float(min_elim - max_surv)


def sample_feasible_shares(
    season: int,
    week: int,
    names: List[str],
    J_w: np.ndarray,
    elim_week_true: List[int],
    n_accept_target: int,
    n_trials_max: int,
    tol_percent: float,
    tol_rank: float,
) -> Optional[WeekSamples]:
    active = np.array([elim_week_true[i] > week for i in range(len(names))], dtype=bool)
    k = sum(1 for e in elim_week_true if e == week)
    if k <= 0 or active.sum() <= 1:
        return None

    regime = season_regime(season)
    rng = np.random.default_rng(season * 1000 + week)
    active_idx = np.where(active)[0]
    names_active = [names[i] for i in active_idx]
    accepted = []
    margins = []
    n_trials = 0

    elim_set = np.array([i for i, e in enumerate(elim_week_true) if e == week], dtype=int)
    while n_trials < n_trials_max and len(accepted) < n_accept_target:
        n_trials += 1
        s_active = rng.dirichlet(np.ones(len(active_idx)))
        s = np.zeros(len(names), dtype=float)
        s[active_idx] = s_active
        margin = margin_for_true_set(J_w, s, active, elim_set, regime)
        tol = tol_percent if regime == "percent" else tol_rank
        if margin >= -tol:
            accepted.append(s_active)
            margins.append(margin)

    if accepted:
        shares = np.stack(accepted, axis=0)
        margins_arr = np.array(margins, dtype=float)
    else:
        shares = np.empty((0, len(active_idx)), dtype=float)
        margins_arr = np.array([], dtype=float)
    return WeekSamples(
        season=season,
        week=week,
        regime=regime,
        names=names_active,
        shares=shares,
        margins=margins_arr,
        n_elim=k,
        n_trials=n_trials,
    )


def summarize_samples(samples: WeekSamples) -> Tuple[List[Dict], Dict]:
    rows = []
    if samples.shares.shape[0] > 0:
        for i, name in enumerate(samples.names):
            vals = samples.shares[:, i]
            rows.append(
                {
                    "season": samples.season,
                    "week": samples.week,
                    "regime": samples.regime,
                    "celebrity_name": name,
                    "s_min": float(np.min(vals)),
                    "s_p10": float(np.percentile(vals, 10)),
                    "s_p50": float(np.percentile(vals, 50)),
                    "s_p90": float(np.percentile(vals, 90)),
                    "s_max": float(np.max(vals)),
                }
            )
    margin_p10 = float(np.percentile(samples.margins, 10)) if samples.margins.size else np.nan
    margin_p50 = float(np.percentile(samples.margins, 50)) if samples.margins.size else np.nan
    margin_p90 = float(np.percentile(samples.margins, 90)) if samples.margins.size else np.nan
    summary = {
        "season": samples.season,
        "week": samples.week,
        "regime": samples.regime,
        "n_active": len(samples.names),
        "n_elim": samples.n_elim,
        "n_accept": samples.shares.shape[0],
        "n_trials": samples.n_trials,
        "accept_rate": float(samples.shares.shape[0] / max(samples.n_trials, 1)),
        "margin_p10": margin_p10,
        "margin_p50": margin_p50,
        "margin_p90": margin_p90,
    }
    return rows, summary


def plot_uncertainty_heatmap(df: pd.DataFrame, season: int, value_col: str, out_name: str) -> None:
    """Heatmap with consistent title and axes: Fan-share p90 (Season N), Week, Contestant."""
    if df.empty or "season" not in df.columns:
        return
    df_season = df[df["season"] == season].copy()
    if df_season.empty:
        return
    df_season = df_season.sort_values(["celebrity_name", "week"])
    names = df_season["celebrity_name"].unique().tolist()
    name_to_idx = {name: i for i, name in enumerate(names)}
    df_season["name_idx"] = df_season["celebrity_name"].map(name_to_idx)
    pivot = df_season.pivot(index="name_idx", columns="week", values=value_col).sort_index()

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Week")
    ax.set_ylabel("Contestant")
    ax.set_title(f"Fan-share p90 (Season {season})")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([names[i] for i in pivot.index])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fan-share p90")
    png_path = os.path.join(FIG_DIR, out_name)
    pdf_path = os.path.join(FIG_DIR, out_name.replace(".png", ".pdf"))
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_acceptance_rate(week_summary: pd.DataFrame, season: int) -> None:
    df = week_summary[week_summary["season"] == season].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(df["week"], df["accept_rate"], marker="o", linewidth=1.5)
    ax.set_xlabel("Week")
    ax.set_ylabel("Acceptance rate")
    ax.set_title(f"Feasible-set acceptance rate (Season {season})")
    ax.set_ylim(0.0, 1.0)
    fig.savefig(os.path.join(FIG_DIR, f"acceptance_rate_season_{season}.png"), bbox_inches="tight", dpi=300)
    fig.savefig(os.path.join(FIG_DIR, f"acceptance_rate_season_{season}.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Constraint-based uncertainty analysis.")
    parser.add_argument("--seasons", type=int, nargs="+", default=[1, 27, 34], help="Seasons for uncertainty heatmap (default: 1 27 34).")
    parser.add_argument("--samples", type=int, default=300, help="Target accepted samples per week.")
    parser.add_argument("--trials", type=int, default=4000, help="Max trials per week.")
    parser.add_argument("--tol-percent", type=float, default=0.02, help="Slack for percent regime.")
    parser.add_argument("--tol-rank", type=float, default=1.0, help="Slack for rank regimes.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    weeks = parse_week_cols(df)
    share_rows = []
    week_rows = []

    for season in sorted(df["season"].unique(), reverse=True):
        season = int(season)
        df_season = df[df["season"] == season].reset_index(drop=True)
        names = df_season["celebrity_name"].tolist()
        J = build_judge_matrix(df_season, weeks)
        placement_true = df_season["placement"].apply(pd.to_numeric, errors="coerce").tolist()
        elim_week_judge = compute_elim_week(J)
        elim_week_schedule = [0] * (J.shape[0] + 1)
        for ew in elim_week_judge:
            if ew <= J.shape[0]:
                elim_week_schedule[ew] += 1
        elim_week_true = infer_elim_week_from_placement(placement_true, elim_week_schedule)
        for i, p in enumerate(placement_true):
            if p is None or (isinstance(p, float) and np.isnan(p)):
                elim_week_true[i] = elim_week_judge[i]

        for w in range(1, J.shape[0] + 1):
            samples = sample_feasible_shares(
                season,
                w,
                names,
                J[w - 1],
                elim_week_true,
                args.samples,
                args.trials,
                args.tol_percent,
                args.tol_rank,
            )
            if samples is None:
                continue
            rows, summary = summarize_samples(samples)
            summary["tol_percent"] = args.tol_percent
            summary["tol_rank"] = args.tol_rank
            share_rows.extend(rows)
            week_rows.append(summary)

    shares = pd.DataFrame(share_rows)
    week_summary = pd.DataFrame(week_rows)
    shares_path = os.path.join(OUT_DIR, "constraints_shares_uncertainty.csv")
    week_path = os.path.join(OUT_DIR, "constraints_week_summary.csv")
    shares.to_csv(shares_path, index=False)
    week_summary.to_csv(week_path, index=False)

    for s in args.seasons:
        if (shares["season"] == s).any():
            plot_uncertainty_heatmap(
                shares,
                int(s),
                "s_p90",
                f"constraints_uncertainty_p90_season_{s}.png",
            )
    if args.seasons:
        plot_acceptance_rate(week_summary, args.seasons[-1])

    print(f"Wrote {shares_path}")
    print(f"Wrote {week_path}")
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
