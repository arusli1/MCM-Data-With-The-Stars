"""
Evaluate AR-Problem1_InvOpt outcomes with consistency + uncertainty metrics.

Reads:
  - AR-Problem1_InvOpt/inferred_shares.csv
  - AR-Problem1_InvOpt/elimination_match.csv (season-level summary from the original run)
  - Data/2026_MCM_Problem_C_Data.csv (to reconstruct active sets and true eliminations)

Writes:
  - AR-Problem1_InvOpt/outputs/weekly_metrics.csv
  - AR-Problem1_InvOpt/outputs/season_metrics_recomputed.csv
  - AR-Problem1_InvOpt/outputs/regime_metrics.csv
  - AR-Problem1_InvOpt/outputs/bootstrap_summary.csv
  - AR-Problem1_InvOpt/outputs/plots/*.png
  - AR-Problem1_InvOpt/invopt_outcomes_report.md

Important: This is a reconstruction model; priors in this folder explicitly use future outcomes
(see model_notes.md). High "match" rates mostly reflect constraint satisfaction, not prediction.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Matplotlib in headless environments
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(HERE / ".cache"))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def _clean_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def regime_for_season(season: int) -> str:
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom"


def parse_week_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df: pd.DataFrame, week: int, cols: List[str]) -> pd.Series:
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def elimination_week(row: pd.Series, last_active_week: int) -> Optional[int]:
    if isinstance(row.get("results"), str) and "Eliminated Week" in row["results"]:
        return int(row["results"].split("Eliminated Week ")[1])
    if isinstance(row.get("results"), str) and "Withdrew" in row["results"]:
        return last_active_week
    return None


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
    return {"names": names, "J": J[:max_week_active], "elim_week": elim_week, "max_week": max_week_active}


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_dwts = pd.read_csv(ROOT / "Data" / "2026_MCM_Problem_C_Data.csv", na_values=["N/A"])
    df_dwts["celebrity_name"] = df_dwts["celebrity_name"].map(_clean_name)
    df_dwts["season"] = pd.to_numeric(df_dwts["season"], errors="raise").astype(int)

    inferred = pd.read_csv(HERE / "inferred_shares.csv")
    inferred["celebrity_name"] = inferred["celebrity_name"].map(_clean_name)
    inferred["season"] = pd.to_numeric(inferred["season"], errors="raise").astype(int)
    inferred["week"] = pd.to_numeric(inferred["week"], errors="raise").astype(int)
    inferred["s_map"] = pd.to_numeric(inferred["s_map"], errors="coerce")

    match = pd.read_csv(HERE / "elimination_match.csv")
    match["season"] = pd.to_numeric(match["season"], errors="raise").astype(int)
    return df_dwts, inferred, match


def _shares_for_week(inferred: pd.DataFrame, season: int, week: int) -> Dict[str, float]:
    sub = inferred[(inferred["season"] == season) & (inferred["week"] == week)]
    return {r.celebrity_name: float(r.s_map) for r in sub.itertuples(index=False) if pd.notna(r.s_map)}


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    if len(p) == 0:
        return np.nan
    return float(-(p * np.log(p)).sum())


def evaluate_week(
    *,
    season: int,
    week: int,
    regime: str,
    struct: Dict,
    shares_by_name: Dict[str, float],
) -> Dict:
    names = struct["names"]
    J = struct["J"]
    elim_week = struct["elim_week"]

    w_idx = week - 1
    if w_idx < 0 or w_idx >= J.shape[0]:
        return {"season": season, "week": week, "status": "week_out_of_range"}

    active_idx = np.where(J[w_idx] > 0)[0]
    if len(active_idx) < 2:
        return {"season": season, "week": week, "status": "too_few_active"}

    # True eliminations (can be empty in finale/no-elim weeks)
    true_elim = [i for i, ew in enumerate(elim_week) if ew == week and i in set(active_idx)]
    k_elim = len(true_elim)

    # Build shares vector aligned to idx
    s = np.full(len(names), np.nan, dtype=float)
    for i in active_idx:
        n = names[i]
        if n in shares_by_name:
            s[i] = shares_by_name[n]

    s_active = s[active_idx]
    if np.any(~np.isfinite(s_active)):
        return {"season": season, "week": week, "status": "missing_shares_for_active"}
    # Normalize defensively
    s_active = np.clip(s_active, 0.0, 1.0)
    s_active = s_active / s_active.sum() if s_active.sum() > 0 else np.full_like(s_active, 1.0 / len(s_active))

    out: Dict = {
        "season": season,
        "week": week,
        "regime": regime,
        "n_active": int(len(active_idx)),
        "k_elim": int(k_elim),
        "has_elim": int(k_elim > 0),
        "share_entropy": _entropy(s_active),
        "share_entropy_norm": float(_entropy(s_active) / np.log(len(s_active))) if len(s_active) > 1 else np.nan,
        "share_hhi": float(np.sum(s_active**2)),
        "status": "ok",
    }

    # If no elimination observed that week, we treat as "hit" per the model's match-rate logic.
    if k_elim == 0:
        out.update({"hit_set": 1, "top1": np.nan, "elim_rank": np.nan, "margin": np.nan})
        return out

    if regime == "percent":
        j_pct = J[w_idx, active_idx] / J[w_idx, active_idx].sum()
        C = j_pct + s_active  # lower is worse
        order = np.argsort(C)  # ascending = worst first
        pred = set(active_idx[order[:k_elim]])

        # rank of true eliminated under risk ordering (1=most at risk)
        ranks = {int(active_idx[i]): int(pos + 1) for pos, i in enumerate(order)}
        elim_ranks = [ranks[i] for i in true_elim]
        elim_rank = float(np.mean(elim_ranks))

        margin = np.nan
        if len(order) >= k_elim + 1:
            # separation between kth-worst and (k+1)th-worst combined score
            margin = float(C[order[k_elim]] - C[order[k_elim - 1]])

        out.update(
            {
                "hit_set": int(set(true_elim).issubset(pred)),
                "top1": int(k_elim == 1 and (active_idx[order[0]] in set(true_elim))),
                "elim_rank": elim_rank,
                "margin": margin,
            }
        )
        return out

    # rank or bottom: use combined rank R = rJ + rF; higher is worse.
    # rJ: 1 is best judge (highest score)
    rJ = stats.rankdata(-J[w_idx, active_idx], method="ordinal")
    rF = stats.rankdata(-s_active, method="ordinal")
    R = rJ + rF
    order = np.argsort(-R)  # descending = worst first

    if regime == "bottom" and k_elim == 1:
        pred = set(active_idx[order[:2]])
    else:
        pred = set(active_idx[order[:k_elim]])

    ranks = {int(active_idx[i]): int(pos + 1) for pos, i in enumerate(order)}  # 1=most at risk
    elim_ranks = [ranks[i] for i in true_elim]
    elim_rank = float(np.mean(elim_ranks))

    margin = np.nan
    if len(order) >= k_elim + 1:
        margin = float(R[order[k_elim - 1]] - R[order[k_elim]])  # separation (bigger=more confident)

    out.update(
        {
            "hit_set": int(set(true_elim).issubset(pred)),
            "top1": int(k_elim == 1 and (active_idx[order[0]] in set(true_elim))),
            "elim_rank": elim_rank,
            "margin": margin,
        }
    )
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 123) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    vals = values[np.isfinite(values)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    means = []
    for _ in range(n_boot):
        samp = rng.choice(vals, size=len(vals), replace=True)
        means.append(float(np.mean(samp)))
    means = np.array(means)
    return float(np.mean(vals)), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def df_to_md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].map(lambda v: "NA" if pd.isna(v) else str(v))
    headers = list(df.columns)
    rows = df.values.tolist()
    out = []
    out.append("| " + " | ".join(h.replace("|", "\\|") for h in headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(v).replace("|", "\\|") for v in r) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    df_dwts, inferred, match = load_inputs()
    week_cols = parse_week_cols(df_dwts)

    out_dir = HERE / "outputs"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    weekly_rows: List[Dict] = []
    for season in sorted(df_dwts["season"].unique()):
        df_season = df_dwts[df_dwts["season"] == season].reset_index(drop=True)
        struct = build_season_struct(df_season, week_cols)
        regime = regime_for_season(int(season))
        for week in range(1, struct["max_week"] + 1):
            shares_map = _shares_for_week(inferred, int(season), week)
            weekly_rows.append(
                evaluate_week(season=int(season), week=week, regime=regime, struct=struct, shares_by_name=shares_map)
            )

    weekly = pd.DataFrame(weekly_rows)
    weekly.to_csv(out_dir / "weekly_metrics.csv", index=False)

    # Recompute season metrics from weekly
    ok = weekly[weekly["status"] == "ok"].copy()
    season_metrics = (
        ok.groupby("season")
        .agg(
            weeks=("week", "count"),
            matched=("hit_set", "sum"),
            match_rate=("hit_set", "mean"),
            top1_acc=("top1", "mean"),
            mean_elim_rank=("elim_rank", "mean"),
            median_margin=("margin", "median"),
        )
        .reset_index()
    )
    season_metrics["regime"] = season_metrics["season"].map(regime_for_season)
    season_metrics.to_csv(out_dir / "season_metrics_recomputed.csv", index=False)

    # Regime-level summaries (weighted by weeks)
    regime_metrics = (
        ok.groupby("regime")
        .agg(
            weeks=("week", "count"),
            match_rate=("hit_set", "mean"),
            top1_acc=("top1", "mean"),
            mean_elim_rank=("elim_rank", "mean"),
            median_margin=("margin", "median"),
            mean_entropy_norm=("share_entropy_norm", "mean"),
            mean_hhi=("share_hhi", "mean"),
        )
        .reset_index()
        .sort_values("regime")
    )
    regime_metrics.to_csv(out_dir / "regime_metrics.csv", index=False)

    # Bootstrap uncertainty for key metrics (overall and per regime)
    boot_rows = []
    for key, df_sub in [("overall", ok)] + [(f"regime:{r}", ok[ok["regime"] == r]) for r in sorted(ok["regime"].unique())]:
        hit = df_sub["hit_set"].to_numpy(dtype=float)
        top1 = df_sub["top1"].to_numpy(dtype=float)
        elim_rank = df_sub["elim_rank"].to_numpy(dtype=float)
        margin = df_sub["margin"].to_numpy(dtype=float)

        for metric_name, arr in [
            ("match_rate", hit),
            ("top1_acc", top1),
            ("mean_elim_rank", elim_rank),
            ("median_margin", margin),
        ]:
            m, lo, hi = bootstrap_ci(arr, n_boot=2000, seed=7)
            boot_rows.append({"slice": key, "metric": metric_name, "estimate": m, "ci95_low": lo, "ci95_high": hi, "n": int(np.isfinite(arr).sum())})

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out_dir / "bootstrap_summary.csv", index=False)

    # Plots
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=ok, x="regime", y="elim_rank")
    plt.title("Rank of true eliminated under model risk ordering (lower is better)")
    plt.tight_layout()
    plt.savefig(plot_dir / "elim_rank_by_regime.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=ok[ok["k_elim"] == 1], x="regime", y="margin")
    plt.title("Elimination margin (separation) for single-elimination weeks")
    plt.tight_layout()
    plt.savefig(plot_dir / "margin_by_regime_single_elim.png", dpi=200)
    plt.close()

    # Report (research-paper style)
    # Use elimination_match.csv for the original season-level summary (includes kendall_tau, mean_abs_rank_diff).
    merged_season = season_metrics.merge(match, on="season", how="left", suffixes=("_recomputed", "_reported"))
    merged_season = merged_season.sort_values("season")

    # Key highlights
    overall_match = ok["hit_set"].mean()
    overall_top1 = ok["top1"].mean()

    lines = []
    lines.append("# Inverse Optimization (InvOpt) outcomes: consistency & uncertainty report\n\n")
    lines.append("## Abstract\n")
    lines.append(
        "We evaluate the inverse-optimization reconstruction outputs in `AR-Problem1_InvOpt/`, "
        "quantifying (i) **consistency** with the stated elimination rules and (ii) **uncertainty** "
        "in reported evaluation metrics via bootstrap confidence intervals. "
        "Because the InvOpt model’s popularity prior uses future outcomes by design (see `model_notes.md`), "
        "these results should be interpreted as **reconstruction consistency**, not predictive accuracy.\n\n"
    )

    lines.append("## Data & outputs evaluated\n")
    lines.append("- **DWTS source**: `Data/2026_MCM_Problem_C_Data.csv`\n")
    lines.append("- **Inferred fan shares (MAP)**: `AR-Problem1_InvOpt/inferred_shares.csv` (`s_map`)\n")
    lines.append("- **Season-level reported diagnostics**: `AR-Problem1_InvOpt/elimination_match.csv`\n")
    lines.append("\n")

    lines.append("## Methods\n")
    lines.append("### Active set and eliminations\n")
    lines.append(
        "For each season/week, the active set is contestants with positive weekly judge totals (J>0). "
        "True elimination week is parsed from the `results` field; withdrawals are treated as elimination at the last active week.\n\n"
    )
    lines.append("### Regime-specific risk ordering\n")
    lines.append("- **Percent (S3–S27)**: C = j_pct + s. Lower C implies higher elimination risk.\n")
    lines.append("- **Rank (S1–S2)**: combined rank R = r_J + r_F. Higher R implies higher risk.\n")
    lines.append("- **Bottom-two (S28+)**: the eliminated must lie within the bottom-two by R (for single eliminations).\n\n")

    lines.append("### Consistency metrics\n")
    lines.append("- **Set match**: whether the true eliminated set is a subset of the model’s predicted worst set (mirrors the code’s logic).\n")
    lines.append("- **Top-1 accuracy** (single-elimination weeks): whether the model’s single highest-risk contestant matches the eliminated.\n")
    lines.append("- **Eliminated risk rank**: rank position of the true eliminated under the model’s risk ordering (1 = most at risk).\n\n")

    lines.append("### Uncertainty metrics\n")
    lines.append(
        "The folder does not currently include the ensemble uncertainty CSVs referenced in `model_notes.md` "
        "(they would require `cvxpy` and repeated randomized solves). "
        "We therefore report uncertainty as **bootstrap 95% confidence intervals** over weeks for aggregate evaluation metrics, "
        "plus **per-week separation margins** (how close the decision boundary is) as a proxy for identifiability.\n\n"
    )

    lines.append("## Results\n")
    lines.append(f"- Overall set-match rate (week-weighted): **{overall_match:.3f}**\n")
    lines.append(f"- Overall top-1 accuracy (single-elim weeks): **{overall_top1:.3f}**\n\n")

    lines.append("### Regime-level summary (recomputed)\n")
    lines.append(df_to_md_table(regime_metrics.round(4)))
    lines.append("\n")

    lines.append("### Season-level summary (recomputed + reported diagnostics)\n")
    # After merge, overlapping columns (weeks/matched/match_rate) are suffixed.
    season_view = merged_season.rename(
        columns={
            "weeks_recomputed": "weeks",
            "matched_recomputed": "matched",
            "match_rate_recomputed": "match_rate",
            "weeks_reported": "weeks_reported",
            "matched_reported": "matched_reported",
            "match_rate_reported": "match_rate_reported",
        }
    )
    show_cols = [
        "season",
        "regime",
        "weeks",
        "matched",
        "match_rate",
        "top1_acc",
        "mean_elim_rank",
        "median_margin",
        "match_rate_reported",
        "kendall_tau",
        "mean_abs_rank_diff",
    ]
    show_cols = [c for c in show_cols if c in season_view.columns]
    lines.append(df_to_md_table(season_view[show_cols].round(4)))
    lines.append("\n")

    lines.append("### Bootstrap uncertainty (95% CI)\n")
    lines.append(
        "We bootstrap weeks within each slice (overall / per-regime) and report the sampling distribution "
        "of aggregate metrics.\n\n"
    )
    lines.append(df_to_md_table(boot.round(5)))
    lines.append("\n")

    lines.append("### Plots\n")
    lines.append(f"- `{(plot_dir / 'elim_rank_by_regime.png').as_posix()}`\n")
    lines.append(f"- `{(plot_dir / 'margin_by_regime_single_elim.png').as_posix()}`\n\n")

    lines.append("## Discussion\n")
    lines.append(
        "1) **Very high match rates in percent-era seasons are expected** because the optimization enforces the elimination "
        "rules as hard constraints, and the popularity prior is constructed using realized elimination timing.\n\n"
        "2) **Bottom-two seasons show weaker guarantees** because the constraints are less informative (membership in a bottom-two set "
        "instead of a unique minimum) and discrete ranking introduces many equivalent solutions.\n\n"
        "3) **Uncertainty is dominated by identifiability rather than sampling noise**: when margins are small, many nearby share vectors "
        "can satisfy the constraints. Ensemble-based uncertainty (randomized priors) would quantify this more directly.\n"
    )

    lines.append("\n## Reproducibility notes\n")
    lines.append(
        "- This analysis recomputes weekly metrics from `inferred_shares.csv` and the original DWTS data.\n"
        "- If you want true solution uncertainty (std/quantiles of shares), run the ensemble pipeline (see `infer_votes_ensemble.py`) "
        "after installing `cvxpy` and `pulp`.\n"
    )

    report_path = HERE / "invopt_outcomes_report.md"
    report_path.write_text("".join(lines), encoding="utf-8")

    print(f"Wrote: {report_path}")
    print(f"Wrote outputs under: {out_dir}")


if __name__ == "__main__":
    main()


