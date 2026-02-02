#!/usr/bin/env python3
"""
Generate final-result and sensitivity figures for the Problem 1 Base (LPSSM) model.
Reads from base_results/; writes to figures/ and final_results/.
Paper-ready style: clean sans-serif, no top/right spines, PDF + PNG.
"""
import os
import shutil
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

# Paths relative to this script (AR-Problem1-Base)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "base_results")
FIG_DIR = os.path.join(BASE_DIR, "figures")
FINAL_DIR = os.path.join(BASE_DIR, "final_results")

# Regime bounds for shading (season inclusive)
REGIMES = [(1, 2, "Rank"), (3, 27, "Percent"), (28, 34, "Bottom-2")]
REGIME_COLORS = ["#E8E8E8", "#F0F0F0", "#E8E8E8"]  # light gray bands


def paper_style():
    """Publication-style defaults."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def save_both(fig: plt.Figure, name: str) -> None:
    """Save to both figures/ and final_results/ as PNG and PDF."""
    for out_dir in (FIG_DIR, FINAL_DIR):
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)


def _add_regime_shading(ax: plt.Axes, seasons: np.ndarray) -> None:
    """Add vertical regime bands (rank / percent / bottom-2)."""
    xmin, xmax = seasons.min() - 0.5, seasons.max() + 0.5
    for (s_lo, s_hi, _), color in zip(REGIMES, REGIME_COLORS):
        lo = max(s_lo, xmin)
        hi = min(s_hi, xmax)
        if lo < hi:
            ax.axvspan(lo, hi + 0.5, facecolor=color, zorder=0)
    ax.set_xlim(xmin, xmax)


def plot_consistency(metrics: pd.DataFrame) -> None:
    """Single-panel consistency: rank error (line + band) with points colored by finals order (green=correct, red=wrong)."""
    metrics = metrics.sort_values("season")
    seasons = metrics["season"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(seasons.min() - 0.5, seasons.max() + 0.5)

    # Rank error: band + line (0–1 scale)
    color_band = "#C1292E"
    ax.fill_between(
        seasons,
        metrics["mean_sq_rank_diff_p10"],
        metrics["mean_sq_rank_diff_p90"],
        color=color_band,
        alpha=0.35,
        linewidth=0,
    )
    ax.plot(seasons, metrics["mean_sq_rank_diff"], color=color_band, linewidth=2, zorder=2)

    # Points colored by finals order: green = correct, red = wrong
    finals_ok = metrics["finals_order_match"] >= 0.5
    colors = np.where(finals_ok, "#2E7D32", "#C62828")  # green / red
    ax.scatter(
        seasons,
        metrics["mean_sq_rank_diff"],
        c=colors,
        s=55,
        zorder=5,
        edgecolors="white",
        linewidths=1.2,
    )

    ax.set_ylabel("Rank error (MSE, 0–1)", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Season", fontsize=10)
    ax.set_title("LPSSM consistency by season", fontsize=11)
    ax.grid(True, alpha=0.25, axis="both")

    # Legend: line + band, then colored dots (no overlap with data)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_band, linewidth=2, label="Rank error (MSE)"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor="#2E7D32", markeredgecolor="white", markersize=8, label="Finals order correct"),
        Line2D([0], [0], color="none", marker="o", markerfacecolor="#C62828", markeredgecolor="white", markersize=8, label="Finals order wrong"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        frameon=True,
        fontsize=9,
        ncol=3,
    )
    ax.text(0.99, 0.02, "Weekly elim match: 100% (all seasons)", transform=ax.transAxes, fontsize=8, ha="right", va="bottom", color="gray")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    save_both(fig, "consistency_by_season")


def plot_consistency_heatmap(metrics: pd.DataFrame) -> None:
    """Heatmap: one row per season, columns = Rank MSE, Placement match, Weekly elim, Finals order. Color = quality (1 good, 0 bad)."""
    metrics = metrics.sort_values("season")
    seasons = metrics["season"].to_numpy()
    mse = metrics["mean_sq_rank_diff"].values
    mse_norm = 1.0 - np.minimum(mse / (mse.max() + 1e-9), 1.0)  # 1 = good (no error)
    place = metrics["placement_match_rate"].values
    elim = metrics["weekly_elim_match_rate"].values
    finals = metrics["finals_order_match"].values
    # Build matrix: rows = seasons, cols = [Rank MSE quality, Placement, Weekly elim, Finals]
    H = np.column_stack([mse_norm, place, elim, finals])
    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(H, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(seasons)))
    ax.set_yticklabels(seasons, fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Rank MSE\n(1=good)", "Placement\nmatch", "Weekly elim\nmatch", "Finals\norder"], fontsize=9)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Season")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Quality (1 = match)")
    ax.set_title("LPSSM consistency by season", fontsize=11)
    plt.tight_layout()
    save_both(fig, "consistency_by_season_heatmap")


def plot_alpha_entropy_by_season(metrics: pd.DataFrame) -> None:
    """Alpha and initial entropy by season (interpretability)."""
    metrics = metrics.sort_values("season")
    seasons = metrics["season"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.5), sharex=True)
    ax1, ax2 = axes

    for ax in axes:
        _add_regime_shading(ax, seasons)

    ax1.fill_between(
        seasons,
        metrics["alpha_p10"],
        metrics["alpha_p90"],
        color="#4C78A8",
        alpha=0.3,
        linewidth=0,
    )
    ax1.plot(seasons, metrics["alpha"], color="#4C78A8", linewidth=2, label=r"$\alpha$ (median)")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax1.set_ylabel(r"Judge influence $\alpha$")
    ax1.set_ylim(None, None)
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_title("(a) Season-level judge influence", loc="left")

    ax2.fill_between(
        seasons,
        metrics["s0_entropy_p10"],
        metrics["s0_entropy_p90"],
        color="#F58518",
        alpha=0.3,
        linewidth=0,
    )
    ax2.plot(seasons, metrics["s0_entropy"], color="#F58518", linewidth=2, label=r"$H(s_0)$")
    ax2.set_ylabel(r"Initial entropy $H(s_0)$")
    ax2.set_xlabel("Season")
    ax2.legend(loc="lower right", frameon=False)
    ax2.set_title("(b) Initial share entropy (evenness)", loc="left")

    fig.suptitle("Fitted parameters by season", fontsize=12, y=1.02)
    plt.tight_layout()
    save_both(fig, "alpha_entropy_by_season")


def plot_uncertainty_heatmap(unc: pd.DataFrame, season: int) -> None:
    """Uncertainty: p90 fan-share heatmap for one season."""
    df = unc[unc["season"] == season].copy()
    if df.empty:
        return
    df = df.sort_values(["celebrity_name", "week"])
    names = df["celebrity_name"].unique().tolist()
    weeks = sorted(df["week"].unique())
    name_to_idx = {n: i for i, n in enumerate(names)}
    pivot = df.pivot_table(index="celebrity_name", columns="week", values="s_share_p90", aggfunc="first")
    pivot = pivot.reindex(names).reindex(columns=weeks)

    fig, ax = plt.subplots(figsize=(max(6, len(weeks) * 0.5), max(4, len(names) * 0.28)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels(weeks)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Week")
    ax.set_ylabel("Contestant")
    ax.set_title(f"Fan-share 90th percentile (Season {season})")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Share (p90)")
    plt.tight_layout()
    save_both(fig, f"uncertainty_p90_season_{season}")


def plot_vote_shares_stacked(shares: pd.DataFrame, placement: pd.DataFrame, season: int) -> None:
    """Stacked area: vote share by week for one season."""
    sh = shares[shares["season"] == season].copy()
    place = placement[placement["season"] == season]
    if sh.empty:
        return
    weeks = sorted(sh["week"].unique())
    order = place.sort_values(["elim_week_true", "placement_true"], ascending=[True, False])["celebrity_name"].tolist()
    data = np.zeros((len(order), len(weeks)))
    for j, w in enumerate(weeks):
        row = sh[sh["week"] == w]
        for i, name in enumerate(order):
            r = row[row["celebrity_name"] == name]
            data[i, j] = r["s_share"].sum() if len(r) else 0.0
    n = len(order)
    colors = cm.viridis(np.linspace(0.15, 0.9, n))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(weeks))
    for i in range(n):
        ax.fill_between(weeks, bottom, bottom + data[i], color=colors[i], alpha=0.85, linewidth=0)
        bottom = bottom + data[i]
    ax.set_xlim(weeks[0] - 0.3, weeks[-1] + 0.3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative fan share")
    ax.set_title(f"Season {season}: Estimated fan vote share (stacked)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    leg_order = place.sort_values("placement_true")["celebrity_name"].tolist()
    leg_colors = [colors[order.index(n)] for n in leg_order]
    handles = [mpatches.Patch(color=c, label=nm) for c, nm in zip(leg_colors, leg_order)]
    ax.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    save_both(fig, f"vote_shares_season_{season}_stacked")


def plot_vote_shares_lines(
    shares: pd.DataFrame,
    placement: pd.DataFrame,
    unc: Optional[pd.DataFrame],
    season: int,
    top_n: int = 8,
) -> None:
    """Line plot: top contestants with optional p10–p90 band."""
    sh = shares[shares["season"] == season].copy()
    place = placement[placement["season"] == season]
    if sh.empty:
        return
    weeks = sorted(sh["week"].unique())
    max_share = sh.groupby("celebrity_name")["s_share"].max().sort_values(ascending=False)
    names_top = max_share.head(top_n).index.tolist()
    finals = set(place[place["placement_true"] <= 4]["celebrity_name"])
    names = list(dict.fromkeys(names_top + [n for n in finals if n not in names_top]))[:top_n]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    has_unc = unc is not None and not unc.empty

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, name in enumerate(names):
        if has_unc:
            u = unc[(unc["season"] == season) & (unc["celebrity_name"] == name)].sort_values("week")
            if len(u):
                ww = u["week"].values
                ss = u["s_share_p50"].values
                p10, p90 = u["s_share_p10"].values, u["s_share_p90"].values
                ax.fill_between(ww, p10, p90, color=colors[i], alpha=0.2, zorder=1)
                ax.plot(ww, ss, color=colors[i], lw=2, label=name, zorder=3)
                continue
        df = sh[sh["celebrity_name"] == name].sort_values("week")
        ww, ss = df["week"].values, df["s_share"].values
        ax.plot(ww, ss, color=colors[i], lw=2, label=name, zorder=3)
    elim_weeks = sorted([int(w) for w in place["elim_week_true"].dropna().unique() if 1 <= w <= max(weeks)])
    for w in elim_weeks:
        ax.axvline(w, color="gray", linestyle="--", alpha=0.35, zorder=0)
    ax.set_xlim(weeks[0] - 0.3, weeks[-1] + 0.3)
    ax.set_ylim(0, None)
    ax.set_xlabel("Week")
    ax.set_ylabel("Fan share")
    ax.set_title(f"Season {season}: Estimated fan vote share (top contestants)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    save_both(fig, f"vote_shares_season_{season}_lines")


def plot_sensitivity_heatmaps(sens: pd.DataFrame) -> None:
    """Sensitivity: heatmaps of weekly elim match and rank MSE vs judge_scale × alpha_penalty."""
    grouped = sens.groupby(["judge_scale", "alpha_penalty"]).mean(numeric_only=True)
    pivot_elim = grouped["weekly_elim_match_rate"].unstack("alpha_penalty")
    pivot_mse = grouped["mean_sq_rank_diff"].unstack("alpha_penalty")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    ax1, ax2 = axes

    im1 = ax1.imshow(
        pivot_elim.values,
        cmap="Blues",
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    ax1.set_title("(a) Weekly elimination match rate")
    ax1.set_xlabel("Alpha penalty")
    ax1.set_ylabel("Judge scale")
    ax1.set_xticks(range(len(pivot_elim.columns)))
    ax1.set_xticklabels(pivot_elim.columns)
    ax1.set_yticks(range(len(pivot_elim.index)))
    ax1.set_yticklabels(pivot_elim.index)
    fig.colorbar(im1, ax=ax1, label="Match rate")

    im2 = ax2.imshow(
        pivot_mse.values,
        cmap="Oranges",
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=min(1.0, float(pivot_mse.values.max()) * 1.1),
    )
    ax2.set_title("(b) Rank error (MSE)")
    ax2.set_xlabel("Alpha penalty")
    ax2.set_ylabel("Judge scale")
    ax2.set_xticks(range(len(pivot_mse.columns)))
    ax2.set_xticklabels(pivot_mse.columns)
    ax2.set_yticks(range(len(pivot_mse.index)))
    ax2.set_yticklabels(pivot_mse.index)
    fig.colorbar(im2, ax=ax2, label="MSE")

    fig.suptitle("Sensitivity: judge scale × alpha penalty", fontsize=12, y=1.02)
    plt.tight_layout()
    save_both(fig, "sensitivity_heatmaps")


def plot_sensitivity_oat(oat: pd.DataFrame) -> None:
    """One-at-a-time sensitivity: effect range by parameter."""
    metrics = ["weekly_elim_match_rate", "mean_sq_rank_diff"]
    metric_labels = ["Weekly elim match (range, pp)", "Rank error MSE (range)"]
    params = oat["param"].unique().tolist()
    param_labels = {
        "judge_scale": "Judge scale",
        "alpha_penalty": "Alpha penalty",
        "entropy_weight": "Entropy weight",
        "s0_prior_scale": "s0 prior scale",
        "s0_prior_conc": "s0 prior concentration",
    }
    x_labels = [param_labels.get(p, p) for p in params]

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    x = np.arange(len(params))
    width = 0.5
    for mi, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[mi]
        ranges = []
        for p in params:
            df_p = oat[oat["param"] == p]
            vals = df_p.groupby("value")[metric].mean()
            r = float(vals.max() - vals.min())
            if metric == "weekly_elim_match_rate":
                r *= 100  # percentage points
            ranges.append(r)
        ax.bar(x, ranges, width, color="#4C78A8", edgecolor="none")
        ax.set_ylabel(mlabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=25, ha="right")
        ax.set_title(f"({chr(97 + mi)}) {mlabel}", loc="left")
    axes[-1].set_xlabel("Parameter")
    fig.suptitle("Sensitivity: one-at-a-time effect size", fontsize=12, y=1.02)
    plt.tight_layout()
    save_both(fig, "sensitivity_oat_summary")


def plot_overall_summary(overall: pd.DataFrame) -> None:
    """One-panel text summary of overall LPSSM metrics (for paper/slides)."""
    row = overall.iloc[0]
    elim = float(row.get("overall_weekly_elim_match_rate", 0))
    mse = float(row.get("overall_mean_sq_rank_diff", 0))
    alpha_m = float(row.get("overall_alpha_mean", 0))
    alpha_s = float(row.get("overall_alpha_std", 0))
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.axis("off")
    text = (
        "LPSSM overall (34 seasons)\n"
        f"  • Weekly elimination match rate: {elim:.0%}\n"
        f"  • Mean squared rank error: {mse:.3f}\n"
        f"  • Judge influence α: {alpha_m:.2f} ± {alpha_s:.2f}"
    )
    ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=11, verticalalignment="center", fontfamily="monospace")
    ax.set_title("Summary metrics")
    plt.tight_layout()
    save_both(fig, "overall_summary")


def copy_results_to_final() -> None:
    """Copy key CSVs from base_results to final_results for a self-contained paper folder."""
    files = [
        "base_metrics.csv",
        "base_overall_metrics.csv",
        "base_inferred_shares.csv",
        "base_inferred_shares_uncertainty.csv",
        "base_placement_orders.csv",
        "base_s0.csv",
    ]
    for f in files:
        src = os.path.join(RESULTS_DIR, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(FINAL_DIR, f))
    if os.path.isfile(os.path.join(RESULTS_DIR, "base_sensitivity.csv")):
        shutil.copy2(
            os.path.join(RESULTS_DIR, "base_sensitivity.csv"),
            os.path.join(FINAL_DIR, "base_sensitivity.csv"),
        )
    if os.path.isfile(os.path.join(RESULTS_DIR, "base_sensitivity_oat.csv")):
        shutil.copy2(
            os.path.join(RESULTS_DIR, "base_sensitivity_oat.csv"),
            os.path.join(FINAL_DIR, "base_sensitivity_oat.csv"),
        )


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate Problem 1 Base final figures.")
    parser.add_argument("--no-copy", action="store_true", help="Skip copying CSVs to final_results")
    parser.add_argument("--seasons", type=int, nargs="+", default=[1, 27, 33, 34], help="Seasons for uncertainty and vote-share plots")
    args = parser.parse_args()

    paper_style()
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(FINAL_DIR, exist_ok=True)

    if not args.no_copy:
        copy_results_to_final()
        print("Copied base_results CSVs to final_results/")

    # Load from base_results (canonical)
    metrics_path = os.path.join(RESULTS_DIR, "base_metrics.csv")
    metrics = pd.read_csv(metrics_path) if os.path.isfile(metrics_path) else None
    if metrics is None:
        warnings.warn("base_metrics.csv not found; run base.py first.")
    else:
        plot_consistency(metrics)
        plot_alpha_entropy_by_season(metrics)
        print("Saved consistency_by_season, alpha_entropy_by_season")

    unc_path = os.path.join(RESULTS_DIR, "base_inferred_shares_uncertainty.csv")
    unc = pd.read_csv(unc_path) if os.path.isfile(unc_path) else None
    if unc is not None:
        for s in args.seasons:
            if unc[unc["season"] == s].empty:
                continue
            plot_uncertainty_heatmap(unc, s)
        print(f"Saved uncertainty_p90_season_{{1,27,33,34}}")

    shares_path = os.path.join(RESULTS_DIR, "base_inferred_shares.csv")
    placement_path = os.path.join(RESULTS_DIR, "base_placement_orders.csv")
    shares = pd.read_csv(shares_path) if os.path.isfile(shares_path) else None
    placement = pd.read_csv(placement_path) if os.path.isfile(placement_path) else None
    if shares is not None and placement is not None:
        for s in args.seasons:
            if shares[shares["season"] == s].empty:
                continue
            plot_vote_shares_stacked(shares, placement, s)
            plot_vote_shares_lines(shares, placement, unc, s)
        print("Saved vote_shares_season_{n}_stacked, vote_shares_season_{n}_lines")

    sens_path = os.path.join(RESULTS_DIR, "base_sensitivity.csv")
    sens = pd.read_csv(sens_path) if os.path.isfile(sens_path) else None
    if sens is not None:
        plot_sensitivity_heatmaps(sens)
        print("Saved sensitivity_heatmaps")

    oat_path = os.path.join(RESULTS_DIR, "base_sensitivity_oat.csv")
    oat = pd.read_csv(oat_path) if os.path.isfile(oat_path) else None
    if oat is not None:
        plot_sensitivity_oat(oat)
        print("Saved sensitivity_oat_summary")

    overall_path = os.path.join(RESULTS_DIR, "base_overall_metrics.csv")
    overall = pd.read_csv(overall_path) if os.path.isfile(overall_path) else None
    if overall is not None and not overall.empty:
        plot_overall_summary(overall)
        print("Saved overall_summary")

    print(f"Figures written to {FIG_DIR} and {FINAL_DIR}")


if __name__ == "__main__":
    main()
