import argparse
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "AR-Problem1-Base/base_results"
FIG_DIR = "AR-Problem1-Base/figures"


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


def plot_consistency(metrics: pd.DataFrame) -> None:
    seasons = metrics["season"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    ax1, ax2 = axes

    ax1.plot(seasons, metrics["weekly_elim_match_rate"], color="#4C78A8", label="Weekly elim match")
    ax1.fill_between(
        seasons,
        metrics["weekly_elim_match_rate_p10"],
        metrics["weekly_elim_match_rate_p90"],
        color="#4C78A8",
        alpha=0.2,
        linewidth=0,
        label="p10-p90",
    )
    ax1.set_ylabel("Match Rate")
    ax1.legend(frameon=False)

    ax2.plot(seasons, metrics["mean_sq_rank_diff"], color="#F58518", label="Rank error (MSE)")
    ax2.fill_between(
        seasons,
        metrics["mean_sq_rank_diff_p10"],
        metrics["mean_sq_rank_diff_p90"],
        color="#F58518",
        alpha=0.2,
        linewidth=0,
        label="p10-p90",
    )
    ax2.set_ylabel("Rank Error (MSE)")
    ax2.set_xlabel("Season")
    ax2.legend(frameon=False)

    fig.suptitle("Consistency with Eliminations")
    save_figure(fig, "consistency_by_season")
    plt.close(fig)


def plot_uncertainty(season_unc: pd.DataFrame, season: int) -> None:
    """Heatmap with same title and axes as constraint uncertainty: Fan-share p90 (Season N)."""
    df = season_unc[season_unc["season"] == season].copy()
    if df.empty:
        warnings.warn(f"No uncertainty data for season {season}")
        return
    df = df.sort_values(["celebrity_name", "week"])
    names = df["celebrity_name"].unique().tolist()
    name_to_idx = {name: i for i, name in enumerate(names)}
    df["name_idx"] = df["celebrity_name"].map(name_to_idx)

    pivot = df.pivot(index="name_idx", columns="week", values="s_share_p90")
    pivot = pivot.sort_index()

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
    save_figure(fig, f"uncertainty_p90_season_{season}")
    plt.close(fig)


def plot_sensitivity(sens: pd.DataFrame) -> None:
    grouped = sens.groupby(["judge_scale", "alpha_penalty"]).mean(numeric_only=True)
    pivot_elim = grouped["weekly_elim_match_rate"].unstack("alpha_penalty")
    pivot_mse = grouped["mean_sq_rank_diff"].unstack("alpha_penalty")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    ax1, ax2 = axes

    im1 = ax1.imshow(
        pivot_elim.values, cmap="Blues", aspect="auto", origin="lower", vmin=0.0, vmax=1.0
    )
    ax1.set_title("Weekly Elim Match (mean)")
    ax1.set_xlabel("Alpha penalty")
    ax1.set_ylabel("Judge scale")
    ax1.set_xticks(range(len(pivot_elim.columns)))
    ax1.set_xticklabels(pivot_elim.columns)
    ax1.set_yticks(range(len(pivot_elim.index)))
    ax1.set_yticklabels(pivot_elim.index)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Match rate (0-1)")

    im2 = ax2.imshow(
        pivot_mse.values, cmap="Oranges", aspect="auto", origin="lower", vmin=0.0, vmax=1.0
    )
    ax2.set_title("Rank Error (MSE, mean)")
    ax2.set_xlabel("Alpha penalty")
    ax2.set_ylabel("Judge scale")
    ax2.set_xticks(range(len(pivot_mse.columns)))
    ax2.set_xticklabels(pivot_mse.columns)
    ax2.set_yticks(range(len(pivot_mse.index)))
    ax2.set_yticklabels(pivot_mse.index)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("MSE (0-1)")

    fig.suptitle("Sensitivity Analysis")
    save_figure(fig, "sensitivity_heatmaps")
    plt.close(fig)


def plot_sensitivity_summary(sens: pd.DataFrame) -> None:
    metrics = {
        "weekly_elim_match_rate": "Weekly elim match (mean)",
        "mean_sq_rank_diff": "Rank error MSE (mean)",
        "elim_week_mae": "Elim week MAE (mean)",
    }
    params = ["judge_scale", "alpha_penalty"]
    rows = []
    for param in params:
        grouped = sens.groupby(param).mean(numeric_only=True)
        for metric in metrics:
            vals = grouped[metric].to_numpy()
            rows.append(
                {
                    "param": param,
                    "metric": metric,
                    "range": float(vals.max() - vals.min()),
                }
            )
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x = np.arange(len(metrics))
    width = 0.35
    for i, param in enumerate(params):
        vals = []
        for metric in metrics:
            vals.append(
                summary.loc[
                    (summary["param"] == param) & (summary["metric"] == metric), "range"
                ].iloc[0]
            )
        ax.bar(x + (i - 0.5) * width, vals, width, label=param)
    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics.values()))
    ax.set_ylabel("Range across settings")
    ax.set_title("Sensitivity Analysis: One-at-a-time effect size")
    ax.legend(frameon=False)
    save_figure(fig, "sensitivity_one_at_a_time")
    plt.close(fig)


def plot_oat_summary(oat: pd.DataFrame) -> None:
    metrics = {
        "weekly_elim_match_rate": "Weekly elim match (range, pp)",
        "mean_sq_rank_diff": "Rank error MSE (range)",
    }
    rows = []
    for param in sorted(oat["param"].unique()):
        df_param = oat[oat["param"] == param]
        grouped = df_param.groupby("value").mean(numeric_only=True)
        for metric in metrics:
            vals = grouped[metric].to_numpy()
            metric_range = float(vals.max() - vals.min())
            if metric == "weekly_elim_match_rate":
                metric_range *= 100.0  # show in percentage points
            rows.append({"param": param, "metric": metric, "range": metric_range})
    summary = pd.DataFrame(rows)

    params = summary["param"].unique().tolist()
    x = np.arange(len(params))
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
    for ax, metric in zip(axes, metrics):
        vals = [
            summary.loc[(summary["param"] == p) & (summary["metric"] == metric), "range"].iloc[0]
            for p in params
        ]
        ax.bar(x, vals, color="#4C78A8")
        ax.set_ylabel(metrics[metric])
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(params, rotation=30, ha="right")
    fig.suptitle("Sensitivity Analysis: One-at-a-time effect size")
    save_figure(fig, "sensitivity_oat_summary")
    plt.close(fig)


def load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        warnings.warn(f"Missing file: {path}")
        return None
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Charts for Problem 1 base model.")
    parser.add_argument("--seasons", type=int, nargs="+", default=[1, 27, 34], help="Seasons for uncertainty heatmap (default: 1 27 34).")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    configure_plot_style()

    metrics = load_csv(os.path.join(RESULTS_DIR, "base_metrics.csv"))
    if metrics is not None:
        plot_consistency(metrics)

    uncertainty = load_csv(os.path.join(RESULTS_DIR, "base_inferred_shares_uncertainty.csv"))
    if uncertainty is not None:
        for s in args.seasons:
            plot_uncertainty(uncertainty, s)

    sensitivity = load_csv(os.path.join(RESULTS_DIR, "base_sensitivity.csv"))
    if sensitivity is not None:
        plot_sensitivity(sensitivity)
        plot_sensitivity_summary(sensitivity)
    else:
        warnings.warn("Sensitivity file missing. Run SENSITIVITY_MODE=1 base.py to create it.")

    oat = load_csv(os.path.join(RESULTS_DIR, "base_sensitivity_oat.csv"))
    if oat is not None:
        plot_oat_summary(oat)

    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
