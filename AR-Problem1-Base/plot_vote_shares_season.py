#!/usr/bin/env python3
"""
Plot estimated fan vote shares for a single season from the base (LPSSM) model.
Produces a creative visualization: stacked area + line plot with optional uncertainty bands.
"""
import os
import argparse
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import cm

# Default paths relative to repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARES_CSV = os.path.join(REPO_ROOT, "AR-Problem1-Base", "final_results", "base_inferred_shares.csv")
UNCERTAINTY_CSV = os.path.join(REPO_ROOT, "AR-Problem1-Base", "final_results", "base_inferred_shares_uncertainty.csv")
PLACEMENT_CSV = os.path.join(REPO_ROOT, "AR-Problem1-Base", "final_results", "base_placement_orders.csv")
OUT_DIR = os.path.join(REPO_ROOT, "AR-Problem1-Base", "final_results")


def load_season_data(
    season: int, shares_path: str, uncertainty_path: str, placement_path: str
):
    sh = pd.read_csv(shares_path)
    sh = sh[sh["season"] == season].copy()
    sh["s_share"] = sh["s_share"].clip(0, 1)

    place = pd.read_csv(placement_path)
    place = place[place["season"] == season][
        ["celebrity_name", "placement_true", "elim_week_true"]
    ]

    unc = None
    if os.path.isfile(uncertainty_path):
        unc = pd.read_csv(uncertainty_path)
        unc = unc[unc["season"] == season].copy()

    return sh, place, unc


def _draw_stacked(ax, season: int, sh: pd.DataFrame, place: pd.DataFrame):
    """Draw stacked area on given axes (for side-by-side or standalone)."""
    weeks = sorted(sh["week"].unique())
    order = place.sort_values(["elim_week_true", "placement_true"], ascending=[True, False])[
        "celebrity_name"
    ].tolist()
    names = order
    data = np.zeros((len(names), len(weeks)))
    for j, w in enumerate(weeks):
        row = sh[sh["week"] == w]
        for i, name in enumerate(names):
            r = row[row["celebrity_name"] == name]
            data[i, j] = r["s_share"].sum() if len(r) else 0.0
    n = len(names)
    colors = cm.viridis(np.linspace(0.15, 0.9, n))
    bottom = np.zeros(len(weeks))
    for i in range(n):
        ax.fill_between(weeks, bottom, bottom + data[i], color=colors[i], alpha=0.85, linewidth=0)
        bottom = bottom + data[i]
    ax.set_xlim(weeks[0] - 0.3, weeks[-1] + 0.3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Week", fontsize=10)
    ax.set_ylabel("Cumulative fan share", fontsize=10)
    ax.set_title("Stacked", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    leg_ord = place.sort_values("placement_true")["celebrity_name"].tolist()
    leg_colors = [colors[names.index(n)] for n in leg_ord]
    handles = [mpatches.Patch(color=c, label=nm) for c, nm in zip(leg_colors, leg_ord)]
    ax.legend(handles=handles, fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))


def _draw_lines(ax, season: int, sh: pd.DataFrame, place: pd.DataFrame, unc: Optional[pd.DataFrame], top_n: int = 6):
    """Draw line plot (tab10) on given axes."""
    weeks = sorted(sh["week"].unique())
    max_share = sh.groupby("celebrity_name")["s_share"].max().sort_values(ascending=False)
    names_top = max_share.head(top_n).index.tolist()
    finals = set(place[place["placement_true"] <= 4]["celebrity_name"])
    names = list(dict.fromkeys(names_top + [n for n in finals if n not in names_top]))[:max(top_n, 6)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    has_unc = unc is not None and not unc.empty
    for i, name in enumerate(names):
        if has_unc:
            u = unc[unc["celebrity_name"] == name].sort_values("week")
            if len(u):
                ww = u["week"].values
                ss = u["s_share_p50"].values
                p10, p90 = u["s_share_p10"].values, u["s_share_p90"].values
                last_active = np.where(ss > 1e-6)[0]
                if len(last_active):
                    last_active = last_active[-1]
                    ww, ss = ww[: last_active + 1], ss[: last_active + 1]
                    p10, p90 = p10[: last_active + 1], p90[: last_active + 1]
                ax.fill_between(ww, p10, p90, color=colors[i], alpha=0.22, zorder=1)
                ax.plot(ww, ss, color=colors[i], lw=2, label=name, zorder=3)
                continue
        df = sh[sh["celebrity_name"] == name].sort_values("week")
        ww, ss = df["week"].values, df["s_share"].values
        last_active = np.where(ss > 1e-6)[0]
        if len(last_active):
            last_active = last_active[-1]
            ww, ss = ww[: last_active + 1], ss[: last_active + 1]
        ax.plot(ww, ss, color=colors[i], lw=2, label=name, zorder=3)
    elim_weeks = sorted([int(w) for w in place["elim_week_true"].dropna().unique() if 1 <= w <= max(weeks)])
    for w in elim_weeks:
        ax.axvline(w, color="gray", linestyle="--", alpha=0.35, zorder=0)
    ax.set_xlim(weeks[0] - 0.3, weeks[-1] + 0.3)
    ax.set_ylim(0, None)
    ax.set_xlabel("Week", fontsize=10)
    ax.set_ylabel("Fan share", fontsize=10)
    ax.set_title("Top contestants", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)


def plot_side_by_side(
    season: int,
    sh: pd.DataFrame,
    place: pd.DataFrame,
    unc: Optional[pd.DataFrame],
    out_path: str,
):
    """One figure: stacked (left) + line (right). Concise titles, clean."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    fig.patch.set_facecolor("white")
    for ax in (ax0, ax1):
        ax.set_facecolor("#fafafa")
    _draw_stacked(ax0, season, sh, place)
    _draw_lines(ax1, season, sh, place, unc)
    fig.suptitle(f"Season {season}: Estimated fan vote share", fontsize=11, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_stacked_area(season: int, sh: pd.DataFrame, place: pd.DataFrame, out_path: str):
    """Standalone stacked area plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("white")
    _draw_stacked(ax, season, sh, place)
    ax.set_title(f"Season {season}: Estimated fan vote share (stacked)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_lines_with_uncertainty(
    season: int,
    sh: pd.DataFrame,
    place: pd.DataFrame,
    unc: Optional[pd.DataFrame],
    out_path: str,
    top_n: int = 6,
):
    """Standalone line plot (tab10 colors)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("white")
    _draw_lines(ax, season, sh, place, unc, top_n)
    ax.set_title(f"Season {season}: Estimated fan vote share (top contestants)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot vote shares for one season")
    ap.add_argument("--season", type=int, default=33, help="Season number")
    ap.add_argument("--shares", default=SHARES_CSV, help="Path to base_inferred_shares.csv")
    ap.add_argument("--uncertainty", default=UNCERTAINTY_CSV, help="Path to base_inferred_shares_uncertainty.csv")
    ap.add_argument("--placement", default=PLACEMENT_CSV, help="Path to base_placement_orders.csv")
    ap.add_argument("--out-dir", default=OUT_DIR, help="Output directory")
    ap.add_argument("--side-by-side", action="store_true", help="Also produce a single side-by-side figure")
    args = ap.parse_args()

    sh, place, unc = load_season_data(args.season, args.shares, args.uncertainty, args.placement)
    if sh.empty:
        print("No data for season", args.season)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    base_name = f"vote_shares_season_{args.season}"

    # Default: two separate figures (use these in LaTeX with two \\includegraphics for side-by-side)
    plot_stacked_area(
        args.season,
        sh,
        place,
        os.path.join(args.out_dir, f"{base_name}_stacked.png"),
    )
    plot_lines_with_uncertainty(
        args.season,
        sh,
        place,
        unc,
        os.path.join(args.out_dir, f"{base_name}_lines.png"),
    )
    if args.side_by_side:
        plot_side_by_side(
            args.season,
            sh,
            place,
            unc,
            os.path.join(args.out_dir, f"{base_name}_sidebyside.png"),
        )


if __name__ == "__main__":
    main()
