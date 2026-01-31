"""
Viewership vs. controversy analysis for Problem 2c.
Uses only Wikipedia-verified viewership (seasons 12, 18, 19, 31 excluded).
Correlates mean viewers with continuous controversy metrics:
  - mean_controversy: mean |judge_percentile - placement_percentile| per contestant per season
  - max_controversy: max discrepancy in that season
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Data"
OUT = ROOT / "outputs"
FIGS = ROOT / "all-paper-info"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def season_regime(season: int) -> str:
    """Scoring regime: rank (s1-2), percent (s3-27), rank_bottom2 (s28+)."""
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "rank_bottom2"


def main():
    # Load viewership — Wikipedia-verified only (no fallback seasons)
    view = pd.read_csv(DATA / "dwts_viewership.csv")
    view = view[view["mean_viewers"].notna()].copy()
    excluded = set(range(1, 35)) - set(view["season"].astype(int))
    if excluded:
        print(f"Excluded (no Wikipedia ratings): seasons {sorted(excluded)}")

    # Load controversy: continuous metric = mean |judge_percentile - placement_percentile| per contestant per season
    contr = pd.read_csv(ROOT / "outputs" / "problem2b_controversy_list.csv")
    contr_agg = contr.groupby("season").agg(
        mean_controversy=("controversy_score", "mean"),
        max_controversy=("controversy_score", "max"),
        std_controversy=("controversy_score", "std"),
    ).reset_index()

    df = view.merge(contr_agg, on="season", how="left")
    df["regime"] = df["season"].apply(season_regime)

    # Correlations: mean_controversy vs viewership
    valid = df["mean_controversy"].notna()
    r_pearson, p_pearson = stats.pearsonr(df.loc[valid, "mean_controversy"], df.loc[valid, "mean_viewers"])
    r_spearman, p_spearman = stats.spearmanr(df.loc[valid, "mean_controversy"], df.loc[valid, "mean_viewers"])

    # Partial correlation: controversy vs viewership controlling for season
    def partial_corr(x, y, z):
        r_xy, _ = stats.pearsonr(x, y)
        r_xz, _ = stats.pearsonr(x, z)
        r_yz, _ = stats.pearsonr(y, z)
        r = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2) + 1e-10)
        n = len(x)
        t = r * np.sqrt(max(0, n - 3)) / np.sqrt(1 - r**2 + 1e-10)
        p = 2 * (1 - stats.t.cdf(abs(t), max(1, n - 3)))
        return r, p
    r_partial, p_partial = partial_corr(
        df.loc[valid, "mean_controversy"].values,
        df.loc[valid, "mean_viewers"].values,
        df.loc[valid, "season"].values,
    )

    # Residual viewership
    coef = np.polyfit(df["season"], df["mean_viewers"], 1)
    df["viewers_resid"] = df["mean_viewers"] - np.polyval(coef, df["season"])
    r_resid, p_resid = stats.pearsonr(df.loc[valid, "mean_controversy"], df.loc[valid, "viewers_resid"])

    # --- Plot 1: Scatter mean_controversy vs viewership by regime ---
    fig, ax = plt.subplots(figsize=(8, 5))
    reg_colors = {"rank": "#E64B35", "percent": "#4DBBD5", "rank_bottom2": "#00A087"}
    for regime in ["rank", "percent", "rank_bottom2"]:
        sub = df[(df["regime"] == regime) & valid]
        if len(sub) > 0:
            ax.scatter(sub["mean_controversy"], sub["mean_viewers"], s=70, alpha=0.85,
                       c=reg_colors[regime], label=regime.replace("_", " "), edgecolors="white", linewidth=1)
    for _, row in df[valid].iterrows():
        ax.annotate(f"S{int(row['season'])}", (row["mean_controversy"], row["mean_viewers"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    z = np.polyfit(df.loc[valid, "mean_controversy"], df.loc[valid, "mean_viewers"], 1)
    xl = np.linspace(df["mean_controversy"].min(), df["mean_controversy"].max(), 50)
    ax.plot(xl, np.poly1d(z)(xl), "k--", alpha=0.6, label="trend")
    ax.set_xlabel("Mean controversy (|judge % − placement %| per contestant)", fontsize=11)
    ax.set_ylabel("Mean viewership (millions)", fontsize=11)
    ax.set_title("Viewership vs. mean controversy by season\n"
                 "Pearson r = {:.3f} (p = {:.3f}); partial r (controlling season) = {:.3f} (p = {:.3f})".format(
                     r_pearson, p_pearson, r_partial, p_partial))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS / "viewership_controversy_scatter.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "viewership_controversy_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: Viewership and mean controversy over time ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(df["season"] - 0.2, df["mean_viewers"], width=0.4, label="Mean viewership (M)", color="#0173B2", alpha=0.8)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(df["season"], df["mean_controversy"], "o-", color="#DE8F05", label="Mean controversy", linewidth=2, markersize=6)
    ax2.set_xlabel("Season", fontsize=11)
    ax2.set_ylabel("Mean viewership (millions)", fontsize=11, color="#0173B2")
    ax2_twin.set_ylabel("Mean controversy (|judge % − placement %|)", fontsize=11, color="#DE8F05")
    ax2.set_title("Viewership and controversy intensity by season (Wikipedia-verified only)")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(FIGS / "viewership_controversy_by_season.pdf", dpi=150, bbox_inches="tight")
    fig2.savefig(FIGS / "viewership_controversy_by_season.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 3: Residual viewership vs mean controversy ---
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.scatter(df.loc[valid, "mean_controversy"], df.loc[valid, "viewers_resid"], s=70, c="#0173B2", alpha=0.8, edgecolors="white", linewidth=1)
    for _, row in df[valid].iterrows():
        ax3.annotate(f"S{int(row['season'])}", (row["mean_controversy"], row["viewers_resid"]),
                     fontsize=7, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Mean controversy (|judge % − placement %|)", fontsize=11)
    ax3.set_ylabel("Viewership residual (after detrending by season)", fontsize=11)
    ax3.set_title("Controversy intensity vs. viewership deviation from trend\n(r = {:.3f}, p = {:.3f})".format(
        r_resid, p_resid))
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(FIGS / "viewership_controversy_residual.pdf", dpi=150, bbox_inches="tight")
    fig3.savefig(FIGS / "viewership_controversy_residual.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 4: Max controversy vs viewership (alternative metric) ---
    r_max, p_max = stats.pearsonr(df.loc[valid, "max_controversy"], df.loc[valid, "mean_viewers"])
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ax4.scatter(df.loc[valid, "max_controversy"], df.loc[valid, "mean_viewers"], s=70, c="#0173B2", alpha=0.8, edgecolors="white", linewidth=1)
    for _, row in df[valid].iterrows():
        ax4.annotate(f"S{int(row['season'])}", (row["max_controversy"], row["mean_viewers"]),
                     fontsize=7, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
    z4 = np.polyfit(df.loc[valid, "max_controversy"], df.loc[valid, "mean_viewers"], 1)
    xl4 = np.linspace(df["max_controversy"].min(), df["max_controversy"].max(), 50)
    ax4.plot(xl4, np.poly1d(z4)(xl4), "k--", alpha=0.6)
    ax4.set_xlabel("Max controversy in season (|judge % − placement %|)", fontsize=11)
    ax4.set_ylabel("Mean viewership (millions)", fontsize=11)
    ax4.set_title("Viewership vs. max controversy\n(r = {:.3f}, p = {:.3f})".format(r_max, p_max))
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    fig4.savefig(FIGS / "viewership_controversy_max.pdf", dpi=150, bbox_inches="tight")
    fig4.savefig(FIGS / "viewership_controversy_max.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary
    summary = [
        "## Viewership vs. Controversy Analysis",
        "",
        "**Data:** Wikipedia viewership, Problem 2b controversy. **Wikipedia-verified seasons only.**",
        "",
        "**Controversy metric:** Mean |judge_percentile − placement_percentile| per contestant per season (continuous).",
        "",
        f"**Seasons:** {len(df)}",
        f"**Pearson r (mean controversy vs viewership):** {r_pearson:.3f} (p = {p_pearson:.4f})",
        f"**Spearman ρ:** {r_spearman:.3f} (p = {p_spearman:.4f})",
        f"**Partial r (controlling season):** {r_partial:.3f} (p = {p_partial:.4f})",
        f"**Residual correlation:** {r_resid:.3f} (p = {p_resid:.4f})",
        f"**Max controversy vs viewership:** r = {r_max:.3f} (p = {p_max:.4f})",
        "",
        "**Interpretation:** Raw correlation is positive (r ≈ 0.33) but confounded by time: early seasons "
        "had both higher viewership and different controversy levels. **Partial correlation (controlling season) "
        "is strongly negative (r ≈ {:.2f}, p ≈ {:.3f})**: within-era, seasons with higher mean controversy "
        "tended to have *lower* viewership than expected.".format(r_partial, p_partial),
        "",
        "**Limitations:** Viewership declines over time; small N.",
        "",
        "**Figures:** viewership_controversy_scatter.pdf, viewership_controversy_by_season.pdf, viewership_controversy_residual.pdf, viewership_controversy_max.pdf",
    ]
    with open(OUT / "viewership_controversy_summary.md", "w") as f:
        f.write("\n".join(summary))

    df[["season", "mean_viewers", "mean_controversy", "max_controversy", "std_controversy", "viewers_resid", "regime"]].to_csv(
        OUT / "viewership_controversy_season_metrics.csv", index=False
    )

    # --- Paper-ready figures (publication quality) ---
    _make_paper_figures(df, valid, r_resid, p_resid, r_partial, p_partial)

    print("\n".join(summary))
    print(f"\nSaved: {FIGS}/viewership_controversy_*.pdf, {FIGS}/paper_viewership_*.pdf")


def _make_paper_figures(df, valid, r_resid, p_resid, r_partial, p_partial):
    """Generate polished 2-panel figure for paper."""
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11})
    notable = {11: "Bristol Palin", 27: "Bobby Bones", 2: "Jerry Rice", 4: "Billy Ray Cyrus"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    c_view = "#2E86AB"
    c_cont = "#E94F37"

    # Panel A: Viewership + mean controversy over season
    ax1.fill_between(df["season"], df["mean_viewers"], alpha=0.35, color=c_view)
    ax1.plot(df["season"], df["mean_viewers"], "o-", color=c_view, linewidth=2, markersize=5, label="Viewership (M)")
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Mean viewership (millions)", color=c_view)
    ax1.tick_params(axis="y", labelcolor=c_view)
    ax1.set_xlim(0.5, 34.5)
    ax1.set_ylim(0, None)

    ax1b = ax1.twinx()
    ax1b.plot(df["season"], df["mean_controversy"], "s--", color=c_cont, linewidth=1.5, markersize=4, alpha=0.9, label="Mean controversy")
    ax1b.set_ylabel("Mean controversy (|judge % − placement %|)", color=c_cont)
    ax1b.tick_params(axis="y", labelcolor=c_cont)
    ax1b.set_ylim(0, 0.35)
    ax1.set_title("(a) Viewership and controversy intensity over time")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.25)

    # Panel B: Residual scatter (within-era relationship)
    ax2.scatter(df.loc[valid, "mean_controversy"], df.loc[valid, "viewers_resid"],
                s=80, c="#2E86AB", alpha=0.85, edgecolors="white", linewidth=1.5, zorder=3)
    for _, row in df[valid].iterrows():
        s = int(row["season"])
        lbl = f"S{s}" + (f" ({notable[s]})" if s in notable else "")
        ax2.annotate(lbl, (row["mean_controversy"], row["viewers_resid"]),
                     fontsize=7, ha="center", va="bottom" if row["viewers_resid"] >= 0 else "top",
                     xytext=(0, 6), textcoords="offset points")
    z = np.polyfit(df.loc[valid, "mean_controversy"], df.loc[valid, "viewers_resid"], 1)
    xl = np.linspace(df["mean_controversy"].min(), df["mean_controversy"].max(), 50)
    ax2.plot(xl, np.poly1d(z)(xl), "k-", linewidth=2, alpha=0.7, label=f"Trend (r = {r_resid:.2f}, p = {p_resid:.3f})")
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.6)
    ax2.set_xlabel("Mean controversy (|judge % − placement %| per contestant)")
    ax2.set_ylabel("Viewership residual (vs. season trend)")
    ax2.set_title("(b) Within-era: controversy vs. viewership deviation")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.set_ylim(-2.5, 4.2)

    plt.tight_layout()
    fig.savefig(FIGS / "paper_viewership_controversy.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "paper_viewership_controversy.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Single-panel alternative: scatter only (if space is tight)
    fig2, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(df.loc[valid, "mean_controversy"], df.loc[valid, "mean_viewers"],
               s=90, c="#2E86AB", alpha=0.85, edgecolors="white", linewidth=1.5)
    for _, row in df[valid].iterrows():
        s = int(row["season"])
        lbl = f"S{s}" + (f" ({notable[s]})" if s in notable else "")
        ax.annotate(lbl, (row["mean_controversy"], row["mean_viewers"]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5), textcoords="offset points")
    z2 = np.polyfit(df.loc[valid, "mean_controversy"], df.loc[valid, "mean_viewers"], 1)
    xl2 = np.linspace(df["mean_controversy"].min(), df["mean_controversy"].max(), 50)
    ax.plot(xl2, np.poly1d(z2)(xl2), "k--", linewidth=1.5, alpha=0.6)
    ax.set_xlabel("Mean controversy (|judge % − placement %| per contestant)")
    ax.set_ylabel("Mean viewership (millions)")
    ax.set_title("Viewership vs. controversy by season\n(partial r = {:.2f}, p = {:.3f} controlling season)".format(r_partial, p_partial))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(FIGS / "paper_viewership_controversy_scatter.pdf", dpi=300, bbox_inches="tight")
    fig2.savefig(FIGS / "paper_viewership_controversy_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
