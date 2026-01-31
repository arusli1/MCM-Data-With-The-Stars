"""
Viewership vs. controversy analysis for Problem 2c.
Correlates Wikipedia viewership (mean viewers per season) with controversy count.
Outputs: scatter plot, correlation stats, summary.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Data"
OUT = ROOT / "outputs"
FIGS = ROOT / "all-paper-info"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    # Load viewership
    view = pd.read_csv(DATA / "dwts_viewership.csv")
    view = view[view["mean_viewers"].notna()].copy()

    # Load controversy count per season
    contr = pd.read_csv(ROOT / "outputs" / "problem2b_controversy_classified.csv")
    contr_counts = contr[contr["controversial"]].groupby("season").size().reset_index(name="n_controversial")

    # Merge (all seasons, fill 0 for no controversy)
    seasons = view["season"].unique()
    df = view.merge(contr_counts, on="season", how="left")
    df["n_controversial"] = df["n_controversial"].fillna(0).astype(int)

    # Correlation: controversy count vs mean viewers
    r_pearson, p_pearson = np.corrcoef(df["n_controversial"], df["mean_viewers"])[0, 1], None
    from scipy import stats
    r_pearson, p_pearson = stats.pearsonr(df["n_controversial"], df["mean_viewers"])
    r_spearman, p_spearman = stats.spearmanr(df["n_controversial"], df["mean_viewers"])

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["n_controversial"], df["mean_viewers"], s=80, alpha=0.8, c="#0173B2", edgecolors="white", linewidth=1.5)
    for _, row in df[df["n_controversial"] > 0].iterrows():
        ax.annotate(f"S{int(row['season'])}", (row["n_controversial"], row["mean_viewers"]),
                    fontsize=8, ha="left", va="bottom", xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Number of controversial contestants (GMM)", fontsize=11)
    ax.set_ylabel("Mean viewership (millions)", fontsize=11)
    ax.set_title("Viewership vs. controversy by season\n(Pearson r = {:.3f}, p = {:.3f}; Spearman ρ = {:.3f}, p = {:.3f})".format(
        r_pearson, p_pearson, r_spearman, p_spearman), fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS / "viewership_controversy_scatter.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "viewership_controversy_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Alternative: viewership over time, annotate controversy seasons
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(df["season"] - 0.2, df["mean_viewers"], width=0.4, label="Mean viewership (M)", color="#0173B2", alpha=0.8)
    ax2_twin = ax2.twinx()
    ax2_twin.bar(df["season"] + 0.2, df["n_controversial"], width=0.4, label="Controversial count", color="#DE8F05", alpha=0.8)
    ax2.set_xlabel("Season", fontsize=11)
    ax2.set_ylabel("Mean viewership (millions)", fontsize=11, color="#0173B2")
    ax2_twin.set_ylabel("Controversial contestants", fontsize=11, color="#DE8F05")
    ax2.set_title("Viewership and controversy by season")
    ax2.legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(FIGS / "viewership_controversy_by_season.pdf", dpi=150, bbox_inches="tight")
    fig2.savefig(FIGS / "viewership_controversy_by_season.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary
    summary = [
        "## Viewership vs. Controversy Analysis",
        "",
        "**Data sources:** Wikipedia (viewership per episode), Problem 2b (GMM controversial count).",
        "",
        f"**Seasons:** {len(df)}",
        f"**Pearson r:** {r_pearson:.3f} (p = {p_pearson:.4f})",
        f"**Spearman ρ:** {r_spearman:.3f} (p = {p_spearman:.4f})",
        "",
        "**Interpretation:** " + (
            "Positive correlation: seasons with more controversial contestants tend to have higher viewership."
            if r_pearson > 0.1 else
            "No strong linear correlation: controversy count does not clearly predict viewership."
            if abs(r_pearson) < 0.2 else
            "Negative correlation: seasons with more controversy had lower viewership."
        ),
        "",
        "**Limitations:** Viewership declines over time (cord-cutting); controversy is rare (21 total).",
        "Season-level confounding (year, competition) limits causal inference.",
        "",
        "**Figures:** viewership_controversy_scatter.pdf, viewership_controversy_by_season.pdf",
    ]
    with open(OUT / "viewership_controversy_summary.md", "w") as f:
        f.write("\n".join(summary))

    print("\n".join(summary))
    print(f"\nSaved: {FIGS / 'viewership_controversy_scatter.pdf'}, {FIGS / 'viewership_controversy_by_season.pdf'}")


if __name__ == "__main__":
    main()
