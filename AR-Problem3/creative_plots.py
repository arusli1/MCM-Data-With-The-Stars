#!/usr/bin/env python3
"""
Problem 3: Creative multi-panel plots — all the interesting relationships
between judge, fan, success, age, industry, partner. 4–6 panels per figure.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from eda import load_data

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "sans-serif"})


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3)


def plot_six_key_relationships(df):
    """6-panel: Judge vs Fan (color=age), corr matrix, Age→Judge, Age→Fan, Judge slope vs Fan slope, Pro partner judge vs fan."""
    sub = df.dropna(subset=["mean_judge_w1_3", "mean_fan_w1_3", "age", "judge_slope", "fan_slope"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # 1. Judge vs Fan scatter, color by age
    ax = axes[0, 0]
    sc = ax.scatter(sub["mean_judge_w1_3"], sub["mean_fan_w1_3"], c=sub["age"], cmap="viridis", alpha=0.7, s=36)
    # Diagonal = perfect rank agreement (min judge, min fan) to (max judge, max fan)
    j_min, j_max = sub["mean_judge_w1_3"].min(), sub["mean_judge_w1_3"].max()
    f_min, f_max = sub["mean_fan_w1_3"].min(), sub["mean_fan_w1_3"].max()
    ax.plot([j_min, j_max], [f_min, f_max], "k--", alpha=0.5, label="Rank agreement")
    r, p = pearsonr(sub["mean_judge_w1_3"], sub["mean_fan_w1_3"])
    ax.set_xlabel("Mean judge score (W1–3)")
    ax.set_ylabel("Mean fan share (W1–3)")
    ax.set_title(f"Judge vs Fan (color = age)\nr = {r:.3f}")
    plt.colorbar(sc, ax=ax, label="Age")
    _style_ax(ax)

    # 2. Correlation matrix: age, judge, fan, success
    ax = axes[0, 1]
    cols = ["age", "mean_judge_w1_3", "mean_fan_w1_3", "success_score"]
    labels = ["Age", "Judge", "Fan", "Success"]
    d = df[cols].dropna()
    corr = d.corr()
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="equal")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("Correlation matrix")
    plt.colorbar(im, ax=ax, label="r")

    # 3. Age vs Judge (scatter + trend + optional CI band)
    ax = axes[0, 2]
    s = df.dropna(subset=["age", "mean_judge_w1_3"])
    ax.scatter(s["age"], s["mean_judge_w1_3"], alpha=0.4, s=28, c="#2E86AB")
    z = np.polyfit(s["age"], s["mean_judge_w1_3"], 1)
    xl = np.linspace(s["age"].min(), s["age"].max(), 50)
    ax.plot(xl, np.poly1d(z)(xl), "k-", lw=2)
    r, p = pearsonr(s["age"], s["mean_judge_w1_3"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Mean judge score")
    ax.set_title(f"Age → Judge  r = {r:.3f}")
    _style_ax(ax)

    # 4. Age vs Fan
    ax = axes[1, 0]
    s = df.dropna(subset=["age", "mean_fan_w1_3"])
    ax.scatter(s["age"], s["mean_fan_w1_3"], alpha=0.4, s=28, c="#E94F37")
    z = np.polyfit(s["age"], s["mean_fan_w1_3"], 1)
    xl = np.linspace(s["age"].min(), s["age"].max(), 50)
    ax.plot(xl, np.poly1d(z)(xl), "k-", lw=2)
    r, p = pearsonr(s["age"], s["mean_fan_w1_3"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Mean fan share")
    ax.set_title(f"Age → Fan  r = {r:.3f}")
    _style_ax(ax)

    # 5. Judge slope vs Fan slope (improvement W1→W3)
    ax = axes[1, 1]
    ax.scatter(sub["judge_slope"], sub["fan_slope"], alpha=0.5, s=32, c="#44AF69")
    r, p = pearsonr(sub["judge_slope"], sub["fan_slope"])
    ax.axhline(0, color="gray", ls=":")
    ax.axvline(0, color="gray", ls=":")
    ax.set_xlabel("Judge improvement (W1→W3 slope)")
    ax.set_ylabel("Fan improvement (W1→W3 slope)")
    ax.set_title(f"Improvement: Judge vs Fan  r = {r:.3f}")
    _style_ax(ax)

    # 6. Pro partner: mean judge vs mean fan (one point per partner)
    by_partner = df.groupby("partner_b").agg(
        mean_judge=("mean_judge_w1_3", "mean"),
        mean_fan=("mean_fan_w1_3", "mean"),
        n=("season", "count"),
    ).reset_index()
    by_partner = by_partner[by_partner["n"] >= 5]
    ax = axes[1, 2]
    ax.scatter(by_partner["mean_judge"], by_partner["mean_fan"], s=by_partner["n"] * 4, alpha=0.7, c="#6C5CE7")
    for _, row in by_partner.iterrows():
        ax.annotate(row["partner_b"].split()[0][:6], (row["mean_judge"], row["mean_fan"]), fontsize=7, alpha=0.8)
    r, _ = pearsonr(by_partner["mean_judge"], by_partner["mean_fan"])
    ax.set_xlabel("Partner mean judge score")
    ax.set_ylabel("Partner mean fan share")
    ax.set_title(f"Pro partner: Judge vs Fan (n≥5)  r = {r:.3f}")
    _style_ax(ax)

    plt.suptitle("Key relationships: Judge, Fan, Success, Age, Partner", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGS / "creative_six_key_relationships.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "creative_six_key_relationships.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_six_by_subgroup(df):
    """6-panel: Judge vs Fan by industry (4 small scatter); Judge vs Fan by age bin; Outcomes by industry (3 bars); Outcomes by partner; Judge–fan r by subgroup; Success vs Judge & Fan."""
    sub = df.dropna(subset=["mean_judge_w1_3", "mean_fan_w1_3", "age", "industry_b"])
    sub = sub.copy()
    sub["age_bin"] = pd.cut(sub["age"], bins=[0, 30, 40, 50, 100], labels=["<30", "30–40", "40–50", "50+"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # 1. Judge vs Fan, colored by industry (top 4 industries)
    ax = axes[0, 0]
    top_ind = sub["industry_b"].value_counts().head(4).index.tolist()
    for ind in top_ind:
        s = sub[sub["industry_b"] == ind]
        ax.scatter(s["mean_judge_w1_3"], s["mean_fan_w1_3"], label=ind[:12], alpha=0.6, s=28)
    ax.set_xlabel("Judge score")
    ax.set_ylabel("Fan share")
    ax.set_title("Judge vs Fan by industry")
    ax.legend(fontsize=8)
    _style_ax(ax)

    # 2. Judge vs Fan by age bin
    ax = axes[0, 1]
    for ab in ["<30", "30–40", "40–50", "50+"]:
        s = sub[sub["age_bin"] == ab]
        if len(s) > 5:
            ax.scatter(s["mean_judge_w1_3"], s["mean_fan_w1_3"], label=ab, alpha=0.6, s=28)
    ax.set_xlabel("Judge score")
    ax.set_ylabel("Fan share")
    ax.set_title("Judge vs Fan by age bin")
    ax.legend(fontsize=8)
    _style_ax(ax)

    # 3. Mean outcome (Judge, Fan, Success) by industry — 3 grouped bars
    ax = axes[0, 2]
    top_ind = df["industry_b"].value_counts().head(6).index.tolist()
    sub2 = df[df["industry_b"].isin(top_ind)]
    by_ind = sub2.groupby("industry_b").agg(
        judge=("mean_judge_w1_3", "mean"),
        fan=("mean_fan_w1_3", "mean"),
        success=("success_score", "mean"),
    ).reindex(top_ind).dropna()
    x = np.arange(len(by_ind))
    w = 0.25  # bar width
    ax.bar(x - w, by_ind["judge"], w, label="Judge", color="#2E86AB")
    ax.bar(x, by_ind["fan"] * 30, w, label="Fan×30", color="#E94F37", alpha=0.9)
    ax.bar(x + w, by_ind["success"] * 10, w, label="Success×10", color="#44AF69", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([i[:8] for i in by_ind.index], rotation=45, ha="right")
    ax.set_ylabel("Mean (scaled)")
    ax.set_title("Outcomes by industry")
    ax.legend(fontsize=8)
    _style_ax(ax)

    # 4. Mean outcome by top partners (Judge, Fan, Success)
    ax = axes[1, 0]
    top_part = df["partner_b"].value_counts().head(8).index.tolist()
    sub2 = df[df["partner_b"].isin(top_part)]
    by_part = sub2.groupby("partner_b").agg(
        judge=("mean_judge_w1_3", "mean"),
        fan=("mean_fan_w1_3", "mean"),
        success=("success_score", "mean"),
    ).reindex(top_part).dropna()
    x = np.arange(len(by_part))
    ax.bar(x - w, by_part["judge"], w, label="Judge", color="#2E86AB")
    ax.bar(x, by_part["fan"] * 30, w, label="Fan×30", color="#E94F37", alpha=0.9)
    ax.bar(x + w, by_part["success"] * 10, w, label="Success×10", color="#44AF69", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([p[:6] for p in by_part.index], rotation=45, ha="right")
    ax.set_ylabel("Mean (scaled)")
    ax.set_title("Outcomes by pro partner")
    ax.legend(fontsize=8)
    _style_ax(ax)

    # 5. Judge–fan agreement (r) by subgroup
    ax = axes[1, 1]
    rows = []
    for name, grp in sub.groupby("industry_b"):
        if len(grp) >= 15:
            r_val, _ = pearsonr(grp["mean_judge_w1_3"], grp["mean_fan_w1_3"])
            rows.append({"group": str(name)[:14], "r": r_val})
    for name, grp in sub.groupby("age_bin", observed=True):
        if len(grp) >= 15 and pd.notna(name):
            r_val, _ = pearsonr(grp["mean_judge_w1_3"], grp["mean_fan_w1_3"])
            rows.append({"group": f"Age {name}", "r": r_val})
    if rows:
        rdf = pd.DataFrame(rows).sort_values("r")
        colors = ["#2E86AB" if r > 0 else "#E94F37" for r in rdf["r"]]
        ax.barh(rdf["group"], rdf["r"], color=colors, alpha=0.8)
        ax.axvline(0, color="black", ls="--")
        ax.set_xlabel("Corr(Judge, Fan)")
        ax.set_title("Judge–fan agreement by subgroup")
    _style_ax(ax)

    # 6. Success score vs Judge (and vs Fan) — scatter or binned
    ax = axes[1, 2]
    s = df.dropna(subset=["success_score", "mean_judge_w1_3", "mean_fan_w1_3"])
    ax.scatter(s["mean_judge_w1_3"], s["success_score"], alpha=0.4, s=30, c="#2E86AB", label="Judge")
    ax.scatter(s["mean_fan_w1_3"] * 30, s["success_score"], alpha=0.4, s=30, c="#E94F37", label="Fan×30")
    ax.set_xlabel("Judge score / Fan share×30")
    ax.set_ylabel("Success score")
    ax.set_title("Success vs Judge & Fan")
    ax.legend(fontsize=8)
    _style_ax(ax)

    plt.suptitle("By subgroup: industry, age, partner", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGS / "creative_six_by_subgroup.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "creative_six_by_subgroup.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_four_trajectories_placement(df):
    """4-panel: Age vs Judge with CI band; Age vs Fan with CI band; Placement vs Judge; Placement vs Fan."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    sub = df.dropna(subset=["age", "mean_judge_w1_3"])
    sub = sub.sort_values("age")
    ax = axes[0, 0]
    ax.scatter(sub["age"], sub["mean_judge_w1_3"], alpha=0.35, s=25, c="#2E86AB")
    # Local trend with simple bin means ± sem
    bins = np.arange(18, 85, 5)
    sub["age_bin"] = np.digitize(sub["age"], bins)
    by_bin = sub.groupby("age_bin").agg(
        age_mid=("age", "mean"),
        mean=("mean_judge_w1_3", "mean"),
        sem=("mean_judge_w1_3", "sem"),
        n=("age", "count"),
    ).reset_index()
    by_bin = by_bin[by_bin["n"] >= 5]
    ax.plot(by_bin["age_mid"], by_bin["mean"], "k-", lw=2)
    ax.fill_between(by_bin["age_mid"], by_bin["mean"] - by_bin["sem"], by_bin["mean"] + by_bin["sem"], alpha=0.2, color="gray")
    ax.set_xlabel("Age")
    ax.set_ylabel("Mean judge score")
    ax.set_title("Age → Judge (mean ± SEM by age bin)")
    _style_ax(ax)

    sub = df.dropna(subset=["age", "mean_fan_w1_3"])
    sub = sub.sort_values("age")
    ax = axes[0, 1]
    ax.scatter(sub["age"], sub["mean_fan_w1_3"], alpha=0.35, s=25, c="#E94F37")
    sub["age_bin"] = np.digitize(sub["age"], bins)
    by_bin = sub.groupby("age_bin").agg(
        age_mid=("age", "mean"),
        mean=("mean_fan_w1_3", "mean"),
        sem=("mean_fan_w1_3", "sem"),
        n=("age", "count"),
    ).reset_index()
    by_bin = by_bin[by_bin["n"] >= 5]
    ax.plot(by_bin["age_mid"], by_bin["mean"], "k-", lw=2)
    ax.fill_between(by_bin["age_mid"], by_bin["mean"] - by_bin["sem"], by_bin["mean"] + by_bin["sem"], alpha=0.2, color="gray")
    ax.set_xlabel("Age")
    ax.set_ylabel("Mean fan share")
    ax.set_title("Age → Fan (mean ± SEM by age bin)")
    _style_ax(ax)

    sub = df.dropna(subset=["placement", "mean_judge_w1_3"])
    ax = axes[1, 0]
    ax.scatter(sub["placement"], sub["mean_judge_w1_3"], alpha=0.4, s=28, c="#2E86AB")
    r, p = pearsonr(sub["placement"], sub["mean_judge_w1_3"])
    ax.set_xlabel("Final placement (1 = winner)")
    ax.set_ylabel("Mean judge score (W1–3)")
    ax.set_title(f"Placement vs Judge  r = {r:.3f}")
    _style_ax(ax)

    sub = df.dropna(subset=["placement", "mean_fan_w1_3"])
    ax = axes[1, 1]
    ax.scatter(sub["placement"], sub["mean_fan_w1_3"], alpha=0.4, s=28, c="#E94F37")
    r, p = pearsonr(sub["placement"], sub["mean_fan_w1_3"])
    ax.set_xlabel("Final placement (1 = winner)")
    ax.set_ylabel("Mean fan share (W1–3)")
    ax.set_title(f"Placement vs Fan  r = {r:.3f}")
    _style_ax(ax)

    plt.suptitle("Trajectories (age) and final placement", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGS / "creative_four_trajectories_placement.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "creative_four_trajectories_placement.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_five_correlations_compact(df):
    """5-panel compact: Corr matrix (age, judge, fan, success, placement); Age–Judge; Age–Fan; Judge–Fan; Success vs Judge & Fan (dual)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # 0. Correlation matrix (5 vars)
    ax = axes[0]
    cols = ["age", "mean_judge_w1_3", "mean_fan_w1_3", "success_score", "placement"]
    labels = ["Age", "Judge", "Fan", "Success", "Place"]
    d = df[cols].dropna()
    corr = d.corr()
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="equal")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title("Correlation matrix")
    plt.colorbar(im, ax=ax, label="r")

    # 1. Age vs Judge
    ax = axes[1]
    s = df.dropna(subset=["age", "mean_judge_w1_3"])
    ax.scatter(s["age"], s["mean_judge_w1_3"], alpha=0.4, s=22, c="#2E86AB")
    z = np.polyfit(s["age"], s["mean_judge_w1_3"], 1)
    xl = np.linspace(s["age"].min(), s["age"].max(), 50)
    ax.plot(xl, np.poly1d(z)(xl), "k-", lw=1.5)
    r, _ = pearsonr(s["age"], s["mean_judge_w1_3"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Judge")
    ax.set_title(f"Age → Judge  r = {r:.3f}")
    _style_ax(ax)

    # 2. Age vs Fan
    ax = axes[2]
    s = df.dropna(subset=["age", "mean_fan_w1_3"])
    ax.scatter(s["age"], s["mean_fan_w1_3"], alpha=0.4, s=22, c="#E94F37")
    z = np.polyfit(s["age"], s["mean_fan_w1_3"], 1)
    xl = np.linspace(s["age"].min(), s["age"].max(), 50)
    ax.plot(xl, np.poly1d(z)(xl), "k-", lw=1.5)
    r, _ = pearsonr(s["age"], s["mean_fan_w1_3"])
    ax.set_xlabel("Age")
    ax.set_ylabel("Fan")
    ax.set_title(f"Age → Fan  r = {r:.3f}")
    _style_ax(ax)

    # 3. Judge vs Fan
    ax = axes[3]
    s = df.dropna(subset=["mean_judge_w1_3", "mean_fan_w1_3"])
    ax.scatter(s["mean_judge_w1_3"], s["mean_fan_w1_3"], alpha=0.4, s=22, c="purple")
    r, _ = pearsonr(s["mean_judge_w1_3"], s["mean_fan_w1_3"])
    ax.set_xlabel("Judge")
    ax.set_ylabel("Fan")
    ax.set_title(f"Judge vs Fan  r = {r:.3f}")
    _style_ax(ax)

    # 4. Success vs Judge (scatter) and Success vs Fan (overlay scaled)
    ax = axes[4]
    s = df.dropna(subset=["success_score", "mean_judge_w1_3", "mean_fan_w1_3"])
    ax.scatter(s["mean_judge_w1_3"], s["success_score"], alpha=0.4, s=22, c="#2E86AB", label="Judge")
    ax.scatter(s["mean_fan_w1_3"] * 25, s["success_score"], alpha=0.4, s=22, c="#E94F37", label="Fan×25")
    ax.set_xlabel("Judge score / Fan×25")
    ax.set_ylabel("Success score")
    ax.set_title("Success vs Judge & Fan")
    ax.legend(fontsize=8)
    _style_ax(ax)

    # 5. Placement vs Judge and vs Fan (dual scatter)
    ax = axes[5]
    s = df.dropna(subset=["placement", "mean_judge_w1_3", "mean_fan_w1_3"])
    ax.scatter(s["placement"], s["mean_judge_w1_3"], alpha=0.4, s=22, c="#2E86AB", label="Judge")
    ax2 = ax.twinx()
    ax2.scatter(s["placement"], s["mean_fan_w1_3"], alpha=0.4, s=22, c="#E94F37", label="Fan")
    ax.set_xlabel("Placement (1=winner)")
    ax.set_ylabel("Judge score", color="#2E86AB")
    ax2.set_ylabel("Fan share", color="#E94F37")
    ax.set_title("Placement vs Judge & Fan")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    _style_ax(ax)
    _style_ax(ax2)

    plt.suptitle("Correlations: age, judge, fan, success, placement", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(FIGS / "creative_five_correlations_compact.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "creative_five_correlations_compact.png", dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print("Loading data...")
    df = load_data()
    print("Creating creative 6-panel: key relationships...")
    plot_six_key_relationships(df)
    print("Creating creative 6-panel: by subgroup...")
    plot_six_by_subgroup(df)
    print("Creating creative 4-panel: trajectories & placement...")
    plot_four_trajectories_placement(df)
    print("Creating creative 5-panel: correlations compact...")
    plot_five_correlations_compact(df)
    print(f"Done. Figures in {FIGS}/")


if __name__ == "__main__":
    main()
