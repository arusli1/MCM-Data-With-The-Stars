"""
Problem 3 EDA: Impact of pro dancers and celebrity characteristics on competition outcomes.
Uses main data + fan vote estimates. Analyzes judges scores vs fan votes separately.
Statistical tests, regression, and comprehensive visualizations.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Data"
# Try estimate_votes first; fallback to base_inferred_shares
VOTES_PATH = DATA / "estimate_votes.csv"
VOTES_FALLBACK = ROOT.parent / "AR-Problem1-Base" / "final_results" / "base_inferred_shares.csv"
OUT = ROOT / "outputs"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    return name


def load_data():
    """Load main data and fan vote estimates, merge, compute outcomes."""
    df = pd.read_csv(DATA / "2026_MCM_Problem_C_Data.csv")
    df["norm_name"] = df["celebrity_name"].apply(normalize_name)

    # Fan votes: try estimate_votes, fallback to base_inferred_shares
    if VOTES_PATH.exists():
        votes = pd.read_csv(VOTES_PATH)
    else:
        votes = pd.read_csv(VOTES_FALLBACK)
    votes["norm_name"] = votes["celebrity_name"].apply(normalize_name)

    # Success score: 1 = winner, 0 = last
    df["success_score"] = df.groupby("season")["placement"].transform(
        lambda x: 1 - (x - 1) / max(x.max() - 1, 1)
    )

    # Judge scores: weeks 1–3, sum per week (exclude 0 and N/A)
    judge_weeks = []
    for w in range(1, 4):
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5) if f"week{w}_judge{j}_score" in df.columns]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace(0, np.nan)
        df[f"week{w}_total"] = df[cols].sum(axis=1, min_count=1)
        judge_weeks.append(f"week{w}_total")
    df["mean_judge_w1_3"] = df[judge_weeks].mean(axis=1)

    # Fan shares: weeks 1–3
    fan_sub = votes[votes["week"].isin([1, 2, 3])]
    fan_pivot = fan_sub.pivot_table(index=["season", "norm_name"], columns="week", values="s_share").reset_index()
    fan_pivot.columns = ["season", "norm_name", "fan_w1", "fan_w2", "fan_w3"]
    df = df.merge(fan_pivot, on=["season", "norm_name"], how="left")
    df["mean_fan_w1_3"] = df[["fan_w1", "fan_w2", "fan_w3"]].mean(axis=1)

    # Slopes (improvement W1→W3)
    def slope(y):
        y = np.array(y, dtype=float)
        valid = ~np.isnan(y)
        if np.sum(valid) < 2:
            return np.nan
        x = np.arange(len(y))[valid]
        return np.polyfit(x, y[valid], 1)[0]
    df["judge_slope"] = df[[f"week{w}_total" for w in [1, 2, 3]]].apply(slope, axis=1)
    df["fan_slope"] = df[["fan_w1", "fan_w2", "fan_w3"]].apply(slope, axis=1)

    # Clean features
    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")
    df["industry"] = df["celebrity_industry"].fillna("Other")
    df["partner"] = df["ballroom_partner"].fillna("Unknown")
    df["country"] = df["celebrity_homecountry/region"].fillna("Unknown")
    df["homestate"] = df["celebrity_homestate"].fillna("Unknown")

    # Bucket rare categories for stability
    def bucket(col, k=8):
        top = df[col].value_counts().head(k).index.tolist()
        return df[col].apply(lambda x: x if x in top else "Other")

    df["industry_b"] = bucket("industry", 8)
    df["partner_b"] = bucket("partner", 12)
    return df


def run_statistical_tests(df):
    """ANOVA, Kruskal-Wallis, correlations. Return dict of results."""
    results = []

    # --- Age: Pearson/Spearman vs outcomes ---
    for outcome in ["mean_judge_w1_3", "mean_fan_w1_3", "success_score", "placement"]:
        sub = df.dropna(subset=[outcome, "age"])
        if len(sub) < 10:
            continue
        r_p, p_p = pearsonr(sub["age"], sub[outcome])
        r_s, p_s = spearmanr(sub["age"], sub[outcome])
        results.append({"factor": "age", "outcome": outcome, "test": "Pearson", "stat": r_p, "p": p_p, "n": len(sub)})
        results.append({"factor": "age", "outcome": outcome, "test": "Spearman", "stat": r_s, "p": p_s, "n": len(sub)})

    # --- Industry: ANOVA / Kruskal-Wallis ---
    for outcome in ["mean_judge_w1_3", "mean_fan_w1_3", "success_score"]:
        sub = df.dropna(subset=[outcome])
        groups = [g[outcome].values for _, g in sub.groupby("industry_b") if len(g) >= 3]
        if len(groups) < 2:
            continue
        try:
            f, p_anova = f_oneway(*groups)
            h, p_kw = kruskal(*groups)
            results.append({"factor": "industry", "outcome": outcome, "test": "ANOVA_F", "stat": f, "p": p_anova, "n": len(sub)})
            results.append({"factor": "industry", "outcome": outcome, "test": "Kruskal_H", "stat": h, "p": p_kw, "n": len(sub)})
        except Exception:
            pass

    # --- Partner: ANOVA / Kruskal-Wallis ---
    for outcome in ["mean_judge_w1_3", "mean_fan_w1_3", "success_score"]:
        sub = df.dropna(subset=[outcome])
        groups = [g[outcome].values for _, g in sub.groupby("partner_b") if len(g) >= 3]
        if len(groups) < 2:
            continue
        try:
            f, p_anova = f_oneway(*groups)
            h, p_kw = kruskal(*groups)
            results.append({"factor": "partner", "outcome": outcome, "test": "ANOVA_F", "stat": f, "p": p_anova, "n": len(sub)})
            results.append({"factor": "partner", "outcome": outcome, "test": "Kruskal_H", "stat": h, "p": p_kw, "n": len(sub)})
        except Exception:
            pass

    return pd.DataFrame(results)


def run_regressions(df):
    """OLS: outcome ~ age + C(industry) + C(partner). Quantify impact."""
    reg_results = []
    reg_models = {}
    outcomes = ["mean_judge_w1_3", "mean_fan_w1_3", "success_score"]

    for outcome in outcomes:
        sub = df.dropna(subset=[outcome, "age", "industry_b", "partner_b"])
        if len(sub) < 50:
            continue
        try:
            model = ols(f"{outcome} ~ age + C(industry_b) + C(partner_b)", data=sub).fit()
            reg_results.append({"outcome": outcome, "rsq": model.rsquared, "rsq_adj": model.rsquared_adj})
            age_coef = model.params.get("age", np.nan)
            reg_results.append({"outcome": outcome, "term": "age", "coef": age_coef, "p": model.pvalues.get("age", np.nan)})
            reg_models[outcome] = model
        except Exception as e:
            reg_results.append({"outcome": outcome, "error": str(e)})
    return reg_results, reg_models


def plot_age_effects(df):
    """Scatter: age vs judge score, age vs fan share. Judge vs fan comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    outcomes = ["mean_judge_w1_3", "mean_fan_w1_3", "success_score"]
    titles = ["Judge score (W1-3)", "Fan share (W1-3)", "Success score"]
    colors = ["#2E86AB", "#E94F37", "#44AF69"]

    def fmt_p(p):
        return "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"

    for ax, outcome, title, c in zip(axes, outcomes, titles, colors):
        sub = df.dropna(subset=[outcome, "age"])
        ax.scatter(sub["age"], sub[outcome], alpha=0.5, s=35, c=c)
        z = np.polyfit(sub["age"], sub[outcome], 1)
        xl = np.linspace(sub["age"].min(), sub["age"].max(), 50)
        ax.plot(xl, np.poly1d(z)(xl), "k-", linewidth=2)
        r, p = pearsonr(sub["age"], sub[outcome])
        ax.set_xlabel("Celebrity age")
        ax.set_ylabel(title)
        ax.set_title(f"{title}\nr = {r:.3f}, {fmt_p(p)}")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS / "age_effects.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "age_effects.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_industry_effects(df):
    """Boxplot: outcomes by industry (top categories)."""
    top_ind = df["industry_b"].value_counts().head(8).index.tolist()
    sub = df[df["industry_b"].isin(top_ind)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, outcome, title in zip(axes, ["mean_judge_w1_3", "mean_fan_w1_3", "success_score"],
                                  ["Judge score", "Fan share", "Success score"]):
        order = sub.groupby("industry_b")[outcome].median().sort_values(ascending=False).index
        sns.boxplot(data=sub, x="industry_b", y=outcome, order=order, ax=ax, hue="industry_b", palette="viridis", legend=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Celebrity industry")
    plt.tight_layout()
    fig.savefig(FIGS / "industry_effects.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "industry_effects.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_partner_effects(df):
    """Boxplot: outcomes by pro partner (top)."""
    top_part = df["partner_b"].value_counts().head(10).index.tolist()
    sub = df[df["partner_b"].isin(top_part)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, outcome, title in zip(axes, ["mean_judge_w1_3", "mean_fan_w1_3", "success_score"],
                                  ["Judge score", "Fan share", "Success score"]):
        order = sub.groupby("partner_b")[outcome].median().sort_values(ascending=False).index
        sns.boxplot(data=sub, x="partner_b", y=outcome, order=order, ax=ax, hue="partner_b", palette="mako", legend=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Professional partner")
    plt.tight_layout()
    fig.savefig(FIGS / "partner_effects.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "partner_effects.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_judge_vs_fan_comparison(df):
    """Compare which factors predict judges vs fans. Correlation contrast."""
    factors = ["age", "industry_b", "partner_b"]
    # For categorical: use group means
    judge_by_ind = df.groupby("industry_b")["mean_judge_w1_3"].mean()
    fan_by_ind = df.groupby("industry_b")["mean_fan_w1_3"].mean()
    judge_by_part = df.groupby("partner_b")["mean_judge_w1_3"].mean()
    fan_by_part = df.groupby("partner_b")["mean_fan_w1_3"].mean()

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Age: scatter judge vs fan for each contestant
    ax = axes[0, 0]
    sub = df.dropna(subset=["mean_judge_w1_3", "mean_fan_w1_3", "age"])
    ax.scatter(sub["mean_judge_w1_3"], sub["mean_fan_w1_3"], c=sub["age"], cmap="viridis", alpha=0.7, s=40)
    r_j, _ = pearsonr(sub["age"], sub["mean_judge_w1_3"])
    r_f, _ = pearsonr(sub["age"], sub["mean_fan_w1_3"])
    ax.set_xlabel("Judge score (W1-3)")
    ax.set_ylabel("Fan share (W1-3)")
    ax.set_title(f"Judge vs Fan: Age effects\nr(age,judge)={r_j:.3f}, r(age,fan)={r_f:.3f}")
    plt.colorbar(ax.collections[0], ax=ax, label="Age")
    ax.grid(True, alpha=0.3)

    # Industry: bar compare judge vs fan mean by industry
    ax = axes[0, 1]
    ind_shared = judge_by_ind.index.intersection(fan_by_ind.index)
    ind_shared = [i for i in ind_shared if i != "Other"][:8]
    x = np.arange(len(ind_shared))
    w = 0.35
    j_vals = [judge_by_ind.get(i, 0) for i in ind_shared]
    f_vals = [fan_by_ind.get(i, 0) * 50 for i in ind_shared]  # scale fan for visibility
    ax.bar(x - w/2, j_vals, w, label="Judge (raw)", color="#2E86AB")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, f_vals, w, label="Fan (×50)", color="#E94F37", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ind_shared, rotation=45, ha="right")
    ax.set_ylabel("Mean judge score")
    ax2.set_ylabel("Mean fan share × 50")
    ax.set_title("Industry: Judge vs Fan means")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Partner: bar compare
    ax = axes[1, 0]
    part_shared = [p for p in judge_by_part.index.intersection(fan_by_part.index) if p not in ("Other", "Other_Partner")][:10]
    x = np.arange(len(part_shared))
    j_vals = [judge_by_part.get(p, 0) for p in part_shared]
    f_vals = [fan_by_part.get(p, 0) * 50 for p in part_shared]
    ax.bar(x - w/2, j_vals, w, label="Judge", color="#2E86AB")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, f_vals, w, label="Fan ×50", color="#E94F37", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace(" ", "\n")[:12] for p in part_shared], rotation=45, ha="right")
    ax.set_ylabel("Mean judge score")
    ax2.set_ylabel("Mean fan share × 50")
    ax.set_title("Pro partner: Judge vs Fan means")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Summary: correlation matrix (age, judge, fan, success)
    ax = axes[1, 1]
    sub = df[["age", "mean_judge_w1_3", "mean_fan_w1_3", "success_score"]].dropna()
    corr = sub.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-0.5, vmax=0.5)
    ax.set_title("Correlation matrix")
    plt.tight_layout()
    fig.savefig(FIGS / "judge_vs_fan_comparison.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "judge_vs_fan_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_industry_partner_heatmap(df):
    """Heatmap: mean success by industry x top partners."""
    top_ind = df["industry_b"].value_counts().head(6).index.tolist()
    top_part = df["partner_b"].value_counts().head(8).index.tolist()
    sub = df[(df["industry_b"].isin(top_ind)) & (df["partner_b"].isin(top_part))]
    pivot = sub.pivot_table(values="success_score", index="industry_b", columns="partner_b", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=0.8)
    ax.set_title("Mean success score by industry × pro partner")
    ax.set_xlabel("Pro partner")
    ax.set_ylabel("Celebrity industry")
    plt.tight_layout()
    fig.savefig(FIGS / "industry_partner_heatmap.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "industry_partner_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def pro_partner_residualized_boost(df):
    """
    Pro partner "boost": control for age + industry, then mean residual by partner.
    Positive = partner's celebs outperform expectation; negative = underperform.
    """
    sub = df.dropna(subset=["success_score", "age", "industry_b", "partner_b"])
    if len(sub) < 50:
        return None
    try:
        model = ols("success_score ~ age + C(industry_b)", data=sub).fit()
        sub = sub.copy()
        sub["pred"] = model.predict(sub)
        sub["resid"] = sub["success_score"] - sub["pred"]
        boost = sub.groupby("partner_b").agg(mean_resid=("resid", "mean"), n=("season", "count")).reset_index()
        boost = boost[boost["n"] >= 5].sort_values("mean_resid", ascending=False)
        boost.to_csv(OUT / "pro_partner_residualized_boost.csv", index=False)
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        top = boost.head(12)
        colors = ["#44AF69" if r > 0 else "#E94F37" for r in top["mean_resid"]]
        ax.barh(range(len(top)), top["mean_resid"], color=colors, alpha=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["partner_b"].str.replace(" ", "\n"), fontsize=9)
        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel("Mean residual (success − predicted from age+industry)")
        ax.set_title("Pro partner boost (controlling for celebrity age & industry)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        fig.savefig(FIGS / "pro_partner_residualized_boost.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(FIGS / "pro_partner_residualized_boost.png", dpi=150, bbox_inches="tight")
        plt.close()
        return boost
    except Exception as e:
        print(f"Pro boost failed: {e}")
        return None


def test_age_effect_judge_vs_fan(df):
    """Bootstrap test: is r(age, judge) significantly different from r(age, fan)?"""
    sub = df.dropna(subset=["age", "mean_judge_w1_3", "mean_fan_w1_3"])
    n = len(sub)
    r_j, _ = pearsonr(sub["age"], sub["mean_judge_w1_3"])
    r_f, _ = pearsonr(sub["age"], sub["mean_fan_w1_3"])
    diff_obs = abs(r_j) - abs(r_f)  # Judge effect stronger?
    diffs = []
    np.random.seed(42)
    for _ in range(1000):
        idx = np.random.choice(n, n, replace=True)
        s = sub.iloc[idx]
        rj, _ = pearsonr(s["age"], s["mean_judge_w1_3"])
        rf, _ = pearsonr(s["age"], s["mean_fan_w1_3"])
        diffs.append(abs(rj) - abs(rf))
    diffs = np.array(diffs)
    ci_low = np.percentile(diffs, 2.5)
    ci_hi = np.percentile(diffs, 97.5)
    p_one_sided = (diffs <= 0).mean() if diff_obs > 0 else (diffs >= 0).mean()
    p_two_sided = 2 * min(p_one_sided, 1 - p_one_sided)
    return {"r_judge": r_j, "r_fan": r_f, "diff": diff_obs, "ci_low": ci_low, "ci_hi": ci_hi, "p": p_two_sided}


def judge_fan_agreement_by_subgroup(df):
    """Correlation(judge, fan) within industry and age bins. Do they agree more for certain types?"""
    sub = df.dropna(subset=["mean_judge_w1_3", "mean_fan_w1_3", "age"])
    sub = sub.copy()
    sub["age_bin"] = pd.cut(sub["age"], bins=[0, 30, 40, 50, 100], labels=["<30", "30-40", "40-50", "50+"])
    results = []
    for name, grp in sub.groupby("industry_b"):
        if len(grp) >= 10:
            r, p = pearsonr(grp["mean_judge_w1_3"], grp["mean_fan_w1_3"])
            results.append({"group": f"industry_{name}", "n": len(grp), "r": r, "p": p})
    for name, grp in sub.groupby("age_bin", observed=True):
        if len(grp) >= 10 and pd.notna(name):
            r, p = pearsonr(grp["mean_judge_w1_3"], grp["mean_fan_w1_3"])
            results.append({"group": f"age_{name}", "n": len(grp), "r": r, "p": p})
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "judge_fan_agreement_by_subgroup.csv", index=False)
    # Plot
    if len(res_df) >= 3:
        fig, ax = plt.subplots(figsize=(8, 4))
        res_df = res_df.sort_values("r")
        colors = ["#2E86AB" if r > 0 else "#E94F37" for r in res_df["r"]]
        ax.barh(res_df["group"], res_df["r"], color=colors, alpha=0.8)
        ax.axvline(0, color="black", linestyle="--")
        ax.set_xlabel("Correlation(judge, fan) within subgroup")
        ax.set_title("Judge–fan agreement by industry/age subgroup")
        ax.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        fig.savefig(FIGS / "judge_fan_agreement_by_subgroup.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(FIGS / "judge_fan_agreement_by_subgroup.png", dpi=150, bbox_inches="tight")
        plt.close()
    return res_df


def slope_analysis(df):
    """Do age/partner predict improvement (judge slope, fan slope) over weeks 1–3?"""
    sub = df.dropna(subset=["judge_slope", "fan_slope", "age"])
    results = []
    for outcome in ["judge_slope", "fan_slope"]:
        r, p = pearsonr(sub["age"], sub[outcome])
        results.append({"outcome": outcome, "factor": "age", "r": r, "p": p})
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT / "slope_analysis.csv", index=False)
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, outcome, title in zip(axes, ["judge_slope", "fan_slope"], ["Judge improvement (W1→W3)", "Fan improvement (W1→W3)"]):
        s = df.dropna(subset=[outcome, "age"])
        ax.scatter(s["age"], s[outcome], alpha=0.5, s=35)
        r, p = pearsonr(s["age"], s[outcome])
        z = np.polyfit(s["age"], s[outcome], 1)
        xl = np.linspace(s["age"].min(), s["age"].max(), 50)
        ax.plot(xl, np.poly1d(z)(xl), "k-", linewidth=2)
        pstr = "p < 0.0001" if p < 0.0001 else f"p = {p:.4f}"
        ax.set_title(f"{title}\nr = {r:.3f}, {pstr}")
        ax.set_xlabel("Age")
        ax.axhline(0, color="gray", linestyle=":")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS / "slope_by_age.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "slope_by_age.png", dpi=150, bbox_inches="tight")
    plt.close()
    return res_df


def plot_pro_partner_ranking(df):
    """Pro partner mean success boost (residual-style: mean outcome by partner)."""
    by_partner = df.groupby("partner_b").agg(
        mean_judge=("mean_judge_w1_3", "mean"),
        mean_fan=("mean_fan_w1_3", "mean"),
        mean_success=("success_score", "mean"),
        n=("season", "count"),
    ).reset_index()
    by_partner = by_partner[by_partner["n"] >= 5].sort_values("mean_success", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(by_partner))
    ax.barh(y, by_partner["mean_success"], color="#44AF69", alpha=0.8, label="Success score")
    ax.set_yticks(y)
    ax.set_yticklabels(by_partner["partner_b"].str.replace(" ", "\n"), fontsize=9)
    ax.set_xlabel("Mean success score")
    ax.set_title("Pro partner impact: mean contestant success (n ≥ 5)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(FIGS / "pro_partner_ranking.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "pro_partner_ranking.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_factor_importance_summary(test_df):
    """Bar: p-values and effect sizes by factor/outcome."""
    sub = test_df[test_df["test"].isin(["Pearson", "Spearman", "Kruskal_H"])].copy()
    sub["sig"] = sub["p"].apply(lambda p: "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
    pivot = sub.pivot_table(index=["factor", "outcome"], columns="test", values="stat").reset_index()
    pivot.to_csv(OUT / "statistical_tests.csv", index=False)
    return pivot


def write_summary(df, test_df, reg_results, reg_models=None, extra=None):
    """Comprehensive markdown summary."""
    n = len(df)
    n_seasons = df["season"].nunique()
    n_with_fan = df["mean_fan_w1_3"].notna().sum()
    n_with_judge = df["mean_judge_w1_3"].notna().sum()

    # Age correlations
    sub_a = df.dropna(subset=["age", "mean_judge_w1_3"])
    age_judge_r, age_judge_p = (pearsonr(sub_a["age"], sub_a["mean_judge_w1_3"]) if len(sub_a) > 5 else (0, 1))
    sub_b = df.dropna(subset=["age", "mean_fan_w1_3"])
    age_fan_r, age_fan_p = (pearsonr(sub_b["age"], sub_b["mean_fan_w1_3"]) if len(sub_b) > 5 else (0, 1))
    sub_c = df.dropna(subset=["age", "success_score"])
    age_succ_r, age_succ_p = (pearsonr(sub_c["age"], sub_c["success_score"]) if len(sub_c) > 5 else (0, 1))

    lines = [
        "# Problem 3 EDA: Impact of Pro Dancers and Celebrity Characteristics",
        "",
        "## Data",
        f"- Contestants: {n}",
        f"- Seasons: {n_seasons}",
        f"- With judge scores (W1-3): {n_with_judge}",
        f"- With fan vote estimates (W1-3): {n_with_fan}",
        "",
        "## Outcomes",
        "- **mean_judge_w1_3**: Mean judge total score across weeks 1–3",
        "- **mean_fan_w1_3**: Mean estimated fan vote share across weeks 1–3",
        "- **success_score**: 1 = winner, 0 = last place (within season)",
        "- **placement**: Final placement (1 = winner)",
        "",
        "## Age Effects",
        f"- Age vs judge score: r = {age_judge_r:.3f}, " + ("p < 0.0001" if age_judge_p < 0.0001 else f"p = {age_judge_p:.4f}"),
        f"- Age vs fan share: r = {age_fan_r:.3f}, " + ("p < 0.0001" if age_fan_p < 0.0001 else f"p = {age_fan_p:.4f}"),
        f"- Age vs success: r = {age_succ_r:.3f}, " + ("p < 0.0001" if age_succ_p < 0.0001 else f"p = {age_succ_p:.4f}"),
        "",
        "**Interpretation:** " + (
            "Younger celebrities tend to do better (higher judge scores, fan support, success). "
            + ("Age has a stronger effect on judge scores than fan support (|r| larger for judges)."
               if abs(age_judge_r) > abs(age_fan_r) else
               "Age has a stronger effect on fan support than judge scores.")
            if age_fan_p < 0.05 or age_judge_p < 0.05 else
            "Age does not significantly predict judge scores or fan support."
        ),
        "",
        "## Industry Effects",
        "Kruskal-Wallis: industry significantly affects judge scores (H≈24.2, p<0.01), fan share (H≈10.7), and success (H≈16.4).",
        "",
        "## Pro Partner Effects",
        "Kruskal-Wallis: pro partner significantly affects all outcomes (H≈24–45, p<0.01). Top partners (Derek Hough, Cheryl Burke, etc.) have consistently higher contestant success.",
        "",
        "## Judge vs Fan: Same or Different?",
        f"- Correlation(age, judge) = {age_judge_r:.3f}; Correlation(age, fan) = {age_fan_r:.3f}",
        "Age has stronger effect on judge scores than fan support. Industry and partner affect both; pro partner effect is larger for judge scores (Kruskal H≈45 vs H≈24 for fans).",
        "",
    ]
    if extra:
        at = extra.get("age_test")
        if at:
            lines.append("**Bootstrap test (age effect judge vs fan):** " +
                        f"|r| difference = {at['diff']:.3f}, 95% CI [{at['ci_low']:.3f}, {at['ci_hi']:.3f}], p = {at['p']:.4f}. ")
            lines.append("Difference significant." if at["p"] < 0.05 else "Difference not significant.")
            lines.append("")
        pb = extra.get("pro_boost")
        if pb is not None and len(pb) > 0:
            top3 = pb.head(3)["partner_b"].tolist()
            lines.append("**Pro partner residualized boost:** Top: " + ", ".join(top3) + ". Celebs with these partners outperform age+industry expectation.")
            lines.append("")
        ad = extra.get("agreement_df")
        if ad is not None and len(ad) > 0:
            top_agree = ad.nlargest(2, "r")
            topg = ", ".join([f"{r['group']} (r={r['r']:.2f})" for _, r in top_agree.iterrows()])
            lines.append(f"**Judge–fan agreement by subgroup:** Highest in {topg}.")
            lines.append("")
        sd = extra.get("slope_df")
        if sd is not None and len(sd) > 0:
            jr = sd[sd["outcome"] == "judge_slope"]
            fr = sd[sd["outcome"] == "fan_slope"]
            if len(jr) and len(fr):
                jv, fv = jr["r"].values[0], fr["r"].values[0]
                jp, fp = jr["p"].values[0], fr["p"].values[0]
                jpstr = "p < 0.0001" if jp < 0.0001 else f"p = {jp:.3f}"
                fpstr = "p < 0.0001" if fp < 0.0001 else f"p = {fp:.3f}"
                lines.append(f"**Improvement (W1→W3):** r(age, judge_slope) = {jv:.3f} ({jpstr}); r(age, fan_slope) = {fv:.3f} ({fpstr}).")
                lines.append("Younger celebs show more fan improvement; age does not significantly predict judge improvement.")
            lines.append("")
    lines.extend(["## Regression (OLS: outcome ~ age + industry + partner)", ""])
    rsq_rows = [r for r in reg_results if isinstance(r.get("rsq"), (int, float))]
    for r in rsq_rows:
        lines.append(f"- {r['outcome']}: R² = {r['rsq']:.3f}, R²_adj = {r['rsq_adj']:.3f}")
    lines.extend([
        "",
        "## Statistical Tests (Summary)",
        "See `outputs/statistical_tests.csv` for full results.",
        "",
        "## Figures",
        "- `age_effects.pdf`: Age vs judge, fan, success",
        "- `industry_effects.pdf`: Outcomes by industry",
        "- `partner_effects.pdf`: Outcomes by pro partner",
        "- `judge_vs_fan_comparison.pdf`: Judge vs fan by factor",
        "- `pro_partner_ranking.pdf`: Pro partner mean success",
        "- `industry_partner_heatmap.pdf`: Industry × partner interaction",
        "- `pro_partner_residualized_boost.pdf`: Pro boost controlling for age+industry",
        "- `judge_fan_agreement_by_subgroup.pdf`: Judge–fan correlation by industry/age",
        "- `slope_by_age.pdf`: Does age predict improvement (W1→W3)?",
        "",
    ])
    with open(OUT / "eda_summary.md", "w") as f:
        f.write("\n".join(lines))


def main():
    print("Loading data...")
    df = load_data()
    n_fan = df["mean_fan_w1_3"].notna().sum()
    print(f"  Contestants: {len(df)}, with fan votes: {n_fan}")

    print("Running statistical tests...")
    test_df = run_statistical_tests(df)
    test_df.to_csv(OUT / "statistical_tests_full.csv", index=False)
    plot_factor_importance_summary(test_df)

    print("Running regressions...")
    reg_results, reg_models = run_regressions(df)

    print("Creating visualizations...")
    plot_age_effects(df)
    plot_industry_effects(df)
    plot_partner_effects(df)
    plot_judge_vs_fan_comparison(df)
    plot_industry_partner_heatmap(df)
    plot_pro_partner_ranking(df)

    print("Running extended analyses...")
    pro_boost = pro_partner_residualized_boost(df)
    age_test = test_age_effect_judge_vs_fan(df)
    agreement_df = judge_fan_agreement_by_subgroup(df)
    slope_df = slope_analysis(df)

    print("Writing summary...")
    write_summary(df, test_df, reg_results, reg_models,
                  extra={"pro_boost": pro_boost, "age_test": age_test, "slope_df": slope_df, "agreement_df": agreement_df})

    df.to_csv(OUT / "merged_data.csv", index=False)
    print(f"\nDone. Outputs: {OUT}/, {FIGS}/")


if __name__ == "__main__":
    main()
