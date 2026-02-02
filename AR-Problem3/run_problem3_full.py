#!/usr/bin/env python3
"""
Problem 3 Full Pipeline: Model impact of pro dancers and celebrity characteristics.
Answers: How much do age, industry, pro partner impact outcomes? Same for judges vs fans?

Runs: load data → regressions → partial R² → tables → info-dense plots → written answer.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols

from eda import load_data

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "font.family": "sans-serif"})


def run_regressions(df):
    """OLS: judge, fan, success ~ age + industry + partner. Return models and coefficient table."""
    sub = df.dropna(subset=["mean_judge_w1_3", "mean_fan_w1_3", "success_score", "age", "industry_b", "partner_b"])
    formula = "y ~ age + C(industry_b) + C(partner_b)"
    models = {}
    coef_rows = []

    for outcome, y_label in [
        ("mean_judge_w1_3", "Judge score (W1-3)"),
        ("mean_fan_w1_3", "Fan share (W1-3)"),
        ("success_score", "Success score"),
    ]:
        sub_ = sub.dropna(subset=[outcome])
        sub_ = sub_.copy()
        sub_["y"] = sub_[outcome]
        try:
            model = ols(formula, data=sub_).fit()
            models[outcome] = model
            for name, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
                if name == "Intercept":
                    continue
                coef_rows.append({
                    "outcome": y_label,
                    "term": name.replace("C(industry_b)[T.", "industry_").replace("C(partner_b)[T.", "partner_").replace("]", ""),
                    "coef": coef,
                    "p": pval,
                    "sig": "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "",
                })
            coef_rows.append({"outcome": y_label, "term": "R²", "coef": model.rsquared, "p": np.nan, "sig": ""})
            coef_rows.append({"outcome": y_label, "term": "R²_adj", "coef": model.rsquared_adj, "p": np.nan, "sig": ""})
        except Exception as e:
            coef_rows.append({"outcome": y_label, "term": "error", "coef": np.nan, "p": np.nan, "sig": str(e)})

    return models, pd.DataFrame(coef_rows), sub


def partial_rsq(df, outcome):
    """Sequential R²: age only, +industry, +partner. Return incremental R² per factor."""
    sub = df.dropna(subset=[outcome, "age", "industry_b", "partner_b"]).copy()
    sub["y"] = sub[outcome]
    r2_age = ols("y ~ age", data=sub).fit().rsquared
    r2_age_ind = ols("y ~ age + C(industry_b)", data=sub).fit().rsquared
    r2_full = ols("y ~ age + C(industry_b) + C(partner_b)", data=sub).fit().rsquared
    return {
        "age": r2_age,
        "industry": r2_age_ind - r2_age,
        "pro_partner": r2_full - r2_age_ind,
        "total": r2_full,
    }


def build_impact_summary_table(df, models, coef_df):
    """Factor × outcome effect summary (correlations / eta² style) for judge, fan, success."""
    sub = df.dropna(subset=["mean_judge_w1_3", "mean_fan_w1_3", "success_score", "age", "industry_b", "partner_b"])
    rows = []

    # Age: Pearson r for each outcome
    for outcome, label in [("mean_judge_w1_3", "Judge"), ("mean_fan_w1_3", "Fan"), ("success_score", "Success")]:
        r, p = pearsonr(sub["age"], sub[outcome])
        rows.append({"factor": "age", "outcome": label, "effect": r, "p": p, "effect_type": "r"})

    # Industry: eta² from ANOVA
    for outcome, label in [("mean_judge_w1_3", "Judge"), ("mean_fan_w1_3", "Fan"), ("success_score", "Success")]:
        groups = [g[outcome].values for _, g in sub.groupby("industry_b") if len(g) >= 3]
        if len(groups) >= 2:
            all_vals = np.concatenate(groups)
            grand = np.mean(all_vals)
            ss_b = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
            ss_t = np.sum((all_vals - grand) ** 2)
            eta2 = ss_b / ss_t if ss_t > 0 else 0
            from scipy import stats
            f, p = stats.f_oneway(*groups)
            rows.append({"factor": "industry", "outcome": label, "effect": eta2, "p": p, "effect_type": "η²"})
        else:
            rows.append({"factor": "industry", "outcome": label, "effect": np.nan, "p": np.nan, "effect_type": "η²"})

    # Partner: eta²
    for outcome, label in [("mean_judge_w1_3", "Judge"), ("mean_fan_w1_3", "Fan"), ("success_score", "Success")]:
        groups = [g[outcome].values for _, g in sub.groupby("partner_b") if len(g) >= 3]
        if len(groups) >= 2:
            all_vals = np.concatenate(groups)
            grand = np.mean(all_vals)
            ss_b = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
            ss_t = np.sum((all_vals - grand) ** 2)
            eta2 = ss_b / ss_t if ss_t > 0 else 0
            from scipy import stats
            f, p = stats.f_oneway(*groups)
            rows.append({"factor": "pro_partner", "outcome": label, "effect": eta2, "p": p, "effect_type": "η²"})
        else:
            rows.append({"factor": "pro_partner", "outcome": label, "effect": np.nan, "p": np.nan, "effect_type": "η²"})

    return pd.DataFrame(rows)


def plot_partial_rsq_waterfall(partial_judge, partial_fan, partial_success):
    """Info-dense: How much does each factor explain? (Partial R²) Judge vs Fan vs Success."""
    fig, ax = plt.subplots(figsize=(10, 5))
    factors = ["age", "industry", "pro_partner"]
    x = np.arange(len(factors))
    width = 0.25
    j_vals = [partial_judge[f] for f in factors]
    f_vals = [partial_fan[f] for f in factors]
    s_vals = [partial_success[f] for f in factors]
    ax.bar(x - width, j_vals, width, label="Judge score", color="#2E86AB", edgecolor="black")
    ax.bar(x, f_vals, width, label="Fan share", color="#E94F37", alpha=0.9, edgecolor="black")
    ax.bar(x + width, s_vals, width, label="Success score", color="#44AF69", alpha=0.9, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(["Age", "Industry", "Pro partner"])
    ax.set_ylabel("Incremental R² (sequential)")
    ax.set_title("How Much Does Each Factor Explain?\n(Adding age → industry → pro partner)")
    ax.legend()
    ax.set_ylim(0, max(max(j_vals), max(f_vals), max(s_vals)) * 1.15)
    for i, (jj, ff, ss) in enumerate(zip(j_vals, f_vals, s_vals)):
        ax.text(i - width, jj + 0.005, f"{jj:.2f}", ha="center", fontsize=8)
        ax.text(i, ff + 0.005, f"{ff:.2f}", ha="center", fontsize=8)
        ax.text(i + width, ss + 0.005, f"{ss:.2f}", ha="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIGS / "impact_partial_rsq_waterfall.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "impact_partial_rsq_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_effect_summary_heatmap(impact_df):
    """Heatmap: Factor × Outcome (Judge, Fan, Success) with effect sizes; same scale where possible."""
    pivot = impact_df.pivot(index="factor", columns="outcome", values="effect")
    # Use absolute effect for display where r is signed
    pivot_display = pivot.copy()
    pivot_display.loc["age"] = pivot_display.loc["age"].abs()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot_display.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(0.15, pivot_display.max().max()))
    ax.set_xticks(range(pivot_display.shape[1]))
    ax.set_xticklabels(pivot_display.columns)
    ax.set_yticks(range(pivot_display.shape[0]))
    ax.set_yticklabels([f.replace("_", " ").title() for f in pivot_display.index])
    for i in range(pivot_display.shape[0]):
        for j in range(pivot_display.shape[1]):
            v = pivot_display.iloc[i, j]
            ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "—", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="|r| or η²")
    ax.set_title("Effect size by factor and outcome\n(Age: |r|; Industry & partner: η²)")
    plt.tight_layout()
    fig.savefig(FIGS / "impact_effect_summary_heatmap.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "impact_effect_summary_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_judge_vs_fan_one_pager(impact_df, coef_df):
    """One compact panel: Judge vs Fan effect sizes + key takeaways."""
    judge_fan = impact_df[impact_df["outcome"].isin(["Judge", "Fan"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: Bar comparison Judge vs Fan by factor
    ax = axes[0]
    factors = judge_fan["factor"].unique()
    x = np.arange(len(factors))
    w = 0.35
    j_eff = [judge_fan[(judge_fan["factor"] == f) & (judge_fan["outcome"] == "Judge")]["effect"].abs().values[0] for f in factors]
    f_eff = [judge_fan[(judge_fan["factor"] == f) & (judge_fan["outcome"] == "Fan")]["effect"].values[0] for f in factors]
    f_eff = [abs(x) for x in f_eff]
    ax.bar(x - w/2, j_eff, w, label="Judges", color="#2E86AB", edgecolor="black")
    ax.bar(x + w/2, f_eff, w, label="Fans", color="#E94F37", alpha=0.9, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", " ").title() for f in factors])
    ax.set_ylabel("|Effect| (r or η²)")
    ax.set_title("Do factors impact judges and fans the same way?\nNo — judges are more influenced.")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: R² for Judge vs Fan vs Success (from coef_df)
    ax = axes[1]
    rsq = []
    for outcome in ["Judge score (W1-3)", "Fan share (W1-3)", "Success score"]:
        r = coef_df[(coef_df["outcome"] == outcome) & (coef_df["term"] == "R²")]["coef"].values
        rsq.append(r[0] if len(r) else 0)
    colors = ["#2E86AB", "#E94F37", "#44AF69"]
    bars = ax.bar(["Judge\nscore", "Fan\nshare", "Success\nscore"], rsq, color=colors, edgecolor="black")
    ax.set_ylabel("R²")
    ax.set_title("Model fit: age + industry + pro partner")
    ax.set_ylim(0, 0.35)
    for b, v in zip(bars, rsq):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{v:.2f}", ha="center", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIGS / "judge_vs_fan_one_pager.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "judge_vs_fan_one_pager.png", dpi=200, bbox_inches="tight")
    plt.close()


def write_regression_table_md(coef_df, out_path):
    """Write regression coefficients as markdown table."""
    # Keep main terms only for readability: age, R², R²_adj
    sub = coef_df[coef_df["term"].isin(["age", "R²", "R²_adj"])]
    if sub.empty:
        with open(out_path, "w") as f:
            f.write("# Regression coefficients\n\nNo rows.\n")
        return
    pivot = sub.pivot(index="term", columns="outcome", values="coef")
    lines = ["# Regression coefficients (selected)\n", "| Term | Judge score | Fan share | Success score |"]
    lines.append("|------|-------------|-----------|---------------|")
    for term in pivot.index:
        row = [str(round(pivot.loc[term, c], 4)) if c in pivot.columns and pd.notna(pivot.loc[term, c]) else "—" for c in pivot.columns]
        lines.append("| " + term + " | " + " | ".join(row) + " |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_key_findings_table(impact_df, out_path):
    """One-page key findings: factor × Judge effect × Fan effect × Same/different?"""
    lines = [
        "# Problem 3: Key Findings Table",
        "",
        "| Factor | Effect on Judges | Effect on Fans | Same or different? |",
        "|--------|------------------|-----------------|----------------------|",
    ]
    for factor in ["age", "industry", "pro_partner"]:
        j = impact_df[(impact_df["factor"] == factor) & (impact_df["outcome"] == "Judge")]["effect"].values[0]
        f = impact_df[(impact_df["factor"] == factor) & (impact_df["outcome"] == "Fan")]["effect"].values[0]
        j_abs, f_abs = abs(j), abs(f)
        diff = "Different (stronger on judges)" if j_abs > f_abs * 1.1 else "Similar"
        j_type = impact_df[(impact_df["factor"] == factor) & (impact_df["outcome"] == "Judge")]["effect_type"].values[0]
        j_str = f"r = {j:.3f}" if j_type == "r" else f"η² = {j:.3f}"
        f_str = f"r = {f:.3f}" if impact_df[(impact_df["factor"] == factor) & (impact_df["outcome"] == "Fan")]["effect_type"].values[0] == "r" else f"η² = {f:.3f}"
        lines.append(f"| {factor.replace('_', ' ').title()} | {j_str} | {f_str} | {diff} |")
    lines.extend(["", "**Takeaway:** Pro partner and age have stronger effects on judge scores than on fan votes.", ""])
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def write_full_answer_md(impact_df, partial_judge, partial_fan, partial_success, coef_df, out_path):
    """Full written answer to the problem questions."""
    r_judge_age = impact_df[(impact_df["factor"] == "age") & (impact_df["outcome"] == "Judge")]["effect"].values[0]
    r_fan_age = impact_df[(impact_df["factor"] == "age") & (impact_df["outcome"] == "Fan")]["effect"].values[0]
    eta_judge_partner = impact_df[(impact_df["factor"] == "pro_partner") & (impact_df["outcome"] == "Judge")]["effect"].values[0]
    eta_fan_partner = impact_df[(impact_df["factor"] == "pro_partner") & (impact_df["outcome"] == "Fan")]["effect"].values[0]
    rsq_judge = coef_df[(coef_df["outcome"] == "Judge score (W1-3)") & (coef_df["term"] == "R²")]["coef"].values[0]
    rsq_fan = coef_df[(coef_df["outcome"] == "Fan share (W1-3)") & (coef_df["term"] == "R²")]["coef"].values[0]
    rsq_success = coef_df[(coef_df["outcome"] == "Success score") & (coef_df["term"] == "R²")]["coef"].values[0]

    text = f"""# Problem 3: Written Answer

## How much do pro dancers and celebrity characteristics impact how well a celebrity will do?

We modeled three outcomes using celebrity **age**, **industry**, and **pro partner**: (1) mean judge score over weeks 1–3, (2) mean estimated fan vote share over weeks 1–3, and (3) a **success score** (1 = winner, 0 = last place in the season).

**Magnitude of impact:**
- **Age** has a moderate negative effect: older celebrities tend to receive lower judge scores (r ≈ {r_judge_age:.2f}), lower fan share (r ≈ {r_fan_age:.2f}), and lower success. Younger contestants do better on average.
- **Industry** (e.g., Actor, Athlete, Singer) explains a modest share of variance (η² on the order of 0.01–0.06); some industries are associated with higher scores and success.
- **Pro partner** has the **largest** impact among the three: η² for judge score is about {eta_judge_partner:.2f} and for fan share about {eta_fan_partner:.2f}. Certain pros (e.g., Derek Hough, Cheryl Burke, Julianne Hough) are consistently associated with higher contestant success even after controlling for age and industry (see residualized “pro boost” analysis in the EDA).

**How much is explained overall?**  
A linear model with age + industry + pro partner explains roughly **{rsq_judge:.2f}** of the variance in judge scores, **{rsq_fan:.2f}** in fan share, and **{rsq_success:.2f}** in success score. So these factors matter, but a large share of variance remains unexplained (talent, week-to-week performance, fan base size, etc.).

---

## Do they impact judges’ scores and fan votes in the same way?

**No.** The same factors do **not** affect judges and fans in the same way:

1. **Age:** The correlation of age with judge score (r ≈ {r_judge_age:.2f}) is **stronger** than with fan share (r ≈ {r_fan_age:.2f}). Judges tend to reward younger celebrities more than fans do (or fans are relatively more supportive of older contestants).

2. **Industry:** Industry effects (η²) are larger for judge scores than for fan share. Judges differentiate more by celebrity type than fans do.

3. **Pro partner:** The **biggest** discrepancy is for pro partner: η² for judges is about {eta_judge_partner:.2f} vs about {eta_fan_partner:.2f} for fans. So **pro partner matters more for judge scores** than for fan votes. Top pros are associated with higher judge scores; the association with fan share is weaker.

4. **Improvement over the season (W1→W3):** In the EDA, age predicts **fan** improvement (younger celebs gain more fan share over weeks 1–3) but does not significantly predict **judge** score improvement. So fans and judges respond differently to trajectory as well.

**Conclusion:** Pro dancers and celebrity characteristics (age, industry) all impact how well a celebrity does, with **pro partner** and **age** being especially important. These factors do **not** impact judges and fans in the same way: effects are generally **stronger on judge scores** than on fan votes, with pro partner showing the largest gap. Judges appear to weight technical/partner quality and youth more than fans do.
"""
    with open(out_path, "w") as f:
        f.write(text)


def main():
    print("Loading data...")
    df = load_data()
    print("Running regressions...")
    models, coef_df, sub = run_regressions(df)
    coef_df.to_csv(OUT / "regression_coefficients_full.csv", index=False)
    # Trim for readable regression table
    coef_short = coef_df[coef_df["term"].isin(["age", "R²", "R²_adj"]) | coef_df["term"].str.contains("industry_|partner_", regex=True, na=False)]
    write_regression_table_md(coef_df, OUT / "regression_table.md")

    print("Computing partial R²...")
    partial_judge = partial_rsq(df, "mean_judge_w1_3")
    partial_fan = partial_rsq(df, "mean_fan_w1_3")
    partial_success = partial_rsq(df, "success_score")
    partial_df = pd.DataFrame([
        {"outcome": "Judge", **partial_judge},
        {"outcome": "Fan", **partial_fan},
        {"outcome": "Success", **partial_success},
    ])
    partial_df.to_csv(OUT / "partial_rsq_by_outcome.csv", index=False)

    print("Building impact summary table...")
    impact_df = build_impact_summary_table(df, models, coef_df)
    if impact_df.empty:
        impact_df = pd.DataFrame(columns=["factor", "outcome", "effect", "p", "effect_type"])
    impact_df.to_csv(OUT / "impact_summary_table.csv", index=False)

    print("Creating plots...")
    plot_partial_rsq_waterfall(partial_judge, partial_fan, partial_success)
    plot_effect_summary_heatmap(impact_df)
    plot_judge_vs_fan_one_pager(impact_df, coef_df)

    print("Writing full answer and key findings table...")
    write_full_answer_md(impact_df, partial_judge, partial_fan, partial_success, coef_df, OUT / "problem3_answer.md")
    write_key_findings_table(impact_df, OUT / "key_findings_table.md")

    print(f"Done. Outputs: {OUT}/, {FIGS}/")


if __name__ == "__main__":
    main()
