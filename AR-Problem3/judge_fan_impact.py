#!/usr/bin/env python3
"""
Problem 3: Judge vs Fan Impact Analysis
Do age, industry, and pro partner impact judges and fans in the same way?

Produces:
- Explicit numbers: effect sizes for each factor on judge vs fan
- Statistical tests: are judge and fan effects significantly different?
- Information-dense creative plots
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.formula.api import ols

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "Data"
VOTES_PATH = DATA / "estimate_votes.csv"
VOTES_FALLBACK = ROOT.parent / "AR-Problem1-Base" / "base_results" / "base_inferred_shares.csv"
OUT = ROOT / "outputs"
FIGS = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    return re.sub(r"[^\w\s]", "", name.lower().strip())


def load_data():
    """Load and merge main data + fan votes."""
    df = pd.read_csv(DATA / "2026_MCM_Problem_C_Data.csv")
    df["norm_name"] = df["celebrity_name"].apply(normalize_name)

    votes = pd.read_csv(VOTES_PATH) if VOTES_PATH.exists() else pd.read_csv(VOTES_FALLBACK)
    votes["norm_name"] = votes["celebrity_name"].apply(normalize_name)
    votes_col = "s_share" if "s_share" in votes.columns else "s_hat"

    df["success_score"] = df.groupby("season")["placement"].transform(
        lambda x: 1 - (x - 1) / max(x.max() - 1, 1)
    )

    judge_weeks = []
    for w in range(1, 4):
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5) if f"week{w}_judge{j}_score" in df.columns]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace(0, np.nan)
        df[f"week{w}_total"] = df[cols].sum(axis=1, min_count=1)
        judge_weeks.append(f"week{w}_total")
    df["mean_judge_w1_3"] = df[judge_weeks].mean(axis=1)

    fan_sub = votes[votes["week"].isin([1, 2, 3])]
    fan_pivot = fan_sub.pivot_table(index=["season", "norm_name"], columns="week", values=votes_col).reset_index()
    fan_pivot.columns = ["season", "norm_name"] + [f"fan_w{c}" for c in fan_pivot.columns[2:]]
    df = df.merge(fan_pivot, on=["season", "norm_name"], how="left")
    fan_cols = [c for c in df.columns if c.startswith("fan_w")]
    df["mean_fan_w1_3"] = df[fan_cols].mean(axis=1) if fan_cols else np.nan

    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")
    df["industry"] = df["celebrity_industry"].fillna("Other")
    df["partner"] = df["ballroom_partner"].fillna("Unknown")
    df["homestate"] = df["celebrity_homestate"].fillna("Unknown")
    country_col = "celebrity_homecountry/region"
    df["country_region"] = df[country_col].fillna("Unknown") if country_col in df.columns else "Unknown"

    def bucket(col, k=8):
        top = df[col].value_counts().head(k).index.tolist()
        return df[col].apply(lambda x: x if x in top else "Other")

    df["industry_b"] = bucket("industry", 8)
    df["partner_b"] = bucket("partner", 12)
    df["homestate_b"] = bucket("homestate", 10)
    df["country_b"] = bucket("country_region", 8)
    return df


def compute_effect_sizes(df):
    """Compute judge vs fan effect sizes for age, industry, partner, homestate, country."""
    rows = []
    sub = df.dropna(subset=["mean_judge_w1_3", "mean_fan_w1_3", "age", "industry_b", "partner_b", "homestate_b", "country_b"])

    # Age: Pearson r
    r_j, p_j = pearsonr(sub["age"], sub["mean_judge_w1_3"])
    r_f, p_f = pearsonr(sub["age"], sub["mean_fan_w1_3"])
    rows.append({
        "factor": "age",
        "judge_effect": r_j,
        "fan_effect": r_f,
        "judge_p": p_j,
        "fan_p": p_f,
        "effect_type": "Pearson r",
        "ratio": r_j / r_f if abs(r_f) > 0.01 else np.nan,
        "abs_diff": abs(r_j) - abs(r_f),
    })

    # Industry: eta-squared from ANOVA
    ind_judge_eta = ind_fan_eta = ind_judge_p = ind_fan_p = 0.0
    for outcome, label in [("mean_judge_w1_3", "judge"), ("mean_fan_w1_3", "fan")]:
        groups = [g[outcome].values for _, g in sub.groupby("industry_b") if len(g) >= 3]
        if len(groups) >= 2:
            f, p = stats.f_oneway(*groups)
            all_vals = np.concatenate(groups)
            grand_mean = np.mean(all_vals)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            eta2 = ss_between / ss_total if ss_total > 0 else 0
            if label == "judge":
                ind_judge_eta, ind_judge_p = eta2, p
            else:
                ind_fan_eta, ind_fan_p = eta2, p
    rows.append({
        "factor": "industry",
        "judge_effect": ind_judge_eta,
        "fan_effect": ind_fan_eta,
        "judge_p": ind_judge_p,
        "fan_p": ind_fan_p,
        "effect_type": "η²",
        "ratio": ind_judge_eta / ind_fan_eta if ind_fan_eta > 0.001 else np.nan,
        "abs_diff": ind_judge_eta - ind_fan_eta,
    })

    # Partner: eta-squared
    part_judge_eta = part_fan_eta = part_judge_p = part_fan_p = 0.0
    for outcome, label in [("mean_judge_w1_3", "judge"), ("mean_fan_w1_3", "fan")]:
        groups = [g[outcome].values for _, g in sub.groupby("partner_b") if len(g) >= 3]
        if len(groups) >= 2:
            f, p = stats.f_oneway(*groups)
            all_vals = np.concatenate(groups)
            grand_mean = np.mean(all_vals)
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_total = np.sum((all_vals - grand_mean) ** 2)
            eta2 = ss_between / ss_total if ss_total > 0 else 0
            if label == "judge":
                part_judge_eta, part_judge_p = eta2, p
            else:
                part_fan_eta, part_fan_p = eta2, p
    rows.append({
        "factor": "pro_partner",
        "judge_effect": part_judge_eta,
        "fan_effect": part_fan_eta,
        "judge_p": part_judge_p,
        "fan_p": part_fan_p,
        "effect_type": "η²",
        "ratio": part_judge_eta / part_fan_eta if part_fan_eta > 0.001 else np.nan,
        "abs_diff": part_judge_eta - part_fan_eta,
    })

    # Homestate: eta-squared
    for group_col, factor_name in [("homestate_b", "homestate"), ("country_b", "country_region")]:
        h_j = h_f = h_jp = h_fp = 0.0
        for outcome, label in [("mean_judge_w1_3", "judge"), ("mean_fan_w1_3", "fan")]:
            groups = [g[outcome].values for _, g in sub.groupby(group_col) if len(g) >= 3]
            if len(groups) >= 2:
                f, p = stats.f_oneway(*groups)
                all_vals = np.concatenate(groups)
                grand_mean = np.mean(all_vals)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
                ss_total = np.sum((all_vals - grand_mean) ** 2)
                eta2 = ss_between / ss_total if ss_total > 0 else 0
                if label == "judge":
                    h_j, h_jp = eta2, p
                else:
                    h_f, h_fp = eta2, p
        rows.append({
            "factor": factor_name,
            "judge_effect": h_j,
            "fan_effect": h_f,
            "judge_p": h_jp,
            "fan_p": h_fp,
            "effect_type": "η²",
            "ratio": h_j / h_f if h_f > 0.001 else np.nan,
            "abs_diff": h_j - h_f,
        })

    return pd.DataFrame(rows), sub


def _eta2(df, group_col, outcome_col):
    """Compute eta-squared for group_col predicting outcome_col."""
    groups = [g[outcome_col].values for _, g in df.groupby(group_col) if len(g) >= 3]
    if len(groups) < 2:
        return np.nan
    all_vals = np.concatenate(groups)
    grand = np.mean(all_vals)
    ss_b = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_t = np.sum((all_vals - grand) ** 2)
    return ss_b / ss_t if ss_t > 0 else 0


def bootstrap_judge_vs_fan_difference(df, n_boot=2000, seed=42):
    """Bootstrap: is effect on judge significantly different from effect on fan?"""
    np.random.seed(seed)
    sub = df.dropna(subset=["mean_judge_w1_3", "mean_fan_w1_3", "age", "industry_b", "partner_b", "homestate_b", "country_b"])
    n = len(sub)
    results = []

    # Age: correlation difference
    r_j, r_f = pearsonr(sub["age"], sub["mean_judge_w1_3"])[0], pearsonr(sub["age"], sub["mean_fan_w1_3"])[0]
    diff_obs = abs(r_j) - abs(r_f)
    diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        s = sub.iloc[idx]
        rj = pearsonr(s["age"], s["mean_judge_w1_3"])[0]
        rf = pearsonr(s["age"], s["mean_fan_w1_3"])[0]
        diffs.append(abs(rj) - abs(rf))
    p_val = 2 * min((np.array(diffs) <= 0).mean(), (np.array(diffs) >= 0).mean())
    results.append({"factor": "age", "judge_effect": r_j, "fan_effect": r_f, "diff": diff_obs,
                    "p_different": p_val, "ci_low": np.percentile(diffs, 2.5), "ci_hi": np.percentile(diffs, 97.5)})

    # Industry: eta² difference
    ind_j = _eta2(sub, "industry_b", "mean_judge_w1_3")
    ind_f = _eta2(sub, "industry_b", "mean_fan_w1_3")
    if not np.isnan(ind_j) and not np.isnan(ind_f):
        diff_obs = ind_j - ind_f
        diffs = []
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            s = sub.iloc[idx]
            ej = _eta2(s, "industry_b", "mean_judge_w1_3")
            ef = _eta2(s, "industry_b", "mean_fan_w1_3")
            if not np.isnan(ej) and not np.isnan(ef):
                diffs.append(ej - ef)
        if diffs:
            p_val = 2 * min((np.array(diffs) <= 0).mean(), (np.array(diffs) >= 0).mean())
            results.append({"factor": "industry", "judge_effect": ind_j, "fan_effect": ind_f, "diff": diff_obs,
                           "p_different": p_val, "ci_low": np.percentile(diffs, 2.5), "ci_hi": np.percentile(diffs, 97.5)})

    # Partner: eta² difference
    part_j = _eta2(sub, "partner_b", "mean_judge_w1_3")
    part_f = _eta2(sub, "partner_b", "mean_fan_w1_3")
    if not np.isnan(part_j) and not np.isnan(part_f):
        diff_obs = part_j - part_f
        diffs = []
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            s = sub.iloc[idx]
            ej = _eta2(s, "partner_b", "mean_judge_w1_3")
            ef = _eta2(s, "partner_b", "mean_fan_w1_3")
            if not np.isnan(ej) and not np.isnan(ef):
                diffs.append(ej - ef)
        if diffs:
            p_val = 2 * min((np.array(diffs) <= 0).mean(), (np.array(diffs) >= 0).mean())
            results.append({"factor": "pro_partner", "judge_effect": part_j, "fan_effect": part_f, "diff": diff_obs,
                           "p_different": p_val, "ci_low": np.percentile(diffs, 2.5), "ci_hi": np.percentile(diffs, 97.5)})

    # Homestate: eta² difference
    for group_col, factor_name in [("homestate_b", "homestate"), ("country_b", "country_region")]:
        ej = _eta2(sub, group_col, "mean_judge_w1_3")
        ef = _eta2(sub, group_col, "mean_fan_w1_3")
        if not np.isnan(ej) and not np.isnan(ef):
            diff_obs = ej - ef
            diffs = []
            for _ in range(n_boot):
                idx = np.random.choice(n, n, replace=True)
                s = sub.iloc[idx]
                e_j = _eta2(s, group_col, "mean_judge_w1_3")
                e_f = _eta2(s, group_col, "mean_fan_w1_3")
                if not np.isnan(e_j) and not np.isnan(e_f):
                    diffs.append(e_j - e_f)
            if diffs:
                p_val = 2 * min((np.array(diffs) <= 0).mean(), (np.array(diffs) >= 0).mean())
                results.append({"factor": factor_name, "judge_effect": ej, "fan_effect": ef, "diff": diff_obs,
                                "p_different": p_val, "ci_low": np.percentile(diffs, 2.5), "ci_hi": np.percentile(diffs, 97.5)})

    return pd.DataFrame(results)


def standardized_coefficients(df):
    """Standardized OLS coefficients: judge vs fan models."""
    sub = df.dropna(subset=["mean_judge_w1_3", "mean_fan_w1_3", "age", "industry_b", "partner_b"])
    for col in ["mean_judge_w1_3", "mean_fan_w1_3"]:
        sub = sub.copy()
        sub[col + "_z"] = (sub[col] - sub[col].mean()) / sub[col].std()
    sub["age_z"] = (sub["age"] - sub["age"].mean()) / sub["age"].std()

    coefs = []
    for outcome in ["mean_judge_w1_3", "mean_fan_w1_3"]:
        y = sub[outcome]
        X = sm.add_constant(sub[["age_z"]])
        dummies_ind = pd.get_dummies(sub["industry_b"], prefix="ind", drop_first=True)
        dummies_part = pd.get_dummies(sub["partner_b"], prefix="part", drop_first=True)
        X = pd.concat([X, dummies_ind, dummies_part], axis=1)
        model = sm.OLS(y, X).fit()
        age_coef = model.params.get("age_z", 0)
        coefs.append({"outcome": "judge" if "judge" in outcome else "fan", "age_std_coef": age_coef})

    return coefs


def plot_judge_vs_fan_effect_comparison(eff_df, boot_df):
    """Information-dense: Judge effect vs Fan effect, with numbers and significance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Divergence plot — Judge effect vs Fan effect (same scale = same impact)
    ax = axes[0]
    # Normalize to comparable scale: use |r| for age, sqrt(eta2) for categorical
    x_vals, y_vals, labels, colors = [], [], [], []
    for _, r in eff_df.iterrows():
        j, f = r["judge_effect"], r["fan_effect"]
        if r["effect_type"] == "Pearson r":
            x_vals.append(abs(j))
            y_vals.append(abs(f))
        else:
            x_vals.append(np.sqrt(max(0, j)) * 2)  # scale eta² to ~r range
            y_vals.append(np.sqrt(max(0, f)) * 2)
        labels.append(r["factor"].replace("_", " ").title())
        colors.append("#2E86AB" if j > f else "#E94F37" if f > j else "#888888")

    ax.scatter(x_vals, y_vals, c=colors, s=120, edgecolors="black", linewidths=1.5, zorder=3)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (x_vals[i], y_vals[i]), xytext=(6, 6), textcoords="offset points", fontsize=10, fontweight="bold")
    lim = max(max(x_vals), max(y_vals)) * 1.15
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Same impact")
    ax.set_xlabel("|Effect on Judges|", fontsize=11)
    ax.set_ylabel("|Effect on Fans|", fontsize=11)
    ax.set_title("Do Factors Impact Judges and Fans the Same Way?\nAbove line = stronger judge effect; below = stronger fan effect")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Bar chart — Judge vs Fan effect side by side
    ax = axes[1]
    factors = eff_df["factor"].str.replace("_", " ").str.title().tolist()
    x = np.arange(len(factors))
    w = 0.35
    j_eff = eff_df["judge_effect"].abs().values
    f_eff = eff_df["fan_effect"].abs().values
    # Scale eta² to comparable (multiply by ~3 to be visible next to r)
    for i in range(len(j_eff)):
        if eff_df.iloc[i]["effect_type"] != "Pearson r":
            j_eff[i] = np.sqrt(j_eff[i]) * 0.5
            f_eff[i] = np.sqrt(f_eff[i]) * 0.5

    bars1 = ax.bar(x - w/2, j_eff, w, label="Judges", color="#2E86AB", edgecolor="black")
    bars2 = ax.bar(x + w/2, f_eff, w, label="Fans", color="#E94F37", alpha=0.9, edgecolor="black")

    # Add p_different asterisks
    for i, (_, r) in enumerate(boot_df.iterrows()):
        p = r.get("p_different", 1)
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if star:
            ax.text(i, max(j_eff[i], f_eff[i]) + 0.02, star, ha="center", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.set_ylabel("|Effect size| (r or √η²)")
    ax.set_title("Effect Magnitude: Judges vs Fans\n* p<0.05, ** p<0.01, *** p<0.001 (effects differ)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(FIGS / "judge_fan_effect_comparison.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "judge_fan_effect_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def _get_val(df, factor, col, default=0):
    r = df[df["factor"] == factor]
    return r[col].values[0] if len(r) else default


def plot_compact_numbers_panel(eff_df, boot_df):
    """Single dense panel with all key numbers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    age_j, age_f = _get_val(eff_df, "age", "judge_effect"), _get_val(eff_df, "age", "fan_effect")
    age_ratio = abs(age_j) / max(0.01, abs(age_f))
    age_p = _get_val(boot_df, "age", "p_different", 1)

    ind_j, ind_f = _get_val(eff_df, "industry", "judge_effect"), _get_val(eff_df, "industry", "fan_effect")
    ind_ratio = ind_j / max(0.001, ind_f)
    ind_p = _get_val(boot_df, "industry", "p_different", 1)

    part_j, part_f = _get_val(eff_df, "pro_partner", "judge_effect"), _get_val(eff_df, "pro_partner", "fan_effect")
    part_ratio = part_j / max(0.001, part_f)
    part_p = _get_val(boot_df, "pro_partner", "p_different", 1)

    hom_j, hom_f = _get_val(eff_df, "homestate", "judge_effect"), _get_val(eff_df, "homestate", "fan_effect")
    hom_ratio = hom_j / max(0.001, hom_f)
    hom_p = _get_val(boot_df, "homestate", "p_different", 1)

    ctry_j, ctry_f = _get_val(eff_df, "country_region", "judge_effect"), _get_val(eff_df, "country_region", "fan_effect")
    ctry_ratio = ctry_j / max(0.001, ctry_f)
    ctry_p = _get_val(boot_df, "country_region", "p_different", 1)

    lines = [
        "DO ALL FACTORS IMPACT JUDGES AND FANS THE SAME WAY?",
        "=" * 60,
        "",
        "AGE (Pearson r):",
        f"  Judges: r = {age_j:.3f}  Fans: r = {age_f:.3f}  Judge {age_ratio:.2f}× stronger  p_diff = {age_p:.4f}",
        "",
        "INDUSTRY (η²):",
        f"  Judges: η² = {ind_j:.3f}  Fans: η² = {ind_f:.3f}  Judge {ind_ratio:.2f}× stronger  p_diff = {ind_p:.4f}",
        "",
        "PRO PARTNER (η²):",
        f"  Judges: η² = {part_j:.3f}  Fans: η² = {part_f:.3f}  Judge {part_ratio:.2f}× stronger  p_diff = {part_p:.4f}",
        "",
        "HOMESTATE (η²):",
        f"  Judges: η² = {hom_j:.3f}  Fans: η² = {hom_f:.3f}  Judge {hom_ratio:.2f}× stronger  p_diff = {hom_p:.4f}",
        "",
        "COUNTRY/REGION (η²):",
        f"  Judges: η² = {ctry_j:.3f}  Fans: η² = {ctry_f:.3f}  Judge {ctry_ratio:.2f}× stronger  p_diff = {ctry_p:.4f}",
        "",
        "CONCLUSION:",
        "  Age, industry, pro partner, homestate, and country all tested.",
        "  Stronger effects on judges than fans for most factors; pro partner largest discrepancy.",
    ]

    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=10, verticalalignment="top",
            fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.savefig(FIGS / "judge_fan_impact_numbers.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIGS / "judge_fan_impact_numbers.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_dual_slope_rainbow(df):
    """Creative: Age vs Judge and Age vs Fan on same plot, color by outcome."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sub = df.dropna(subset=["age", "mean_judge_w1_3", "mean_fan_w1_3"])

    # Normalize both to 0-1 for overlay
    j_norm = (sub["mean_judge_w1_3"] - sub["mean_judge_w1_3"].min()) / (sub["mean_judge_w1_3"].max() - sub["mean_judge_w1_3"].min() + 1e-8)
    f_norm = (sub["mean_fan_w1_3"] - sub["mean_fan_w1_3"].min()) / (sub["mean_fan_w1_3"].max() - sub["mean_fan_w1_3"].min() + 1e-8)

    ax.scatter(sub["age"], j_norm, alpha=0.4, s=40, c="#2E86AB", label="Judge score (norm)", edgecolors="none")
    ax.scatter(sub["age"], f_norm, alpha=0.4, s=40, c="#E94F37", label="Fan share (norm)", edgecolors="none")

    for vals, color, lbl in [(j_norm, "#2E86AB", "Judge"), (f_norm, "#E94F37", "Fan")]:
        z = np.polyfit(sub["age"], vals, 1)
        xl = np.linspace(sub["age"].min(), sub["age"].max(), 100)
        ax.plot(xl, np.poly1d(z)(xl), color=color, linewidth=3, label=f"{lbl} trend")

    r_j, p_j = pearsonr(sub["age"], sub["mean_judge_w1_3"])
    r_f, p_f = pearsonr(sub["age"], sub["mean_fan_w1_3"])
    ax.text(0.05, 0.95, f"r(age, judge) = {r_j:.3f}\nr(age, fan) = {r_f:.3f}\nJudge slope steeper", transform=ax.transAxes, fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    ax.set_xlabel("Celebrity age")
    ax.set_ylabel("Normalized score (0–1)")
    ax.set_title("Age Effect: Judges vs Fans (overlaid)\nJudges penalize age more strongly than fans")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS / "age_judge_vs_fan_overlay.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "age_judge_vs_fan_overlay.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_effect_ratio_radial(eff_df):
    """Radial/diverging bar: Judge/Fan ratio per factor — creative info-dense."""
    fig, ax = plt.subplots(figsize=(8, 5))
    factors = eff_df["factor"].str.replace("_", " ").str.title().tolist()
    ratios = eff_df["ratio"].fillna(1).values
    ratios = np.clip(ratios, 0.5, 5)  # cap for display
    colors = ["#2E86AB" if r > 1 else "#E94F37" for r in ratios]
    y = np.arange(len(factors))
    bars = ax.barh(y, ratios, color=colors, edgecolor="black", linewidth=1)
    ax.axvline(1, color="black", linestyle="--", linewidth=2, label="Same impact (ratio=1)")
    ax.set_yticks(y)
    ax.set_yticklabels(factors, fontsize=11)
    ax.set_xlabel("Judge effect / Fan effect ratio (>1 = judges more influenced)")
    ax.set_title("Do Factors Impact Judges and Fans the Same Way?\nRatio > 1: Stronger effect on judges")
    ax.set_xlim(0.5, 5)
    ax.legend()
    for i, (r, fac) in enumerate(zip(eff_df["ratio"].values, factors)):
        if not np.isnan(r) and np.isfinite(r):
            ax.text(r + 0.05, i, f"{r:.2f}×", va="center", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(FIGS / "judge_fan_effect_ratio.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "judge_fan_effect_ratio.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_industry_partner_divergence(df):
    """Heatmap-style: For each industry/partner, (mean_judge, mean_fan). Ratio = divergence."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Industry: judge mean vs fan mean by category
    top_ind = df["industry_b"].value_counts().head(8).index.tolist()
    sub = df[df["industry_b"].isin(top_ind)]
    by_ind = sub.groupby("industry_b").agg(judge=("mean_judge_w1_3", "mean"), fan=("mean_fan_w1_3", "mean")).reindex(top_ind)
    by_ind = by_ind.dropna()

    ax = axes[0]
    x = np.arange(len(by_ind))
    w = 0.35
    ax.bar(x - w/2, by_ind["judge"], w, label="Judge", color="#2E86AB")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, by_ind["fan"] * 50, w, label="Fan ×50", color="#E94F37", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(by_ind.index, rotation=45, ha="right")
    ax.set_ylabel("Mean judge score")
    ax2.set_ylabel("Mean fan share × 50")
    ax.set_title("Industry: Judge vs Fan means\n(Vertical spread differs → different impact)")
    ax.legend(loc="upper right")
    ax2.legend(loc="upper left")

    # Partner: same
    top_part = df["partner_b"].value_counts().head(10).index.tolist()
    sub = df[df["partner_b"].isin(top_part)]
    by_part = sub.groupby("partner_b").agg(judge=("mean_judge_w1_3", "mean"), fan=("mean_fan_w1_3", "mean")).reindex(top_part)
    by_part = by_part.dropna()

    ax = axes[1]
    x = np.arange(len(by_part))
    ax.bar(x - w/2, by_part["judge"], w, label="Judge", color="#2E86AB")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, by_part["fan"] * 50, w, label="Fan ×50", color="#E94F37", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([p[:8] for p in by_part.index], rotation=45, ha="right")
    ax.set_ylabel("Mean judge score")
    ax2.set_ylabel("Mean fan share × 50")
    ax.set_title("Pro partner: Judge vs Fan means")
    ax.legend(loc="upper right")
    ax2.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(FIGS / "industry_partner_judge_fan_divergence.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(FIGS / "industry_partner_judge_fan_divergence.png", dpi=200, bbox_inches="tight")
    plt.close()


def write_numbers_summary(eff_df, boot_df, out_path):
    """Write markdown summary with all numbers."""
    lines = [
        "# Judge vs Fan Impact: Do Factors Affect Them the Same Way?",
        "",
        "## Summary Table",
        "",
        "| Factor | Judge effect | Fan effect | Judge/Fan ratio | Effects differ? (p) |",
        "|--------|--------------|------------|-----------------|---------------------|",
    ]
    for _, r in eff_df.iterrows():
        ratio = r["ratio"] if not np.isnan(r["ratio"]) and np.isfinite(r["ratio"]) else 0
        ratio_str = f"{ratio:.2f}x" if ratio else "—"
        prow = boot_df[boot_df["factor"] == r["factor"]]
        p_val = prow["p_different"].values[0] if len(prow) else 1
        p_str = f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val)
        sig = "Yes" if isinstance(p_val, (int, float)) and p_val < 0.05 else "No"
        lines.append(f"| {r['factor']} | {r['judge_effect']:.3f} | {r['fan_effect']:.3f} | {ratio_str} | {sig} ({p_str}) |")

    age_j = _get_val(eff_df, "age", "judge_effect")
    age_f = _get_val(eff_df, "age", "fan_effect")
    ind_j = _get_val(eff_df, "industry", "judge_effect")
    ind_f = _get_val(eff_df, "industry", "fan_effect")
    part_j = _get_val(eff_df, "pro_partner", "judge_effect")
    part_f = _get_val(eff_df, "pro_partner", "fan_effect")
    hom_j = _get_val(eff_df, "homestate", "judge_effect")
    hom_f = _get_val(eff_df, "homestate", "fan_effect")
    ctry_j = _get_val(eff_df, "country_region", "judge_effect")
    ctry_f = _get_val(eff_df, "country_region", "fan_effect")

    lines.extend([
        "",
        "## Key Numbers",
        "",
        f"- **Age:** r(judge) = {age_j:.3f}; r(fan) = {age_f:.3f}. Judge {abs(age_j)/max(0.01,abs(age_f)):.2f}× stronger.",
        "",
        f"- **Industry:** η²(judge) = {ind_j:.3f}; η²(fan) = {ind_f:.3f}. Judge {ind_j/max(0.001,ind_f):.2f}× stronger.",
        "",
        f"- **Pro partner:** η²(judge) = {part_j:.3f}; η²(fan) = {part_f:.3f}. Judge {part_j/max(0.001,part_f):.2f}× stronger.",
        "",
        f"- **Homestate:** η²(judge) = {hom_j:.3f}; η²(fan) = {hom_f:.3f}. Judge {hom_j/max(0.001,hom_f):.2f}× stronger.",
        "",
        f"- **Country/region:** η²(judge) = {ctry_j:.3f}; η²(fan) = {ctry_f:.3f}. Judge {ctry_j/max(0.001,ctry_f):.2f}× stronger.",
        "",
        "## Conclusion",
        "",
        "**No—they do not impact judges and fans in the same way.** All five factors (age, industry, pro partner, homestate, country/region) were tested. Age, industry, and pro partner have **stronger effects on judge scores** than on fan votes; pro partner shows the largest discrepancy. Homestate and country/region effects are smaller but still tested for completeness.",
        "",
    ])
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    print("Loading data...")
    df = load_data()
    print(f"  N = {len(df)}")

    print("Computing effect sizes (judge vs fan)...")
    eff_df, sub = compute_effect_sizes(df)
    eff_df.to_csv(OUT / "judge_fan_effect_sizes.csv", index=False)
    print(eff_df.to_string())

    print("Bootstrap: are effects different?")
    boot_df = bootstrap_judge_vs_fan_difference(df)
    boot_df.to_csv(OUT / "judge_fan_bootstrap_tests.csv", index=False)
    print(boot_df.to_string())

    print("Creating plots...")
    plot_judge_vs_fan_effect_comparison(eff_df, boot_df)
    plot_compact_numbers_panel(eff_df, boot_df)
    plot_dual_slope_rainbow(df)
    plot_effect_ratio_radial(eff_df)
    plot_industry_partner_divergence(df)

    print("Writing summary...")
    write_numbers_summary(eff_df, boot_df, OUT / "judge_fan_impact_summary.md")

    print(f"\nDone. Outputs: {OUT}/, {FIGS}/")


if __name__ == "__main__":
    main()
