"""
run_analysis.py

Fixes applied:
- Avoids target leakage for SUCCESS: success model uses ONLY static traits by default.
  (Judge/Fan appeal models also use static traits — which matches the "what traits drive X" question.)
- Robust SHAP handling (list outputs) + signed SHAP uses mean(SHAP) (NOT corr hack).
- Left join on fan votes to avoid selection bias; missing fan shares imputed (and flagged).
- Safer placement normalization (no divide-by-zero).
- Buckets high-cardinality geography to top-N + "Other".
- Cleaner, stable diverging bar charts using matplotlib (no hue quirks).
- Consistent feature cleaning for display.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import shap

# -------------------------
# Configuration
# -------------------------
DATA_PATH = "../Data/2026_MCM_Problem_C_Data.csv"
FAN_VOTES_PATH = "../AR-Problem1-Base/final_results/base_inferred_shares.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Category bucketing (reduce sparse one-hot explosion)
TOP_INDUSTRIES = 6
TOP_PARTNERS = 15
TOP_STATES = 15
TOP_COUNTRIES = 15

# Model settings
RF_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# If True, you will ALSO allow judge/fan aggregates as features for SUCCESS.
# This is usually leakage for “drivers of success” unless you explicitly want
# a retrospective explanation using full-season info.
ALLOW_LEAKY_FEATURES_FOR_SUCCESS = False


# -------------------------
# Helpers
# -------------------------
def safe_spearman(a, b):
    # Spearman without scipy
    ra = pd.Series(a).rank(method="average")
    rb = pd.Series(b).rank(method="average")
    return np.corrcoef(ra, rb)[0, 1]


def bucket_top_n(series: pd.Series, top_n: int, other_label: str):
    vc = series.fillna("Unknown").astype(str).value_counts()
    top = set(vc.nlargest(top_n).index)
    return series.fillna("Unknown").astype(str).apply(lambda x: x if x in top else other_label)


def plot_diverging_barh(df_plot, x_col, y_col, title, filename, xlabel="Impact"):
    # Sort so most negative at bottom, most positive at top (nice diverging view)
    df_plot = df_plot.copy().sort_values(x_col)

    vals = df_plot[x_col].values
    labels = df_plot[y_col].values
    colors = np.where(vals >= 0, "#2ca02c", "#d62728")

    plt.figure(figsize=(12, max(6, 0.35 * len(df_plot))))
    plt.barh(labels, vals, color=colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def clean_feature_name(s: str) -> str:
    return (
        s.replace("partner_clean_", "Partner: ")
         .replace("industry_clean_", "Industry: ")
         .replace("homestate_clean_", "State: ")
         .replace("homecountry_clean_", "Country: ")
         .replace("missing_fan_share_", "Missing Fan Share: ")
    )


# -------------------------
# Load + preprocess
# -------------------------
def load_and_preprocess():
    print("Loading data...")
    df_main = pd.read_csv(DATA_PATH)
    df_fan = pd.read_csv(FAN_VOTES_PATH)

    # --- Avg judge score per contestant across all available weeks ---
    score_cols = [c for c in df_main.columns if ("score" in c and "judge" in c)]

    def get_avg_judge_score(row):
        scores = pd.to_numeric(row[score_cols], errors="coerce")
        # Treat zeros as "not performed / eliminated"
        scores = scores[scores > 0]
        return float(scores.mean()) if len(scores) else np.nan

    df_main["avg_judge_score"] = df_main.apply(get_avg_judge_score, axis=1)

    # --- Fan shares: aggregate by season+celebrity ---
    df_fan_agg = (
        df_fan.groupby(["season", "celebrity_name"], as_index=False)["s_share"]
        .mean()
        .rename(columns={"s_share": "avg_fan_share"})
    )

    # --- Merge (LEFT JOIN to avoid selection bias) ---
    df = pd.merge(df_main, df_fan_agg, on=["season", "celebrity_name"], how="left")

    # Missing fan share flag + impute
    df["missing_fan_share"] = df["avg_fan_share"].isna().astype(int)
    if df["avg_fan_share"].notna().any():
        df["avg_fan_share"] = df["avg_fan_share"].fillna(df["avg_fan_share"].mean())
    else:
        df["avg_fan_share"] = 0.0  # extreme fallback

    print(f"Merged Data Shape (left join): {df.shape}")
    return df


def feature_engineering(df):
    print("Engineering features...")

    # Placement target
    df["placement"] = pd.to_numeric(df["placement"], errors="coerce")

    # Normalize placement by season size (safe)
    season_max = df.groupby("season")["placement"].transform("max")
    denom = (season_max - 1).replace(0, np.nan)
    df["placement_norm"] = (df["placement"] - 1) / denom
    df["success_score"] = 1 - df["placement_norm"]

    # Age
    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median())

    # Bucket categories to top-N + other
    df["industry_clean"] = bucket_top_n(df["celebrity_industry"], TOP_INDUSTRIES, "Other")
    df["partner_clean"] = bucket_top_n(df["ballroom_partner"], TOP_PARTNERS, "Other_Partner")
    df["homestate_clean"] = bucket_top_n(df["celebrity_homestate"], TOP_STATES, "Other_State")
    df["homecountry_clean"] = bucket_top_n(df["celebrity_homecountry/region"], TOP_COUNTRIES, "Other_Country")

    return df


def make_design_matrix(df, include_leaky_for_success: bool):
    """
    Base features: static traits + missing fan share flag.
    Optionally include avg_judge_score and avg_fan_share as features for SUCCESS
    (generally leaky for 'drivers of success', but left as an explicit switch).
    """
    base_cols = [
        "age",
        "industry_clean",
        "partner_clean",
        "homestate_clean",
        "homecountry_clean",
        "missing_fan_share"
    ]

    X_raw = df[base_cols].copy()

    if include_leaky_for_success:
        X_raw["avg_judge_score"] = df["avg_judge_score"]
        X_raw["avg_fan_share"] = df["avg_fan_share"]

    X = pd.get_dummies(X_raw, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X


# -------------------------
# Train + explain
# -------------------------
def shap_importance_table(model, X: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Robust handling for possible list output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)  # (n, p)

    mean_abs = np.abs(shap_values).mean(axis=0)
    signed = shap_values.mean(axis=0)

    out = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
        "signed_shap": signed
    }).sort_values("mean_abs_shap", ascending=False)

    out["feature_clean"] = out["feature"].apply(clean_feature_name)
    return out


def train_and_explain(df):
    print("Training models...")

    targets = {
        "Success (Placement)": "success_score",
        "Judge Appeal (Avg Score)": "avg_judge_score",
        "Fan Appeal (Avg Share)": "avg_fan_share",
    }

    results = {}

    groups = df["season"]
    logo = LeaveOneGroupOut()

    for target_name, target_col in targets.items():
        print(f"\nAnalyzing {target_name}...")

        # Choose design matrix
        include_leaky = (target_name == "Success (Placement)" and ALLOW_LEAKY_FEATURES_FOR_SUCCESS)
        X = make_design_matrix(df, include_leaky_for_success=include_leaky)

        y = pd.to_numeric(df[target_col], errors="coerce")
        y = y.replace([np.inf, -np.inf], np.nan)
        y = y.fillna(y.mean())

        model = RandomForestRegressor(**RF_PARAMS)

        # LOGO CV predictions
        preds = cross_val_predict(model, X, y, cv=logo, groups=groups)
        pearson = np.corrcoef(y, preds)[0, 1]
        spearman = safe_spearman(y, preds)
        r2 = r2_score(y, preds)

        print(f"  LOGO-CV Pearson r:  {pearson:.3f}")
        print(f"  LOGO-CV Spearman ρ: {spearman:.3f}")
        print(f"  LOGO-CV R²:         {r2:.3f}")

        # Fit full for explanation (SHAP)
        model.fit(X, y)
        imp = shap_importance_table(model, X)

        safe_name = (
            target_name.replace(" ", "_")
                       .replace("(", "")
                       .replace(")", "")
                       .replace("/", "_")
        )
        imp.to_csv(os.path.join(OUTPUT_DIR, f"shap_importance_{safe_name}.csv"), index=False)

        # Top 15 overall diverging plot using signed mean SHAP
        top15 = imp.head(15).copy()
        plot_diverging_barh(
            top15,
            x_col="signed_shap",
            y_col="feature_clean",
            title=f"Top Drivers of {target_name} (Signed Mean SHAP)",
            filename=f"shap_{safe_name}.png",
            xlabel="Impact on prediction (negative hurts, positive helps)"
        )

        # Partner-only plot (top 20 partners by magnitude)
        partner_only = imp[imp["feature"].str.startswith("partner_clean_")].copy()
        if not partner_only.empty:
            partner_only = partner_only.head(20)
            plot_diverging_barh(
                partner_only,
                x_col="signed_shap",
                y_col="feature_clean",
                title=f"Partner Impact on {target_name} (Signed Mean SHAP)",
                filename=f"shap_{safe_name}_partners.png",
                xlabel="Impact on prediction (negative hurts, positive helps)"
            )

        results[target_name] = imp

    # Combined comparison table across targets
    all_features = sorted(set().union(*[set(r["feature"]) for r in results.values()]))
    comp = pd.DataFrame({"feature": all_features})
    comp["feature_clean"] = comp["feature"].apply(clean_feature_name)

    for target_name, imp_df in results.items():
        safe_name = (
            target_name.replace(" ", "_")
                       .replace("(", "")
                       .replace(")", "")
                       .replace("/", "_")
        )
        comp[f"signed_{safe_name}"] = comp["feature"].map(imp_df.set_index("feature")["signed_shap"]).fillna(0.0)
        comp[f"abs_{safe_name}"] = comp["feature"].map(imp_df.set_index("feature")["mean_abs_shap"]).fillna(0.0)

    comp.to_csv(os.path.join(OUTPUT_DIR, "shap_comparison_signed_and_abs.csv"), index=False)
    return results


def write_summary(df, results):
    path = os.path.join(OUTPUT_DIR, "analysis_summary.md")
    with open(path, "w") as f:
        f.write("# DWTS Factor Analysis Summary (Fixed)\n\n")
        f.write(f"- Contestants: {len(df)}\n")
        f.write(f"- Seasons: {df['season'].nunique()}\n")
        f.write(f"- Leakage for Success enabled? {ALLOW_LEAKY_FEATURES_FOR_SUCCESS}\n\n")

        for target, imp_df in results.items():
            f.write(f"## Drivers of {target}\n\n")
            f.write("Top 10 features by mean absolute SHAP:\n\n")
            f.write(imp_df[["feature_clean", "signed_shap", "mean_abs_shap"]].head(10).to_markdown(index=False))
            f.write("\n\n")

    print(f"Summary written: {path}")


def main():
    df = load_and_preprocess()
    df = feature_engineering(df)

    # Drop rows with missing critical identifiers
    df = df.dropna(subset=["season", "celebrity_name"])

    results = train_and_explain(df)
    write_summary(df, results)

    print("\nDone. Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
