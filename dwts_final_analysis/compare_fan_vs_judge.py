import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compare_drivers():
    print("Loading SHAP importance data...")
    # Load both models
    judge_df = pd.read_csv(f"{OUTPUT_DIR}/shap_importance_Judge_Appeal_Avg_Score.csv")
    fan_df = pd.read_csv(f"{OUTPUT_DIR}/shap_importance_Fan_Appeal_Avg_Share.csv")

    # Rename columns for merging
    judge_df = judge_df.rename(columns={
        "signed_shap": "judge_signed_shap",
        "mean_abs_shap": "judge_abs_shap"
    })
    fan_df = fan_df.rename(columns={
        "signed_shap": "fan_signed_shap",
        "mean_abs_shap": "fan_abs_shap"
    })

    # Merge on feature
    comparison = pd.merge(
        judge_df[["feature", "feature_clean", "judge_signed_shap", "judge_abs_shap"]],
        fan_df[["feature", "fan_signed_shap", "fan_abs_shap"]],
        on="feature"
    )

    # Normalization: Judge Score is ~1-10 (std ~2), Fan Share is ~0-0.3 (std ~0.05)
    # To compare them, we look at Z-scores of the impacts across features
    comparison["judge_impact_norm"] = (comparison["judge_signed_shap"] - comparison["judge_signed_shap"].mean()) / comparison["judge_signed_shap"].std()
    comparison["fan_impact_norm"] = (comparison["fan_signed_shap"] - comparison["fan_signed_shap"].mean()) / comparison["fan_signed_shap"].std()

    # Calculate Discrepancy (Positive = Favored by Judges more than Fans, Negative = Favored by Fans more than Judges)
    comparison["discrepancy"] = comparison["judge_impact_norm"] - comparison["fan_impact_norm"]

    # Save CSV
    comparison.sort_values("discrepancy", ascending=False).to_csv(f"{OUTPUT_DIR}/fan_vs_judge_comparison.csv", index=False)
    print(f"✓ Saved: fan_vs_judge_comparison.csv")

    # Final visualization: Top Discrepancies
    plot_comparison(comparison)

def plot_comparison(df):
    # Select top 10 favoring judges and top 10 favoring fans
    top_judge = df.nlargest(15, "judge_abs_shap").copy() # Focus on important features
    
    # Or just top discrepancy
    diff_df = df.sort_values("discrepancy", ascending=False)
    top_discrepants = pd.concat([diff_df.head(10), diff_df.tail(10)])

    plt.figure(figsize=(14, 10))
    
    # Dual-bar plot comparing normalized impacts
    melted = top_discrepants.melt(
        id_vars=["feature_clean"], 
        value_vars=["judge_impact_norm", "fan_impact_norm"],
        var_name="Audience", value_name="Relative Impact"
    )
    melted["Audience"] = melted["Audience"].map({"judge_impact_norm": "Judges", "fan_impact_norm": "Fans"})

    sns.barplot(
        data=melted, x="Relative Impact", y="feature_clean", hue="Audience",
        palette={"Judges": "#1f77b4", "Fans": "#ff7f0e"}
    )
    
    plt.axvline(0, color='black', linewidth=1)
    plt.title("Where Fans and Judges Disagree\n(Comparison of Normalized Feature Impacts)", fontsize=16, fontweight='bold')
    plt.xlabel("Normalized Impact (Z-Score)", fontsize=12)
    plt.ylabel("")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.legend(title="Audience")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fan_vs_judge_discrepancy.png", dpi=300)
    plt.close()
    print(f"✓ Created: fan_vs_judge_discrepancy.png")

if __name__ == "__main__":
    compare_drivers()
