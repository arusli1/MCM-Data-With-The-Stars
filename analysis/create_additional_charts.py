import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "results"

def plot_diverging_bars(data, x_col, y_col, title, filename, xlabel="Impact"):
    """Create diverging bar chart with green (positive) and red (negative)"""
    plt.figure(figsize=(12, 10))
    
    # Color based on sign
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in data[x_col]]
    
    sns.barplot(
        data=data, x=x_col, y=y_col,
        hue=data[x_col] > 0, 
        palette={True: '#2ca02c', False: '#d62728'}, 
        legend=False
    )
    
    plt.axvline(0, color='black', linewidth=1)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"Created: {filename}")

# Process each target
targets = {
    'Judge_Appeal_Avg_Score': 'Judge Appeal',
    'Fan_Appeal_Avg_Share': 'Fan Appeal'
}

for file_suffix, display_name in targets.items():
    # Load SHAP importance file
    csv_path = f"{OUTPUT_DIR}/shap_importance_{file_suffix}.csv"
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping...")
        continue
    
    df = pd.read_csv(csv_path)
    
    # 1. ALL VARIABLES CHART (Top 15)
    top_all = df.head(15).copy()
    plot_diverging_bars(
        top_all,
        'signed_shap',
        'feature_clean',
        f"Top Drivers of {display_name}",
        f"shap_{file_suffix}_all_variables.png",
        "Impact (Left=Hurts, Right=Helps)"
    )
    
    # 2. PARTNERS ONLY CHART
    partners_only = df[df['feature'].str.contains('partner_clean')].copy()
    if not partners_only.empty:
        plot_diverging_bars(
            partners_only.head(20),
            'signed_shap',
            'feature_clean',
            f"Partner Impact on {display_name}",
            f"shap_{file_suffix}_partners_only.png",
            "Impact (Left=Hurts, Right=Helps)"
        )
    
    # 3. INDUSTRY ONLY CHART
    industry_only = df[df['feature'].str.contains('industry_clean')].copy()
    if not industry_only.empty:
        plot_diverging_bars(
            industry_only,
            'signed_shap',
            'feature_clean',
            f"Industry Impact on {display_name}",
            f"shap_{file_suffix}_industry_only.png",
            "Impact (Left=Hurts, Right=Helps)"
        )

print("\nAll additional charts created successfully!")
