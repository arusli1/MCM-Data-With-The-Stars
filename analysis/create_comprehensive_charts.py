import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = "results"

def plot_diverging_bars(data, x_col, y_col, title, filename, xlabel="Impact"):
    """Create diverging bar chart with green (positive) and red (negative)"""
    if data.empty:
        print(f"Skipping {filename} - no data")
        return
        
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
    print(f"✓ Created: {filename}")

# Process each target
targets = {
    'Success_Placement': 'Success',
    'Judge_Appeal_Avg_Score': 'Judge Appeal',
    'Fan_Appeal_Avg_Share': 'Fan Appeal'
}

for file_suffix, display_name in targets.items():
    print(f"\n{'='*60}")
    print(f"Processing: {display_name}")
    print(f"{'='*60}")
    
    # Load SHAP importance file
    csv_path = f"{OUTPUT_DIR}/shap_importance_{file_suffix}.csv"
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, skipping...")
        continue
    
    df = pd.read_csv(csv_path)
    
    # 1. POSITIVE DRIVERS ONLY
    positive = df[df['signed_shap'] > 0].copy()
    if not positive.empty:
        plot_diverging_bars(
            positive.head(20),
            'signed_shap',
            'feature_clean',
            f"Positive Drivers of {display_name}",
            f"{file_suffix}_positive_only.png",
            "Positive Impact (Helps)"
        )
    
    # 2. NEGATIVE DRIVERS ONLY
    negative = df[df['signed_shap'] < 0].copy()
    if not negative.empty:
        plot_diverging_bars(
            negative.head(20),
            'signed_shap',
            'feature_clean',
            f"Negative Drivers of {display_name}",
            f"{file_suffix}_negative_only.png",
            "Negative Impact (Hurts)"
        )
    
    # 3. BALLROOM PARTNER RANKING
    partners = df[df['feature'].str.contains('partner_clean', case=False)].copy()
    if not partners.empty:
        plot_diverging_bars(
            partners,
            'signed_shap',
            'feature_clean',
            f"Ballroom Partner Rankings - {display_name}",
            f"{file_suffix}_partner_rankings.png",
            "Impact (Left=Hurts, Right=Helps)"
        )
    
    # 4. CELEBRITY INDUSTRY RANKING
    industry = df[df['feature'].str.contains('industry_clean', case=False)].copy()
    if not industry.empty:
        plot_diverging_bars(
            industry,
            'signed_shap',
            'feature_clean',
            f"Celebrity Industry Rankings - {display_name}",
            f"{file_suffix}_industry_rankings.png",
            "Impact (Left=Hurts, Right=Helps)"
        )
    
    # 5. CELEBRITY HOME STATE RANKING (if exists)
    state = df[df['feature'].str.contains('celebrity_homestate', case=False)].copy()
    if not state.empty:
        # Clean labels
        state['feature_clean'] = state['feature'].str.replace('celebrity_homestate_', 'State: ')
        plot_diverging_bars(
            state.head(20),  # Top 20 states
            'signed_shap',
            'feature_clean',
            f"Home State Rankings - {display_name}",
            f"{file_suffix}_state_rankings.png",
            "Impact (Left=Hurts, Right=Helps)"
        )
    
    # 6. CELEBRITY HOME COUNTRY/REGION RANKING (if exists)
    country = df[df['feature'].str.contains('celebrity_homecountry', case=False)].copy()
    if not country.empty:
        # Clean labels
        country['feature_clean'] = country['feature'].str.replace('celebrity_homecountry/region_', 'Country: ')
        plot_diverging_bars(
            country,
            'signed_shap',
            'feature_clean',
            f"Home Country/Region Rankings - {display_name}",
            f"{file_suffix}_country_rankings.png",
            "Impact (Left=Hurts, Right=Helps)"
        )
    
    # 7. AGE (if exists as a feature)
    age = df[df['feature'].str.contains('^age$', case=False, regex=True)].copy()
    if not age.empty:
        print(f"  Age impact: {age['signed_shap'].values[0]:.6f}")

print("\n" + "="*60)
print("All charts created successfully!")
print("="*60)
