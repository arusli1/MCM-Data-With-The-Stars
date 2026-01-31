import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import shap
import os

# --- Configuration ---
DATA_PATH = "../Data/2026_MCM_Problem_C_Data.csv"
FAN_VOTES_PATH = "../AR-Problem1-Base/final_results/base_inferred_shares.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess():
    print("Loading data...")
    df_main = pd.read_csv(DATA_PATH)
    df_fan = pd.read_csv(FAN_VOTES_PATH)

    # Calculate Average Judge Score per Celebrity per Season
    score_cols = [c for c in df_main.columns if 'score' in c and 'judge' in c]
    
    def get_avg_judge_score(row):
        scores = pd.to_numeric(row[score_cols], errors='coerce')
        scores = scores[scores > 0]  # Exclude 0s (eliminated)
        return scores.mean()

    df_main['avg_judge_score'] = df_main.apply(get_avg_judge_score, axis=1)

    # Process Fan Votes
    df_fan_agg = df_fan.groupby(['season', 'celebrity_name'])['s_share'].mean().reset_index()
    df_fan_agg.rename(columns={'s_share': 'avg_fan_share'}, inplace=True)

    # Merge
    df_merged = pd.merge(df_main, df_fan_agg, on=['season', 'celebrity_name'], how='inner')
    
    print(f"Merged Data Shape: {df_merged.shape}")
    
    return df_merged

def feature_engineering(df):
    print("Engineering features...")
    
    # Target: Placement
    df['placement'] = pd.to_numeric(df['placement'], errors='coerce')
    
    # Normalize placement by season size
    season_counts = df.groupby('season')['placement'].transform('max')
    df['placement_norm'] = (df['placement'] - 1) / (season_counts - 1)
    df['success_score'] = 1 - df['placement_norm']  # Higher is better

    # Features
    df['age'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce').fillna(df['celebrity_age_during_season'].median())

    # Industry (Top 6)
    top_industries = df['celebrity_industry'].value_counts().nlargest(6).index
    df['industry_clean'] = df['celebrity_industry'].apply(lambda x: x if x in top_industries else 'Other')

    # Pro Partner (Top 15)
    top_partners = df['ballroom_partner'].value_counts().nlargest(15).index
    df['partner_clean'] = df['ballroom_partner'].apply(lambda x: x if x in top_partners else 'Other_Partner')
    
    return df

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

def train_and_explain(df):
    print("Training models...")
    
    targets = {
        'Success (Placement)': 'success_score',
        'Judge Appeal (Avg Score)': 'avg_judge_score',
        'Fan Appeal (Avg Share)': 'avg_fan_share'
    }
    
    # Features - NOW INCLUDING STATE AND COUNTRY
    X_raw = df[['age', 'industry_clean', 'partner_clean', 'celebrity_homestate', 'celebrity_homecountry/region']].copy()
    X = pd.get_dummies(X_raw, drop_first=True)
    X = X.fillna(0)
    
    results = {}
    shap_tables = {}
    
    for target_name, target_col in targets.items():
        print(f"Analyzing {target_name}...")
        y = df[target_col].fillna(df[target_col].mean())
        
        # LOGO CV
        groups = df['season']
        logo = LeaveOneGroupOut()
        
        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        preds = cross_val_predict(model, X, y, cv=logo, groups=groups)
        correlation = np.corrcoef(y, preds)[0, 1]
        print(f"  CV Correlation: {correlation:.3f}")
        
        # Train on full set for SHAP
        model.fit(X, y)
        
        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Calculate SIGNED SHAP values
        vals_abs = np.abs(shap_values).mean(0)
        
        # Determine direction via correlation
        directions = []
        for i, col in enumerate(X.columns):
            corr = np.corrcoef(X[col], shap_values[:, i])[0, 1]
            directions.append(1 if corr > 0 else -1)
        
        signed_vals = vals_abs * np.array(directions)
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': vals_abs,
            'signed_shap': signed_vals
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Clean feature names for display
        feature_importance['feature_clean'] = feature_importance['feature'].str.replace('partner_clean_', 'Partner: ').str.replace('industry_clean_', 'Industry: ')
        
        # Save CSV
        safe_name = target_name.replace(" ", "_").replace("(", "").replace(")", "")
        feature_importance.to_csv(f"{OUTPUT_DIR}/shap_importance_{safe_name}.csv", index=False)
        
        # Create diverging bar chart (Top 15)
        top_features = feature_importance.head(15).copy()
        plot_diverging_bars(
            top_features, 
            'signed_shap', 
            'feature_clean',
            f"Top Drivers of {target_name}",
            f"shap_{safe_name}.png",
            "Impact (Left=Hurts, Right=Helps)"
        )
        
        # Partner-only plot
        partner_features = feature_importance[feature_importance['feature'].str.contains('partner_clean')].copy()
        if not partner_features.empty:
            plot_diverging_bars(
                partner_features.head(20),
                'signed_shap',
                'feature_clean',
                f"Partner Impact on {target_name}",
                f"shap_{safe_name}_partners.png",
                "Impact (Left=Hurts, Right=Helps)"
            )
        
        results[target_name] = feature_importance
        shap_tables[target_name] = feature_importance
        
    # Create combined signed comparison table
    all_feats = sorted(set().union(*[df['feature'] for df in shap_tables.values()]))
    comp = pd.DataFrame({'feature': all_feats})
    
    for target_name, imp_df in shap_tables.items():
        safe_name = target_name.replace(" ", "_").replace("(", "").replace(")", "")
        comp[f'impact_{safe_name}'] = comp['feature'].map(imp_df.set_index('feature')['signed_shap']).fillna(0)
    
    comp['feature_clean'] = comp['feature'].str.replace('partner_clean_', 'Partner: ').str.replace('industry_clean_', 'Industry: ')
    comp.to_csv(f"{OUTPUT_DIR}/shap_comparison_signed.csv", index=False)
    
    return results

def main():
    df = load_and_preprocess()
    df = feature_engineering(df)
    results = train_and_explain(df)
    
    # Write summary
    with open(f"{OUTPUT_DIR}/analysis_summary.md", "w") as f:
        f.write("# DWTS Factor Analysis Summary\n\n")
        f.write(f"Based on analysis of {len(df)} contestants across {df['season'].nunique()} seasons.\n\n")
        
        for target, imp_df in results.items():
            f.write(f"## Drivers of {target}\n\n")
            f.write("### Top 10 Features (by Absolute Impact)\n\n")
            f.write(imp_df[['feature_clean', 'signed_shap', 'mean_abs_shap']].head(10).to_markdown(index=False))
            f.write("\n\n")
            
    print("Done. Results saved to results/")

if __name__ == "__main__":
    main()
