import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import shap
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Set output directory
OUTPUT_DIR = "ml_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    import re
    name = name.lower().strip()
    name = re.sub(r'[^\w\s]', '', name)
    return name

def compute_slope(y):
    # Least squares slope across weeks 1-3 where at least two points exist
    y = np.array(y)
    mask = ~np.isnan(y)
    x = np.arange(1, len(y) + 1)[mask]
    y = y[mask]
    if len(y) < 2:
        return np.nan
    return np.polyfit(x, y, 1)[0]

def load_data():
    df_main = pd.read_csv("Data/2026_MCM_Problem_C_Data.csv")
    df_votes = pd.read_csv("Data/estimate_votes.csv")
    
    # Normalize names for merging
    df_main['norm_name'] = df_main['celebrity_name'].apply(normalize_name)
    df_votes['norm_name'] = df_votes['celebrity_name'].apply(normalize_name)
    
    # Pre-merge check
    main_contestants = set(zip(df_main['season'], df_main['norm_name']))
    vote_contestants = set(zip(df_votes['season'], df_votes['norm_name']))
    lost = main_contestants - vote_contestants
    print(f"Rows lost in merge (Main but not in Votes): {len(lost)}")
    for s, n in lost:
        print(f"  Season {s}: {n}")

    # Success score: final placement transformed into success_score in [0,1] (1 = winner)
    # Note: higher placement number means lower rank (e.g. 1st is 1, 10th is 10)
    # Score = 1 - (placement - 1) / (max_placement - 1)
    df_main['success_score'] = df_main.groupby('season')['placement'].transform(lambda x: 1 - (x - 1) / (x.max() - 1))
    
    # Process Judge scores (Weeks 1-3)
    # Columns like week{w}_judge{j}_score
    judge_weeks = []
    for w in range(1, 4):
        cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
        # Filter existing columns
        cols = [c for c in cols if c in df_main.columns]
        # Replace 0 and N/A with NaN
        for c in cols:
            df_main[c] = pd.to_numeric(df_main[c], errors='coerce').replace(0, np.nan)
        # Sum of judges for that week (ignoring NaNs)
        df_main[f'week{w}_total_judge'] = df_main[cols].sum(axis=1, min_count=1)
        judge_weeks.append(f'week{w}_total_judge')
        
    df_main['judge_mean_w1_3'] = df_main[judge_weeks].mean(axis=1)
    df_main['judge_week1'] = df_main['week1_total_judge']
    df_main['judge_slope_w1_3'] = df_main[judge_weeks].apply(compute_slope, axis=1)
    
    # Process Fan scores (Weeks 1-3)
    fan_w1_3 = df_votes[df_votes['week'].isin([1, 2, 3])].copy()
    fan_pivot = fan_w1_3.pivot_table(index=['season', 'norm_name'], columns='week', values='s_share').reset_index()
    fan_pivot.columns = ['season', 'norm_name', 'fan_w1', 'fan_w2', 'fan_w3']
    
    df_main = df_main.merge(fan_pivot, on=['season', 'norm_name'], how='left')
    
    fan_cols = ['fan_w1', 'fan_w2', 'fan_w3']
    df_main['fan_mean_w1_3'] = df_main[fan_cols].mean(axis=1)
    df_main['fan_week1'] = df_main['fan_w1']
    df_main['fan_slope_w1_3'] = df_main[fan_cols].apply(compute_slope, axis=1)
    
    # Categorical Bucketing
    def bucket_top_k(df, col, k, other_label='Other'):
        top_k = df[col].value_counts().index[:k]
        df[col] = df[col].apply(lambda x: x if x in top_k else other_label)
        return df

    # Features to include: age, industry (top-K), homestate (top-K), homecountry (top-K), ballroom_partner (top-15)
    df_main['celebrity_age_during_season'] = pd.to_numeric(df_main['celebrity_age_during_season'], errors='coerce')
    
    # Fill missing country/state with 'Unknown' before bucketing
    df_main['celebrity_industry'] = df_main['celebrity_industry'].fillna('Other')
    df_main['celebrity_homestate'] = df_main['celebrity_homestate'].fillna('Unknown')
    df_main['celebrity_homecountry/region'] = df_main['celebrity_homecountry/region'].fillna('Unknown')
    df_main['ballroom_partner'] = df_main['ballroom_partner'].fillna('Unknown_Partner')
    
    df_main = bucket_top_k(df_main, 'celebrity_industry', 10)
    df_main = bucket_top_k(df_main, 'celebrity_homestate', 10)
    df_main = bucket_top_k(df_main, 'celebrity_homecountry/region', 5)
    df_main = bucket_top_k(df_main, 'ballroom_partner', 15, other_label='Other_Partner')
    
    return df_main

def prepare_modeling_data(df, target_col, features):
    data = df.dropna(subset=[target_col]).copy()
    X = data[features].copy()
    y = data[target_col].copy()
    groups = data['season'].values
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Impute numeric
    imputer = SimpleImputer(strategy='median')
    X[numeric_features] = imputer.fit_transform(X[numeric_features])
    
    # One-hot encode categorical
    X = pd.get_dummies(X, columns=categorical_features)
    
    return X, y, groups

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    pearson, _ = pearsonr(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    return mae, rmse, pearson, spearman

def train_and_cv(X, y, groups, model_type='xgb'):
    logo = LeaveOneGroupOut()
    cv_results = []
    all_preds = np.zeros(len(y))
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if model_type == 'xgb':
            # Simplified tuning
            model = XGBRegressor(max_depth=5, n_estimators=400, learning_rate=0.05, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42)
            
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        all_preds[test_idx] = preds
        
        metrics = evaluate_model(y_test, preds)
        cv_results.append(metrics)
        
    # Final model for SHAP (trained on all data)
    final_model = XGBRegressor(max_depth=5, n_estimators=400, learning_rate=0.05, random_state=42) if model_type == 'xgb' else RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42)
    final_model.fit(X, y)
    
    return cv_results, all_preds, final_model

def run_shap_analysis(model, X, target_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary_{target_name}.png")
    plt.close()
    
    # Importance CSV
    importance = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    importance.to_csv(f"{OUTPUT_DIR}/shap_importance_{target_name}.csv", index=False)
    
    # Top 10 Pos/Neg
    # Calculate feature correlations with SHAP values to determine direction
    feature_directions = []
    for i, col in enumerate(X.columns):
        if np.std(X[col]) > 0:
            corr, _ = pearsonr(X[col], shap_values[:, i])
            feature_directions.append(corr)
        else:
            feature_directions.append(0)
    
    importance['direction'] = feature_directions
    
    top_pos = importance[importance['direction'] > 0].head(10)
    top_neg = importance[importance['direction'] < 0].head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_pos, x='mean_abs_shap', y='feature', palette='Greens_r')
    plt.title(f"Top 10 Positive Factors - {target_name}")
    plt.savefig(f"{OUTPUT_DIR}/shap_top10_pos_{target_name}.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_neg, x='mean_abs_shap', y='feature', palette='Reds_r')
    plt.title(f"Top 10 Negative Factors - {target_name}")
    plt.savefig(f"{OUTPUT_DIR}/shap_top10_neg_{target_name}.png")
    plt.close()
    
    # Per-characteristic graphs
    chars = ['ballroom_partner', 'celebrity_industry', 'celebrity_homestate', 'celebrity_homecountry/region', 'celebrity_age_during_season']
    for char in chars:
        # Find all columns related to this characteristic (if OHE)
        related_cols = [c for c in X.columns if c.startswith(char)]
        if not related_cols: continue
        
        char_importance = importance[importance['feature'].isin(related_cols)].sort_values('mean_abs_shap', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=char_importance.head(10), x='mean_abs_shap', y='feature', palette='viridis')
        plt.title(f"Impact of {char} - {target_name}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_feature_{char.replace('/', '_')}_{target_name}.png")
        plt.close()
        
    return importance

def pro_boost_analysis(df, features_no_partner, target_col='success_score'):
    # Fit model without partner features
    X, y, groups = prepare_modeling_data(df, target_col, features_no_partner)
    _, cv_preds, _ = train_and_cv(X, y, groups, model_type='xgb')
    
    data = df.dropna(subset=[target_col]).copy()
    data['predicted_success'] = cv_preds
    data['residual'] = data[target_col] - data['predicted_success']
    
    # Aggregate mean residual by partner
    partner_res = data.groupby('ballroom_partner')['residual'].agg(['mean', 'count']).reset_index()
    
    # Bootstrap CIs
    seasons = data['season'].unique()
    n_boot = 1000
    boot_means = []
    
    for _ in range(n_boot):
        boot_seasons = np.random.choice(seasons, size=len(seasons), replace=True)
        boot_data = pd.concat([data[data['season'] == s] for s in boot_seasons])
        boot_means.append(boot_data.groupby('ballroom_partner')['residual'].mean())
        
    boot_df = pd.DataFrame(boot_means)
    ci_lower = boot_df.quantile(0.025)
    ci_upper = boot_df.quantile(0.975)
    
    partner_res['ci_lower'] = partner_res['ballroom_partner'].map(ci_lower)
    partner_res['ci_upper'] = partner_res['ballroom_partner'].map(ci_upper)
    
    partner_res = partner_res.sort_values('mean', ascending=False)
    partner_res.to_csv(f"{OUTPUT_DIR}/pro_boost_residuals.csv", index=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.errorbar(partner_res['mean'], partner_res['ballroom_partner'], 
                 xerr=[partner_res['mean'] - partner_res['ci_lower'], partner_res['ci_upper'] - partner_res['mean']],
                 fmt='o', color='royalblue', capsize=5)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Mean Success Residual')
    plt.title('Pro Dancer Boost (Residualized Success)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pro_boost_residuals.png")
    plt.close()
    
    return partner_res

def main():
    df = load_data()
    print(f"Dataset Summary: {len(df)} contestants, {df['season'].nunique()} seasons")
    
    targets = {
        'success': 'success_score',
        'judge': 'judge_mean_w1_3',
        'fan': 'fan_mean_w1_3'
    }
    
    base_features = [
        'celebrity_age_during_season', 'celebrity_industry', 'celebrity_homestate', 
        'celebrity_homecountry/region', 'ballroom_partner'
    ]
    
    # Features for targets
    # A) Success: Celeb static + early window judge/fan performance
    success_features = base_features + ['judge_week1', 'judge_mean_w1_3', 'judge_slope_w1_3', 
                                       'fan_week1', 'fan_mean_w1_3', 'fan_slope_w1_3']
    
    # B & C) Judges and Fans: primarily celeb static (drivers of initial perception)
    driver_features = base_features
    
    results_metrics = []
    importances = {}
    
    for name, target in targets.items():
        print(f"Modeling {name}...")
        features = success_features if name == 'success' else driver_features
        X, y, groups = prepare_modeling_data(df, target, features)
        
        cv_res, cv_preds, final_model = train_and_cv(X, y, groups, model_type='xgb')
        
        # Aggregate metrics
        mean_metrics = np.mean(cv_res, axis=0)
        results_metrics.append({
            'target': name,
            'MAE': mean_metrics[0],
            'RMSE': mean_metrics[1],
            'Pearson': mean_metrics[2],
            'Spearman': mean_metrics[3]
        })
        
        # SHAP
        importances[name] = run_shap_analysis(final_model, X, name)
        
    # Save Metrics
    pd.DataFrame(results_metrics).to_csv(f"{OUTPUT_DIR}/cv_metrics.csv", index=False)
    
    # Comparison Table
    comp_df = importances['success'][['feature', 'mean_abs_shap']].rename(columns={'mean_abs_shap': 'shap_success'})
    comp_df = comp_df.merge(importances['judge'][['feature', 'mean_abs_shap']].rename(columns={'mean_abs_shap': 'shap_judge'}), on='feature', how='outer')
    comp_df = comp_df.merge(importances['fan'][['feature', 'mean_abs_shap']].rename(columns={'mean_abs_shap': 'shap_fan'}), on='feature', how='outer')
    comp_df['judge_minus_fan'] = comp_df['shap_judge'] - comp_df['shap_fan']
    comp_df.to_csv(f"{OUTPUT_DIR}/shap_comparison_table.csv", index=False)
    
    # Pro Boost Analysis
    print("Running Pro Boost Analysis...")
    features_no_partner = [f for f in success_features if f != 'ballroom_partner']
    pro_boost_analysis(df, features_no_partner)
    
    # Robustness: Week 1 vs Weeks 1-3 Success model features
    print("Running Robustness Tests...")
    X1, y1, g1 = prepare_modeling_data(df, 'success_score', base_features + ['judge_week1', 'fan_week1'])
    X13, y13, g13 = prepare_modeling_data(df, 'success_score', success_features)
    
    _, _, model1 = train_and_cv(X1, y1, g1)
    _, _, model13 = train_and_cv(X13, y13, g13)
    
    imp1 = pd.DataFrame({'feature': X1.columns, 'shap_week1': np.abs(shap.TreeExplainer(model1).shap_values(X1)).mean(axis=0)})
    imp13 = pd.DataFrame({'feature': X13.columns, 'shap_w1_3': np.abs(shap.TreeExplainer(model13).shap_values(X13)).mean(axis=0)})
    
    stability = imp13.merge(imp1, on='feature', how='left')
    stability.to_csv(f"{OUTPUT_DIR}/shap_stability_week1_vs_w1_3.csv", index=False)
    
    # Noise Sensitivity
    noise_results = []
    base_fan_imp = importances['fan'].copy()
    for scale in [0.05, 0.10]:
        df_noise = df.copy()
        df_noise['fan_mean_w1_3'] += np.random.normal(0, scale * df_noise['fan_mean_w1_3'].std(), len(df_noise))
        Xn, yn, gn = prepare_modeling_data(df_noise, 'fan_mean_w1_3', driver_features)
        _, _, model_n = train_and_cv(Xn, yn, gn)
        imp_n = pd.DataFrame({'feature': Xn.columns, f'shap_noise_{scale}': np.abs(shap.TreeExplainer(model_n).shap_values(Xn)).mean(axis=0)})
        noise_results.append(imp_n)
    
    sensitivity = base_fan_imp[['feature', 'mean_abs_shap']].rename(columns={'mean_abs_shap': 'shap_original'})
    for res in noise_results:
        sensitivity = sensitivity.merge(res, on='feature', how='left')
    sensitivity.to_csv(f"{OUTPUT_DIR}/fan_noise_sensitivity.csv", index=False)
    
    print("Done! Results saved in ml_results/")

if __name__ == "__main__":
    main()
