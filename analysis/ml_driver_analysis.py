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
OUTPUT_DIR = 'AG_Problem_3'
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
    X = pd.get_dummies(X, columns=categorical_features, dtype=int)
    
    return X, y, groups

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Handle cases with constant predictions or true values
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = 0.0
        spearman = 0.0
    else:
        pearson, _ = pearsonr(y_true, y_pred)
        spearman, _ = spearmanr(y_true, y_pred)
    return mae, rmse, pearson, spearman

def compute_ranking_metrics(df, preds_col, target_col='success_score'):
    # Rank within each season
    df = df.copy()
    df['predicted_success'] = preds_col
    
    # Higher success_score means better rank (lower placement number)
    # We use rank(ascending=False) so highest score is rank 1
    df['predicted_rank'] = df.groupby('season')['predicted_success'].rank(ascending=False, method='min')
    df['actual_rank'] = df.groupby('season')['placement'].rank(ascending=True, method='min')
    
    # Metrics
    # 1. Top-1 Accuracy: Is the predicted rank 1 actually the winner (placement 1)?
    def is_top_1_correct(group):
        pred_winner = group[group['predicted_rank'] == 1]
        if len(pred_winner) == 0: return 0
        # If there's a tie for rank 1, check if any of them is the actual winner
        return 1 if 1 in pred_winner['placement'].values else 0
    
    top1_acc = df.groupby('season').apply(is_top_1_correct).mean()
    
    # 2. Top-3 Accuracy: Is the actual winner in our predicted top 3?
    def is_winner_in_top_3(group):
        top_3_preds = group[group['predicted_rank'] <= 3]
        return 1 if 1 in top_3_preds['placement'].values else 0
    
    top3_acc = df.groupby('season').apply(is_winner_in_top_3).mean()
    
    # 3. Mean Rank Error: How many spots off are we on average?
    df['rank_error'] = np.abs(df['actual_rank'] - df['predicted_rank'])
    mean_rank_error = df['rank_error'].mean()
    
    return top1_acc, top3_acc, mean_rank_error, df[['season', 'celebrity_name', 'actual_rank', 'predicted_rank']]

def train_and_cv(X, y, groups, tune=False):
    logo = LeaveOneGroupOut()
    
    # Defaults
    rf_params = {'n_estimators': 400, 'max_depth': 6, 'random_state': 42}
    xgb_params = {'max_depth': 5, 'n_estimators': 400, 'learning_rate': 0.05, 'random_state': 42}
    
    if tune:
        print("  Tuning XGBoost...")
        from sklearn.model_selection import GridSearchCV
        param_grid = {
            'max_depth': [3, 5, 7],
            'n_estimators': [200, 400],
            'learning_rate': [0.01, 0.05, 0.1]
        }
        grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_absolute_error')
        grid.fit(X, y)
        xgb_params.update(grid.best_params_)
        print(f"  Best XGB Params: {grid.best_params_}")

    models_to_test = {
        'rf': RandomForestRegressor(**rf_params),
        'xgb': XGBRegressor(**xgb_params)
    }
    
    all_metrics = {}
    for m_name, model in models_to_test.items():
        cv_results = []
        all_preds = np.zeros(len(y))
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            m = model.__class__(**model.get_params())
            m.fit(X_train, y_train)
            preds = m.predict(X_test)
            all_preds[test_idx] = preds
            cv_results.append(evaluate_model(y_test, preds))
        
        all_metrics[m_name] = {
            'cv_res': cv_results,
            'preds': all_preds,
            'mean_metrics': np.mean(cv_results, axis=0)
        }
    
    # Train final XGB for SHAP
    final_model = XGBRegressor(**xgb_params)
    final_model.fit(X, y)
    
    return all_metrics, final_model

def format_feature_name(name):
    # Mapping for common features
    mapping = {
        'celebrity_age_during_season': 'Celebrity Age During Season',
        'judge_mean_w1_3': 'Judge Mean (Weeks 1-3)',
        'fan_mean_w1_3': 'Fan Mean (Weeks 1-3)',
        'judge_slope_w1_3': 'Judge Improvement (Weeks 1-3)',
        'fan_slope_w1_3': 'Fan Improvement (Weeks 1-3)',
        'judge_week1': 'Judge Score (Week 1)',
        'fan_week1': 'Fan Score (Week 1)'
    }
    if name in mapping:
        return mapping[name]
    
    # Handle categorical (one-hot encoded)
    prefixes = ['celebrity_industry_', 'ballroom_partner_', 'celebrity_homestate_', 'celebrity_homecountry/region_']
    for p in prefixes:
        if name.startswith(p):
            category = p[:-1].replace('_', ' ').title().replace('Homecountry', 'Home Country')
            value = name[len(p):].replace('_', ' ')
            return f"{category}: {value}"
    
    # Fallback
    return name.replace('_', ' ').title()

def run_shap_analysis(model, X, target_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Summary Plot (Uses the feature names from X, so we need to rename X columns temporarily or use the feature_names arg)
    feature_names_clean = [format_feature_name(c) for c in X.columns]
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names_clean, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary_{target_name}.png")
    plt.close()
    
    # Importance CSV
    importance = pd.DataFrame({
        'feature': feature_names_clean,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    importance.to_csv(f"{OUTPUT_DIR}/shap_importance_{target_name}.csv", index=False)
    
    # 1. SHAP Dependence Plots for Continuous Characteristics
    # age, judge_mean_w1_3, fan_mean_w1_3, slopes
    cont_features = ['celebrity_age_during_season', 'judge_mean_w1_3', 'fan_mean_w1_3', 
                     'judge_slope_w1_3', 'fan_slope_w1_3']
    for feat in cont_features:
        if feat in X.columns:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(feat, shap_values, X, show=False)
            plt.title(f"SHAP Dependence: {format_feature_name(feat)} ({target_name.title()})")
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/shap_dependence_{feat}_{target_name}.png")
            plt.close()

    # 2. Categorical SHAP distribution plots (Beeswarm subselection)
    # industry buckets, partner IDs
    cat_groups = ['celebrity_industry', 'ballroom_partner', 'celebrity_homestate', 'celebrity_homecountry/region']
    for base_cat in cat_groups:
        related_cols = [c for c in X.columns if c.startswith(base_cat)]
        if not related_cols: continue
        
        # Get indices of these columns
        col_indices = [X.columns.get_loc(c) for c in related_cols]
        
        # Plot beeswarm for just these categorical features
        plt.figure(figsize=(10, 6))
        temp_shap = shap_values[:, col_indices]
        temp_X = X.iloc[:, col_indices]
        # Format the column names for the sub-plot
        temp_feat_names = [format_feature_name(c) for c in temp_X.columns]
        shap.summary_plot(temp_shap, temp_X, feature_names=temp_feat_names, plot_type="violin", show=False)
        plt.title(f"SHAP Effects: {format_feature_name(base_cat + '_')} ({target_name.title()})")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_cat_{base_cat.replace('/', '_')}_{target_name}.png")
        plt.close()
    
    # 3. Specialized SHAP graphs (Top 10 Positive and Top 10 Negative)
    feature_directions = []
    for i, col in enumerate(X.columns):
        if np.std(X[col]) > 0:
            corr, _ = pearsonr(X[col], shap_values[:, i])
            feature_directions.append(corr)
        else:
            feature_directions.append(0)
    
    # We need to map directions back to the cleaned importance names if we are using cleaned names,
    # but 'importance' already has the clean names.
    # Re-calculate importance to ensure we have directions matched to clean names correctly.
    clean_importance = pd.DataFrame({
        'feature': feature_names_clean,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0),
        'direction': feature_directions
    }).sort_values('mean_abs_shap', ascending=False)
    
    top_pos = clean_importance[clean_importance['direction'] > 0].head(10)
    top_neg = clean_importance[clean_importance['direction'] < 0].head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_pos, x='mean_abs_shap', y='feature', palette='Greens_r')
    plt.title(f"Top 10 Positive Factors - {target_name.title()}")
    plt.xlabel("Mean Absolute SHAP Value (Impact Magnitude)")
    plt.ylabel("Factor")
    plt.savefig(f"{OUTPUT_DIR}/shap_top10_pos_{target_name}.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_neg, x='mean_abs_shap', y='feature', palette='Reds_r')
    plt.title(f"Top 10 Negative Factors - {target_name.title()}")
    plt.xlabel("Mean Absolute SHAP Value (Impact Magnitude)")
    plt.ylabel("Factor")
    plt.savefig(f"{OUTPUT_DIR}/shap_top10_neg_{target_name}.png")
    plt.close()
    
    # 4. Per-characteristic graphs (focused SHAP visualization)
    chars = ['ballroom_partner', 'celebrity_industry', 'celebrity_homestate', 'celebrity_homecountry/region', 'celebrity_age_during_season']
    for char in chars:
        # Use clean_importance which already has clean names
        # But we need to filter by original prefix. Let's do it slightly differently.
        related_indices = [i for i, c in enumerate(X.columns) if c.startswith(char)]
        if not related_indices: continue
        
        char_importance = pd.DataFrame({
            'feature': [feature_names_clean[i] for i in related_indices],
            'mean_abs_shap': [np.abs(shap_values[:, i]).mean() for i in related_indices]
        }).sort_values('mean_abs_shap', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=char_importance.head(10), x='mean_abs_shap', y='feature', palette='viridis')
        plt.title(f"Impact of {format_feature_name(char + '_')} - {target_name.title()}")
        plt.xlabel("Mean Absolute SHAP Value")
        plt.ylabel("Category")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/shap_feature_{char.replace('/', '_')}_{target_name}.png")
        plt.close()
        
    return importance

def pro_boost_analysis(df, features_no_partner, target_col='success_score'):
    # Fit model without partner features
    X, y, groups = prepare_modeling_data(df, target_col, features_no_partner)
    metrics_bundle, _ = train_and_cv(X, y, groups)
    cv_preds = metrics_bundle['xgb']['preds']
    
    data = df.dropna(subset=[target_col]).copy()
    data['predicted_success'] = cv_preds
    data['residual'] = (data[target_col] - data['predicted_success']).astype(float)
    
    # Aggregate mean residual by partner
    partner_res = data.groupby('ballroom_partner')['residual'].agg(['mean', 'count']).reset_index()
    
    # Bootstrap CIs
    seasons = data['season'].unique()
    n_boot = 1000
    
    # Pre-allocate or use a better structure
    # We want mean residual per partner per bootstrap iteration
    partners = partner_res['ballroom_partner'].tolist()
    boot_matrix = np.full((n_boot, len(partners)), np.nan, dtype=float)
    
    for i in range(n_boot):
        boot_seasons = np.random.choice(seasons, size=len(seasons), replace=True)
        boot_data = pd.concat([data[data['season'] == s] for s in boot_seasons])
        means = boot_data.groupby('ballroom_partner')['residual'].mean().astype(float)
        for j, p in enumerate(partners):
            if p in means.index:
                boot_matrix[i, j] = means[p]
                
    ci_lower = []
    ci_upper = []
    for j in range(len(partners)):
        vals = boot_matrix[:, j]
        vals = vals[~np.isnan(vals)]
        if len(vals) > 0:
            try:
                sorted_vals = np.sort(vals.astype(float))
                # Explicitly take indices for 2.5th and 97.5th percentiles
                l_idx = max(0, int(0.025 * len(sorted_vals)))
                u_idx = min(len(sorted_vals) - 1, int(0.975 * len(sorted_vals)))
                ci_lower.append(sorted_vals[l_idx])
                ci_upper.append(sorted_vals[u_idx])
            except Exception as e:
                print(f"Error for partner {partners[j]}: {e}")
                ci_lower.append(np.nan)
                ci_upper.append(np.nan)
        else:
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
    
    partner_res['ci_lower'] = ci_lower
    partner_res['ci_upper'] = ci_upper
    
    partner_res = partner_res.sort_values('mean', ascending=False)
    partner_res.to_csv(f"{OUTPUT_DIR}/pro_boost_residuals.csv", index=False)
    
    # Plot
    plt.figure(figsize=(10, 12)) # Taller for partners
    # Format partner names for y-axis
    clean_partner_names = [p.replace('_', ' ') if p != 'Other_Partner' else 'Other Professional Partner' for p in partner_res['ballroom_partner']]
    plt.errorbar(partner_res['mean'], clean_partner_names, 
                 xerr=[partner_res['mean'] - partner_res['ci_lower'], partner_res['ci_upper'] - partner_res['mean']],
                 fmt='o', color='royalblue', capsize=5)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Mean Outcome Boost (Residuals)')
    plt.ylabel('Professional Dancer')
    plt.title('Professional Dancer Impact (Success Outperformance)')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pro_boost_residuals.png")
    plt.close()
    
    return partner_res

def main():
    df = load_data()
    print(f"Dataset Summary:")
    print(f"  Contestants: {len(df)}")
    print(f"  Seasons: {df['season'].nunique()}")
    
    # Missingness rates
    early_features = ['judge_mean_w1_3', 'fan_mean_w1_3', 'judge_slope_w1_3', 'fan_slope_w1_3']
    print("Missingness Assessment (Early Window):")
    for feat in early_features:
        miss = df[feat].isna().mean()
        print(f"  {feat}: {miss:.2%}")

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
        
        # Tune only the success model to save time, others use defaults
        metrics_bundle, final_model = train_and_cv(X, y, groups, tune=(name=='success'))
        
        # Save metrics for both RF and XGB
        for m_name, bundle in metrics_bundle.items():
            mean_metrics = bundle['mean_metrics']
            
            # Additional ranking metrics for success models
            top1, top3, mre = np.nan, np.nan, np.nan
            if name == 'success':
                # Filter df to match indices of X/y
                modeling_df = df.iloc[X.index].copy()
                top1, top3, mre, pred_df = compute_ranking_metrics(modeling_df, bundle['preds'])
                pred_df.to_csv(f"{OUTPUT_DIR}/ranking_predictions_{m_name}.csv", index=False)

            results_metrics.append({
                'model': m_name,
                'target': name,
                'MAE': mean_metrics[0],
                'RMSE': mean_metrics[1],
                'Pearson': mean_metrics[2],
                'Spearman': mean_metrics[3],
                'Top1_Acc': top1,
                'Top3_Acc': top3,
                'Mean_Rank_Err': mre
            })
        
        # SHAP using the "final" XGB model
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
    
    _, model1 = train_and_cv(X1, y1, g1)
    _, model13 = train_and_cv(X13, y13, g13)
    
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
        _, model_n = train_and_cv(Xn, yn, gn)
        imp_n = pd.DataFrame({'feature': Xn.columns, f'shap_noise_{scale}': np.abs(shap.TreeExplainer(model_n).shap_values(Xn)).mean(axis=0)})
        noise_results.append(imp_n)
    
    sensitivity = base_fan_imp[['feature', 'mean_abs_shap']].rename(columns={'mean_abs_shap': 'shap_original'})
    for res in noise_results:
        sensitivity = sensitivity.merge(res, on='feature', how='left')
    sensitivity.to_csv(f"{OUTPUT_DIR}/fan_noise_sensitivity.csv", index=False)
    
    print("Done! Results saved in ml_results/")

if __name__ == "__main__":
    main()
