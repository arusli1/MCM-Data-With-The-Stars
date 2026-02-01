import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shap
import os
from xgboost import XGBRegressor

# Direct imports from main script logic
import sys
sys.path.append('analysis')
from ml_driver_analysis import load_data, prepare_modeling_data

OUTPUT_DIR = 'AG_Problem_3'

def analyze_clustering():
    df = load_data()
    
    # We'll use the success target features
    base_features = [
        'celebrity_age_during_season', 'celebrity_industry', 'celebrity_homestate', 
        'celebrity_homecountry/region', 'ballroom_partner'
    ]
    success_features = base_features + ['judge_week1', 'judge_mean_w1_3', 'judge_slope_w1_3', 
                                       'fan_week1', 'fan_mean_w1_3', 'fan_slope_w1_3']
    
    X, y, groups = prepare_modeling_data(df, 'success_score', success_features)
    
    # 1. Feature Correlation Matrix
    plt.figure(figsize=(15, 12))
    # Filter for features with some variance
    corr = X.corr()
    # Find top correlated pairs to avoid giant messy matrix
    sns.heatmap(corr, cmap='RdBu_r', center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_correlation.png")
    plt.close()
    
    # 2. SHAP Clustering (Personas)
    # Train the best model again (using params from previous run)
    xgb_params = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 400, 'random_state': 42}
    model = XGBRegressor(**xgb_params)
    model.fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Standardize SHAP values for clustering
    # Each row is a contestant's "impact profile"
    scaler = StandardScaler()
    shap_scaled = scaler.fit_transform(shap_values)
    
    # Use K-Means to find 4 personas
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(shap_scaled)
    
    # Map back to names
    data_with_clusters = df.iloc[X.index].copy()
    data_with_clusters['persona_cluster'] = clusters
    
    # Profile the clusters
    cluster_profiles = []
    for i in range(4):
        c_data = data_with_clusters[data_with_clusters['persona_cluster'] == i]
        profile = {
            'cluster': i,
            'count': len(c_data),
            'avg_age': c_data['celebrity_age_during_season'].mean(),
            'avg_success': c_data['success_score'].mean(),
            'top_industry': c_data['celebrity_industry'].mode()[0],
            'avg_judge': c_data['judge_mean_w1_3'].mean(),
            'avg_fan': c_data['fan_mean_w1_3'].mean()
        }
        cluster_profiles.append(profile)
    
    profiles_df = pd.DataFrame(cluster_profiles)
    profiles_df.to_csv(f"{OUTPUT_DIR}/contestant_personas.csv", index=False)
    
    # 3. Decision Tree visualization of personas (simplified)
    # We can see which SHAP values define the clusters
    plt.figure(figsize=(12, 6))
    for i in range(4):
        cluster_shap = shap_values[clusters == i]
        plt.bar(np.arange(len(X.columns)), np.mean(cluster_shap, axis=0), alpha=0.5, label=f'Persona {i}')
    
    plt.xticks(np.arange(len(X.columns)), X.columns, rotation=90, fontsize=8)
    plt.legend()
    plt.title("SHAP Profile by Persona Cluster")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/persona_shap_profiles.png")
    plt.close()
    
    print("Clustering analysis complete.")

if __name__ == "__main__":
    analyze_clustering()
