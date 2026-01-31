"""
xgboost_comparison.py

This script performs a head-to-head comparison between 
Random Forest and XGBoost for the DWTS Success Model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
import shap

# Config
DATA_PATH = "../../Data/2026_MCM_Problem_C_Data.csv"
OUTPUT_DIR = "../../results"

def bucket_top_n(series, n, other_label):
    vc = series.fillna("Unknown").astype(str).value_counts()
    top = set(vc.nlargest(n).index)
    return series.fillna("Unknown").astype(str).apply(lambda x: x if x in top else other_label)

def prepare_data():
    df = pd.read_csv(DATA_PATH)
    
    # Target
    df["placement"] = pd.to_numeric(df["placement"], errors="coerce")
    season_max = df.groupby("season")["placement"].transform("max")
    df["success_score"] = 1 - (df["placement"] - 1) / (season_max - 1).replace(0, 1)

    # Features
    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce").fillna(35)
    df["industry"] = bucket_top_n(df["celebrity_industry"], 6, "Other")
    df["partner"] = bucket_top_n(df["ballroom_partner"], 15, "Other_Partner")
    
    # Early Score (W1-2)
    score_cols = [c for c in df.columns if ('score' in c and 'judge' in c) and ('week1_' in c or 'week2_' in c)]
    df["early_score"] = pd.to_numeric(df[score_cols].replace(0, np.nan).mean(axis=1), errors='coerce').fillna(df[score_cols].replace(0, np.nan).stack().mean())

    X_raw = df[["age", "industry", "partner", "early_score"]]
    X = pd.get_dummies(X_raw, drop_first=False)
    y = df["success_score"].fillna(0)
    groups = df["season"]
    
    return X, y, groups, df

def compare_models():
    X, y, groups, df = prepare_data()
    logo = LeaveOneGroupOut()
    
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1),
        "GradientBoosting": HistGradientBoostingRegressor(max_iter=400, max_depth=3, learning_rate=0.01, random_state=42)
    }
    
    comp_results = []
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        preds = cross_val_predict(model, X, y, cv=logo, groups=groups)
        
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        
        # Spearman on ranks
        temp_df = df.copy()
        temp_df["preds"] = preds
        corrs = []
        for s, g in temp_df.groupby("season"):
            if len(g) > 1:
                corrs.append(g["success_score"].corr(g["preds"], method="spearman"))
        spearman = np.mean(corrs)
        
        comp_results.append({
            "Model": name,
            "R2": r2,
            "MAE (Score)": mae,
            "Spearman (Rank Accuracy)": spearman
        })
        
        # SHAP Analysis
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list): shap_values = shap_values[0]
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": X.columns,
            "importance": mean_abs_shap
        }).sort_values("importance", ascending=False)
        
        shap_df.to_csv(f"{OUTPUT_DIR}/shap_importance_raw_{name}.csv", index=False)

    summary_df = pd.DataFrame(comp_results)
    print("\n--- MODEL COMPARISON ---")
    print(summary_df.to_string(index=False))
    
    summary_df.to_csv(f"{OUTPUT_DIR}/model_comparison_results.csv", index=False)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    compare_models()
