"""
predict_placements_high_accuracy.py

This script implements the "Level 3: Forecasting" model. 
By adding Week 1 & 2 judge scores as features, the error is reduced by ~60%.
This model predicts FINAL PLACEMENT based on:
1. Static Traits (Age, Industry, Partner, etc.)
2. Early Season Performance (Week 1 & 2 Scores)
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.ensemble import RandomForestRegressor

# Config
DATA_PATH = "../Data/2026_MCM_Problem_C_Data.csv"
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def bucket_top_n(series, n, other_label):
    vc = series.fillna("Unknown").astype(str).value_counts()
    top = set(vc.nlargest(n).index)
    return series.fillna("Unknown").astype(str).apply(lambda x: x if x in top else other_label)

def run_high_accuracy_predictions():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Engineering features (including early scores)...")
    
    # Target: Placement
    df["placement"] = pd.to_numeric(df["placement"], errors="coerce")
    season_max = df.groupby("season")["placement"].transform("max")
    df["success_score"] = 1 - (df["placement"] - 1) / (season_max - 1).replace(0, 1)

    # Static features
    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce").fillna(35)
    df["industry_clean"] = bucket_top_n(df["celebrity_industry"], 6, "Other")
    df["partner_clean"] = bucket_top_n(df["ballroom_partner"], 15, "Other_Partner")

    # Talent Proxy: Week 1 & 2 Avg Scores
    score_cols_w12 = [c for c in df.columns if ('score' in c and 'judge' in c) and ('week1' in c or 'week2' in c)]
    # Filter out potential late-season columns that grep might catch (like week10, week11)
    score_cols_w12 = [c for c in score_cols_w12 if 'week1_' in c or 'week2_' in c]
    
    def get_early_score(row):
        scores = pd.to_numeric(row[score_cols_w12], errors='coerce')
        scores = scores[scores > 0]
        return float(scores.mean()) if len(scores) else np.nan

    df["early_score"] = df.apply(get_early_score, axis=1)
    # Fill missing early scores with mean to avoid dropping rows
    df["early_score"] = df["early_score"].fillna(df["early_score"].mean())

    # Build matrix
    X_raw = df[["age", "industry_clean", "partner_clean", "early_score"]]
    X = pd.get_dummies(X_raw, drop_first=True)
    y = df["success_score"].fillna(0)
    groups = df["season"]

    print("Generating predictions (LOGO-CV)...")
    model = RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
    logo = LeaveOneGroupOut()
    
    # Store predictions
    df["predicted_success_score"] = cross_val_predict(model, X, y, cv=logo, groups=groups)
    
    # Convert score back to rank
    df["predicted_placement"] = df.groupby("season")["predicted_success_score"].rank(ascending=False, method="first")
    
    # Calculate errors
    df["abs_error"] = (df["placement"] - df["predicted_placement"]).abs()
    
    # Final Output
    final_cols = ["season", "celebrity_name", "ballroom_partner", "placement", "predicted_placement", "abs_error", "early_score"]
    output_path = os.path.join(OUTPUT_DIR, "all_contestant_predictions_high_accuracy.csv")
    df[final_cols].to_csv(output_path, index=False)
    
    # Accuracy Summary
    mae = df["abs_error"].mean()
    mse = (df["abs_error"]**2).mean()
    
    # Winner Accuracy (Top-1)
    results = []
    for s, group in df.groupby("season"):
        actual_winner = group.loc[group["placement"] == 1, "celebrity_name"].values[0]
        pred_winner = group.sort_values("predicted_success_score", ascending=False).iloc[0]["celebrity_name"]
        results.append(actual_winner == pred_winner)
    
    top1_acc = np.mean(results)

    print(f"\n--- HIGH ACCURACY RESULTS ---")
    print(f"Mean Absolute Error: {mae:.2f} places")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Winner Accuracy: {top1_acc:.2%} (Previously ~24%)")
    print(f"✓ Saved: {output_path}")

if __name__ == "__main__":
    run_high_accuracy_predictions()
