"""
perform_train_test.py

This script performs a formal chronological Train-Test split.
Training: Seasons 1-30
Testing: Seasons 31-34 (The latest data)

This serves as a final validation to ensure the model generalizes to new seasons 
and isn't just overfitting on historical data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Config
DATA_PATH = "../../Data/2026_MCM_Problem_C_Data.csv"
OUTPUT_DIR = "../../results"
TEST_SEASONS = [31, 32, 33, 34]

def bucket_top_n(series, n, other_label):
    vc = series.fillna("Unknown").astype(str).value_counts()
    top = set(vc.nlargest(n).index)
    return series.fillna("Unknown").astype(str).apply(lambda x: x if x in top else other_label)

def perform_validation():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    print("Engineering features...")
    # Target
    df["placement"] = pd.to_numeric(df["placement"], errors="coerce")
    season_max = df.groupby("season")["placement"].transform("max")
    df["success_score"] = 1 - (df["placement"] - 1) / (season_max - 1).replace(0, 1)

    # Features
    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce").fillna(35)
    df["industry_clean"] = bucket_top_n(df["celebrity_industry"], 6, "Other")
    df["partner_clean"] = bucket_top_n(df["ballroom_partner"], 15, "Other_Partner")

    # Early Score (W1-2)
    score_cols = [c for c in df.columns if ('score' in c and 'judge' in c) and ('week1_' in c or 'week2_' in c)]
    df["early_score"] = pd.to_numeric(df[score_cols].replace(0, np.nan).mean(axis=1), errors='coerce').fillna(df[score_cols].replace(0, np.nan).stack().mean())

    # Build matrix
    X_raw = df[["season", "age", "industry_clean", "partner_clean", "early_score"]]
    X = pd.get_dummies(X_raw.drop(columns=["season"]), drop_first=False)
    y = df["success_score"].fillna(0)
    seasons = df["season"]

    # Split
    train_idx = ~seasons.isin(TEST_SEASONS)
    test_idx = seasons.isin(TEST_SEASONS)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"Split: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")

    # Train
    model = RandomForestRegressor(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Predict
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Metrics (R2)
    print("\n--- PERFORMANCE METRICS (Success Score 0-1) ---")
    print(f"Train R²: {r2_score(y_train, train_preds):.3f}")
    print(f"Test R²:  {r2_score(y_test, test_preds):.3f}")

    # Rank Metrics for Test Set
    test_df = df[test_idx].copy()
    test_df["pred_score"] = test_preds
    test_df["pred_placement"] = test_df.groupby("season")["pred_score"].rank(ascending=False, method="first")
    
    test_df["abs_error"] = (test_df["placement"] - test_df["pred_placement"]).abs()
    
    mae_rank = test_df["abs_error"].mean()
    mse_rank = (test_df["abs_error"]**2).mean()

    # Top-1 Winner Accuracy (Test Set)
    winners_correct = []
    for s in TEST_SEASONS:
        season_df = test_df[test_df["season"] == s]
        if season_df.empty: continue
        actual_winner = season_df.loc[season_df["placement"] == 1, "celebrity_name"].values[0]
        pred_winner = season_df.sort_values("pred_score", ascending=False).iloc[0]["celebrity_name"]
        winners_correct.append(actual_winner == pred_winner)

    print("\n--- RANK ERROR (Test Set Only: Seasons 31-34) ---")
    print(f"Avg Rank Error (MAE): {mae_rank:.2f} places")
    print(f"Rank MSE:            {mse_rank:.2f}")
    print(f"Winner Accuracy:     {np.mean(winners_correct):.2%}")

    # Save validation results
    test_df[["season", "celebrity_name", "placement", "pred_placement", "abs_error"]].to_csv(os.path.join(OUTPUT_DIR, "validation_results_seasons_31_34.csv"), index=False)
    print(f"\n✓ Saved validation results to {OUTPUT_DIR}/validation_results_seasons_31_34.csv")

if __name__ == "__main__":
    perform_validation()
