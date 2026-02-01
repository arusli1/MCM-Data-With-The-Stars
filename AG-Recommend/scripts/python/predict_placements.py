import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
import os

# Paths
DATA_PATH = "../../Data/2026_MCM_Problem_C_Data.csv"
FAN_VOTES_PATH = "../../AR-Problem1-Base/final_../../results/base_inferred_shares.csv"
OUTPUT_DIR = "../../results"

def load_and_preprocess():
    print("Loading data...")
    df_main = pd.read_csv(DATA_PATH)
    df_fan = pd.read_csv(FAN_VOTES_PATH)

    # Calculate Average Judge Score
    score_cols = [c for c in df_main.columns if 'score' in c and 'judge' in c]
    
    def get_avg_judge_score(row):
        scores = pd.to_numeric(row[score_cols], errors='coerce')
        scores = scores[scores > 0]
        return scores.mean()

    df_main['avg_judge_score'] = df_main.apply(get_avg_judge_score, axis=1)

    # Process Fan Votes
    df_fan_agg = df_fan.groupby(['season', 'celebrity_name'])['s_share'].mean().reset_index()
    df_fan_agg.rename(columns={'s_share': 'avg_fan_share'}, inplace=True)

    # Merge
    df_merged = pd.merge(df_main, df_fan_agg, on=['season', 'celebrity_name'], how='inner')
    
    return df_merged

def feature_engineering(df):
    print("Engineering features...")
    
    # Placement
    df['placement'] = pd.to_numeric(df['placement'], errors='coerce')
    season_counts = df.groupby('season')['placement'].transform('max')
    df['placement_norm'] = (df['placement'] - 1) / (season_counts - 1)
    df['success_score'] = 1 - df['placement_norm']

    # Features
    df['age'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce').fillna(df['celebrity_age_during_season'].median())

    # Industry
    top_industries = df['celebrity_industry'].value_counts().nlargest(6).index
    df['industry_clean'] = df['celebrity_industry'].apply(lambda x: x if x in top_industries else 'Other')

    # Partner
    top_partners = df['ballroom_partner'].value_counts().nlargest(15).index
    df['partner_clean'] = df['ballroom_partner'].apply(lambda x: x if x in top_partners else 'Other_Partner')
    
    return df

def predict_all_placements(df):
    print("\nGenerating predictions for all contestants...")
    
    # Prepare features
    X_raw = df[['age', 'industry_clean', 'partner_clean', 'celebrity_homestate', 'celebrity_homecountry/region']].copy()
    X = pd.get_dummies(X_raw, drop_first=True)
    X = X.fillna(0)
    
    # Target
    y = df['success_score'].fillna(df['success_score'].mean())
    groups = df['season']
    
    # LOGO CV to get predictions
    logo = LeaveOneGroupOut()
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    predictions = np.zeros(len(y))
    
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions[test_idx] = model.predict(X.iloc[test_idx])
    
    # Convert success scores back to placements
    df['predicted_success_score'] = predictions
    
    # For each season, rank by predicted success score
    df['predicted_placement'] = df.groupby('season')['predicted_success_score'].rank(ascending=False, method='first')
    
    # Create output dataframe
    output = df[['season', 'celebrity_name', 'ballroom_partner', 'placement', 'predicted_placement', 
                 'success_score', 'predicted_success_score']].copy()
    
    # Calculate error
    output['placement_error'] = output['placement'] - output['predicted_placement']
    output['absolute_error'] = np.abs(output['placement_error'])
    
    # Sort by season and predicted placement
    output = output.sort_values(['season', 'predicted_placement'])
    
    # Save to CSV
    output.to_csv(f"{OUTPUT_DIR}/all_contestant_predictions.csv", index=False)
    
    print(f"\n✓ Saved predictions for {len(output)} contestants")
    print(f"  Mean Absolute Error: {output['absolute_error'].mean():.2f} places")
    print(f"  Median Absolute Error: {output['absolute_error'].median():.2f} places")
    
    # Create season-by-season summary
    season_summary = []
    for season in sorted(df['season'].unique()):
        season_df = output[output['season'] == season]
        season_summary.append({
            'season': season,
            'contestants': len(season_df),
            'mean_abs_error': season_df['absolute_error'].mean(),
            'median_abs_error': season_df['absolute_error'].median(),
            'predicted_winner': season_df.iloc[0]['celebrity_name'],
            'actual_winner': season_df[season_df['placement'] == 1]['celebrity_name'].values[0]
        })
    
    season_summary_df = pd.DataFrame(season_summary)
    season_summary_df.to_csv(f"{OUTPUT_DIR}/season_prediction_summary.csv", index=False)
    
    print(f"\n✓ Saved season summary")
    print(f"\nWinner Predictions:")
    correct_winners = sum(season_summary_df['predicted_winner'] == season_summary_df['actual_winner'])
    print(f"  Correctly predicted: {correct_winners}/{len(season_summary_df)} seasons")
    
    return output, season_summary_df

def main():
    df = load_and_preprocess()
    df = feature_engineering(df)
    predictions, summary = predict_all_placements(df)
    
    print("\n" + "="*60)
    print("Prediction generation complete!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - all_contestant_predictions.csv")
    print(f"  - season_prediction_summary.csv")

if __name__ == "__main__":
    main()
