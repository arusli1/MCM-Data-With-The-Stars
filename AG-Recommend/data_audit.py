import pandas as pd
import numpy as np
import os

def run_audit(file_path):
    print(f"--- Auditing: {file_path} ---")
    df = pd.read_csv(file_path)
    
    # 1. Basic Stats
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    
    # 2. Duplicates
    full_dupes = df.duplicated().sum()
    print(f"\nExact Duplicate Rows: {full_dupes}")
    
    key_dupes = df.duplicated(subset=['season', 'celebrity_name']).sum()
    print(f"Duplicate (Season, Celebrity) Pairs: {key_dupes}")
    if key_dupes > 0:
        print(df[df.duplicated(subset=['season', 'celebrity_name'], keep=False)].sort_values(['season', 'celebrity_name']))

    # 3. Missing Values
    print("\nMissing Values per Column (Percentage):")
    missing = df.isnull().mean() * 100
    print(missing[missing > 0].sort_values(ascending=False))
    
    # 4. Check for 'N/A' strings (common in this CSV)
    print("\n'N/A' string occurrences per column:")
    na_strings = (df == 'N/A').sum()
    print(na_strings[na_strings > 0].sort_values(ascending=False))

    # 5. Categorical Column Exploration
    cat_cols = ['celebrity_industry', 'celebrity_homestate', 'celebrity_homecountry/region', 'ballroom_partner']
    print("\nUnique Values in Categorical Columns:")
    for col in cat_cols:
        print(f"  {col}: {df[col].nunique()} unique values")

    # 6. Numeric Constraints
    print("\nNumeric Validation:")
    # Age
    df['celebrity_age_during_season'] = pd.to_numeric(df['celebrity_age_during_season'], errors='coerce')
    print(f"  Age: min={df['celebrity_age_during_season'].min()}, max={df['celebrity_age_during_season'].max()}")
    
    # Placement
    print(f"  Placement: min={df['placement'].min()}, max={df['placement'].max()}")
    
    # Scores (check a sample of score columns)
    score_cols = [c for c in df.columns if 'score' in c]
    score_data = []
    for col in score_cols:
        numeric_scores = pd.to_numeric(df[col], errors='coerce')
        # Treat 0 as special/missing if that's the convention
        zeros = (numeric_scores == 0).sum()
        score_data.append({
            'col': col,
            'min': numeric_scores[numeric_scores > 0].min(),
            'max': numeric_scores.max(),
            'zeros': zeros,
            'nans': numeric_scores.isna().sum()
        })
    
    score_summary = pd.DataFrame(score_data)
    print("\nScores Summary (excluding 0 for min):")
    print(score_summary[['col', 'min', 'max', 'zeros', 'nans']].sort_values('nans'))

    # 7. Consistency Checks
    print("\nConsistency Checks:")
    # Check if '1st Place' matches placement 1
    winners = df[df['results'] == '1st Place']
    mismatched_winners = winners[winners['placement'] != 1]
    print(f"  Mismatched '1st Place' vs placement: {len(mismatched_winners)}")
    
    # Check for trailing spaces in names
    spaced_names = df[df['celebrity_name'].str.endswith(' ') | df['celebrity_name'].str.startswith(' ')]
    print(f"  Celebrity names with leading/trailing spaces: {len(spaced_names)}")
    if len(spaced_names) > 0:
        print(spaced_names['celebrity_name'].tolist())

if __name__ == "__main__":
    csv_path = "Data/2026_MCM_Problem_C_Data.csv"
    if os.path.exists(csv_path):
        run_audit(csv_path)
    else:
        print(f"File not found: {csv_path}")
