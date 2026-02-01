import pandas as pd
import numpy as np
import re

def deep_audit(df):
    results = []
    
    # 1. Season Integrity
    results.append("=== 1. Season Integrity ===")
    season_stats = df.groupby('season')['placement'].agg(['count', 'min', 'max']).reset_index()
    results.append(season_stats.to_string())
    
    # Check for missing ranks in each season
    for season in df['season'].unique():
        placements = set(df[df['season'] == season]['placement'])
        max_p = int(df[df['season'] == season]['placement'].max())
        expected = set(range(1, max_p + 1))
        missing = expected - placements
        if missing:
            results.append(f"  Season {season}: Missing placements {missing}")
            
    # Check for multiple 1st places
    winners = df[df['placement'] == 1].groupby('season')['celebrity_name'].count()
    multi_winners = winners[winners > 1]
    if not multi_winners.empty:
        results.append(f"  Warning: Seasons with multiple winners: {multi_winners.to_dict()}")

    # 2. Categorical Typos / Consistency
    results.append("\n=== 2. Categorical Consistency (Unique Values) ===")
    for col in ['celebrity_industry', 'celebrity_homecountry/region', 'celebrity_homestate']:
        unique_vals = sorted([str(x) for x in df[col].unique()])
        results.append(f"\n--- {col} ---")
        results.append(", ".join(unique_vals))

    # 3. Elimination vs Score Logic
    results.append("\n=== 3. Elimination vs Score Logic ===")
    # Extract week from results string (e.g., "Eliminated Week 3")
    def get_elim_week(res):
        if '1st' in res or '2nd' in res or '3rd' in res or '4th' in res or '5th' in res or 'Finalist' in res:
            return 11 # Assume they stay till end
        match = re.search(r'Week (\d+)', res)
        if match:
            return int(match.group(1))
        if 'Withdrew' in res:
            return -1 # Mark as withdrew for manual check
        return 11

    df['elim_week_calc'] = df['results'].apply(get_elim_week)
    
    score_cols = [c for c in df.columns if 'score' in c]
    score_weeks = {}
    for col in score_cols:
        match = re.search(r'week(\d+)', col)
        if match:
            w = int(match.group(1))
            if w not in score_weeks: score_weeks[w] = []
            score_weeks[w].append(col)

    inconsistencies = []
    for idx, row in df.iterrows():
        elim_w = row['elim_week_calc']
        if elim_w == -1: continue # Skip withdrew for now
        
        for w, cols in score_weeks.items():
            has_score = any(pd.to_numeric(row[c], errors='coerce') > 0 for c in cols)
            if w > elim_w and has_score:
                inconsistencies.append(f"  {row['celebrity_name']} (S{row['season']}): Eliminated Week {elim_w} but has score in Week {w}")
            elif w <= elim_w and not has_score and elim_w <= 10:
                 # Only check up to week 10, some early seasons had skips
                 pass

    if inconsistencies:
        results.append("\n".join(inconsistencies))
    else:
        results.append("  No score presence inconsistencies found (Week > Elim).")

    # 4. Score Magnitude Investigation
    results.append("\n=== 4. High Score Investigation (> 10) ===")
    high_scores = []
    for col in score_cols:
        vals = pd.to_numeric(df[col], errors='coerce')
        over_10 = df[vals > 10][['season', 'celebrity_name', col]]
        if not over_10.empty:
            for _, r in over_10.iterrows():
                high_scores.append(f"  S{r['season']} {r['celebrity_name']} {col}: {r[col]}")
    
    if high_scores:
        results.append("\n".join(high_scores[:20])) # Top 20
        results.append(f"  ... total {len(high_scores)} instances of scores > 10")
    
    # 5. Whitespace and Case sensitivity in categorical columns
    results.append("\n=== 5. Whitespace and Case Sensitivity ===")
    for col in ['celebrity_name', 'celebrity_industry', 'celebrity_homestate', 'celebrity_homecountry/region', 'ballroom_partner']:
        spaced = df[df[col].astype(str).str.strip() != df[col].astype(str)]
        if not spaced.empty:
            results.append(f"  Warning: {col} has values with leading/trailing spaces: {spaced[col].unique().tolist()}")
        
        # Case sensitivity check
        lowered = df[col].astype(str).str.lower()
        if df[col].astype(str).nunique() != lowered.nunique():
             # Find culprits
             counts = df[col].astype(str).value_counts()
             lowered_counts = lowered.value_counts()
             for val, count in lowered_counts.items():
                 # Find original values that map to this lower case
                 origs = df[lowered == val][col].unique()
                 if len(origs) > 1:
                     results.append(f"  Warning: {col} has case-sensitive duplicates: {origs.tolist()}")

    return "\n".join(results)

if __name__ == "__main__":
    df = pd.read_csv("Data/2026_MCM_Problem_C_Data.csv")
    report = deep_audit(df)
    with open("analysis/deep_audit_results.txt", "w") as f:
        f.write(report)
    print("Deep audit complete. See analysis/deep_audit_results.txt")
