import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("../Data/2026_MCM_Problem_C_Data.csv")

# Clean Partner Names (Top 15)
top_partners = ["Cheryl Burke", "Derek Hough", "Mark Ballas", "Valentin Chmerkovskiy", 
                "Kym Johnson", "Maksim Chmerkoskiy", "Tony Dovolani", "Karina Smirnoff", 
                "Witney Carson", "Peta Murgatroyd", "Emma Slater", "Sharna Burgess", 
                "Sasha Farber", "Artem Chigvintsev", "Gleb Savchenko"]

def clean_partner(p):
    if pd.isna(p): return "Other"
    for tp in top_partners:
        if tp in str(p): return tp
    return "Other"

df['partner_clean'] = df['ballroom_partner'].apply(clean_partner)
df['placement'] = pd.to_numeric(df['placement'], errors='coerce')

# Calculate simple metrics
stats = df.groupby('partner_clean').agg({
    'placement': ['count', 'mean'],
    'celebrity_name': lambda x: (df.loc[x.index, 'placement'] == 1).sum()
}).reset_index()

stats.columns = ['Partner', 'Appearances', 'Avg_Placement', 'Wins']

# Calculate "Success Score" (1 = win, 0 = last place)
df['season_max'] = df.groupby('season')['placement'].transform('max')
df['success_score'] = 1 - ((df['placement'] - 1) / (df['season_max'] - 1))

# Average Success Score per partner
success_avg = df.groupby('partner_clean')['success_score'].mean().reset_index()
success_avg.columns = ['Partner', 'Avg_Success_Score']

# Merge
stats = stats.merge(success_avg, on='Partner')

# Sort by Avg Success Score (which weights wins heavily)
stats = stats.sort_values('Avg_Success_Score', ascending=False)

print("\n=== PARTNER RANKINGS (By Average Success Score) ===")
print(stats.to_string(index=False))

# Save
stats.to_csv("results/partner_simple_ranking.csv", index=False)
print("\nSaved to results/partner_simple_ranking.csv")
