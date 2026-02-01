
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("../Data/2026_MCM_Problem_C_Data.csv")

# Clean Partner Names
def clean_partner(p):
    if pd.isna(p): return "Unknown"
    top_partners = ["Cheryl Burke", "Derek Hough", "Mark Ballas", "Valentin Chmerkovskiy", "Kym Johnson", "Maksim Chmerkoskiy", "Tony Dovolani", "Karina Smirnoff", "Witney Carson", "Peta Murgatroyd", "Emma Slater", "Sharna Burgess", "Sasha Farber", "Artem Chigvintsev", "Gleb Savchenko"]
    for tp in top_partners:
        if tp in str(p): return tp
    return "Other"

df['partner_clean'] = df['ballroom_partner'].apply(clean_partner)
df['placement'] = pd.to_numeric(df['placement'], errors='coerce')

# Stats
stats = df.groupby('partner_clean')['placement'].agg(['count', 'mean', 'min']).reset_index()
stats.columns = ['Partner', 'Count', 'Avg_Placement', 'Best_Placement']
stats['Wins'] = df[df['placement'] == 1].groupby('partner_clean')['placement'].count().reindex(stats['Partner']).fillna(0)

# Sort by Avg Placement
stats = stats.sort_values('Avg_Placement')
print(stats.to_markdown(index=False))
