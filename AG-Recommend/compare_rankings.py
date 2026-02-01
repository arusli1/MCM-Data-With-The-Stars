import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau
import os
import re

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
})

def parse_week_cols(df):
    weeks = set()
    for c in df.columns:
        m = re.match(r"week(\d+)_judge\d+_score", c)
        if m:
            weeks.add(int(m.group(1)))
    return sorted(weeks)

def get_judge_sum_matrix(df_season, weeks):
    names = df_season["celebrity_name"].tolist()
    max_w = 0
    for w in weeks:
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
        cols = [c for c in cols if c in df_season.columns]
        vals = df_season[cols].replace("N/A", pd.NA).apply(pd.to_numeric, errors="coerce")
        if vals.notna().any().any():
            max_w = w
    
    W, N = max_w, len(names)
    J = np.zeros((W, N))
    for w in range(1, W + 1):
        cols = [f"week{w}_judge{j}_score" for j in range(1, 5)]
        cols = [c for c in cols if c in df_season.columns]
        vals = df_season[cols].replace("N/A", pd.NA).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        J[w-1] = vals.sum(axis=1).values
    return J

def get_fan_shares(season, names, W):
    path = "/Users/athenagao/Downloads/MCM-Data-With-The-Stars/Data/estimate_votes.csv"
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df = df[df["season"] == season]
    if df.empty: return None
    
    name_to_idx = {name: i for i, name in enumerate(names)}
    S = np.zeros((W, len(names)))
    for _, row in df.iterrows():
        w = int(row["week"])
        name = row["celebrity_name"]
        if w <= W and name in name_to_idx:
            S[w-1, name_to_idx[name]] = row["s_share"]
    
    for w in range(W):
        if S[w].sum() > 0:
            S[w] /= S[w].sum()
    return S

def calculate_alignment(outcome, judge, fan):
    """Calculate how well the outcome aligns with judge vs fan rankings."""
    # Negate outcome if it's 'combined rank' where 1 is best
    # Here we assume 'outcome' is a score (higher is better) for direct correlation
    # If using Kendall Tau on raw values, it handles the ordering.
    
    align_j, _ = kendalltau(outcome, judge)
    align_f, _ = kendalltau(outcome, fan)
    
    return align_j, align_f

def main():
    data_path = "/Users/athenagao/Downloads/MCM-Data-With-The-Stars/Data/2026_MCM_Problem_C_Data.csv"
    df = pd.read_csv(data_path)
    weeks_list = parse_week_cols(df)
    seasons = sorted(df["season"].unique())
    
    balance_results = []
    
    print("Quantifying Balance of Power across regimes...")
    for season in seasons:
        df_s = df[df["season"] == season].copy()
        names = df_s["celebrity_name"].tolist()
        J = get_judge_sum_matrix(df_s, weeks_list)
        W, N = J.shape
        S = get_fan_shares(season, names, W)
        
        if S is None: continue
        
        for w in range(W):
            active = J[w] > 0
            if active.sum() < 3: continue
            
            j_w = J[w][active]
            s_w = S[w][active]
            
            # --- 1. Ranking Regime (Ordinal) ---
            # Outcome = neg(Rank(J) + Rank(S)) -> higher is better
            rj = pd.Series(j_w).rank(method='average', ascending=False)
            rf = pd.Series(s_w).rank(method='average', ascending=False)
            outcome_rank = -(rj + rf)
            
            adj, adf = calculate_alignment(outcome_rank, j_w, s_w)
            
            # --- 2. Percentage Regime (Cardinal/Linear) ---
            # Outcome = Pct(J) + Pct(S)
            j_pct = j_w / j_w.sum() if j_w.sum() > 0 else np.zeros_like(j_w)
            outcome_pct = j_pct + s_w
            
            adj_pct, adf_pct = calculate_alignment(outcome_pct, j_w, s_w)
            
            balance_results.append({
                'season': season,
                'week': w + 1,
                'norm_week': w / (W - 1) if W > 1 else 0,
                'rank_align_j': adj,
                'rank_align_f': adf,
                'pct_align_j': adj_pct,
                'pct_align_f': adf_pct,
                'rank_bias': abs(adj - adf),
                'pct_bias': abs(adj_pct - adf_pct)
            })

    df_balance = pd.DataFrame(balance_results)
    df_balance.to_csv('analysis/judge_fan_balance_of_power.csv', index=False)
    
    # --- Visualization 1: Alignment Over Time ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    
    # Ranking Regime
    sns.lineplot(data=df_balance, x='norm_week', y='rank_align_j', ax=axes[0], color='#3498db', linewidth=3, label='Alignment with Judges')
    sns.lineplot(data=df_balance, x='norm_week', y='rank_align_f', ax=axes[0], color='#e67e22', linewidth=3, label='Alignment with Fans')
    axes[0].set_title('Ranking Regime: Balance of Power', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Kendall Tau ($\tau$) Alignment')
    axes[0].set_xlabel('Normalized Week')
    axes[0].set_ylim(0, 1)
    
    # Percentage Regime
    sns.lineplot(data=df_balance, x='norm_week', y='pct_align_j', ax=axes[1], color='#3498db', linewidth=3, label='Alignment with Judges')
    sns.lineplot(data=df_balance, x='norm_week', y='pct_align_f', ax=axes[1], color='#e67e22', linewidth=3, label='Alignment with Fans')
    axes[1].set_title('Percentage Regime: Balance of Power', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Normalized Week')
    
    plt.suptitle('Whose Opinion Dominates? (Ranking vs Percentage Regimes)', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig('analysis/judge_fan_balance_plot.png', dpi=300, bbox_inches='tight')
    
    # --- Visualization 2: Mean Imbalance (Bias) ---
    plt.figure(figsize=(10, 6))
    bias_data = pd.DataFrame({
        'Regime': ['Ranking', 'Percentage'],
        'Mean Imbalance (Bias)': [df_balance['rank_bias'].mean(), df_balance['pct_bias'].mean()]
    })
    sns.barplot(data=bias_data, x='Regime', y='Mean Imbalance (Bias)', palette=['#95a5a6', '#e74c3c'], edgecolor='black')
    plt.title('Regime Fairness: Which Method Minimizes the Influence Gap?', fontsize=16, pad=20)
    plt.ylabel('Mean Bias (|Judge Align - Fan Align|)', fontsize=12)
    plt.ylim(0, max(bias_data['Mean Imbalance (Bias)']) * 1.5)
    
    # Add labels
    for i, v in enumerate(bias_data['Mean Imbalance (Bias)']):
        plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('analysis/regime_fairness_comparison.png', dpi=300)
    
    print("\n--- Balance of Power Summary ---")
    print(f"Ranking Regime Bias:    {df_balance['rank_bias'].mean():.3f}")
    print(f"Percentage Regime Bias: {df_balance['pct_bias'].mean():.3f}")
    
    if df_balance['rank_bias'].mean() < df_balance['pct_bias'].mean():
        print("\nCONCLUSION: The RANKING REGIME is better at minimizing the power imbalance.")
    else:
        print("\nCONCLUSION: The PERCENTAGE REGIME is better at minimizing the power imbalance.")
        
    print("\nPlots saved to analysis/judge_fan_balance_plot.png and analysis/regime_fairness_comparison.png")

if __name__ == "__main__":
    main()
