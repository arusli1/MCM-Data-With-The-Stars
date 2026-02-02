#!/usr/bin/env python3
"""
Problem 2a: Compare rank vs percent methods across seasons.
Does one method favor fan votes more?

Forward simulation (phantom survivors use zeros). See simulation_divergence_limitation.md.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from problem2_utils import (
    DATA_PATH,
    build_judge_matrix,
    elim_schedule_from_judge,
    forward_simulate,
    forward_simulate_simple,
    forward_simulate_judge_only,
    forward_simulate_fan_only,
    load_fan_shares_for_season,
    parse_week_cols,
)

# Paths (script-relative so runs from repo root or AR-Problem2)
_BASE = Path(__file__).resolve().parent
OUTPUT_DIR = _BASE / "outputs"
FIG_DIR = _BASE / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

# Plot style - publication quality
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
    'lines.markersize': 6,
})


def mean_displacement(place1, place2):
    """Mean absolute displacement in placement."""
    return np.mean(np.abs(np.array(place1) - np.array(place2)))


def kendall_tau_distance(place1, place2):
    """Kendall tau distance (fraction of discordant pairs)."""
    n = len(place1)
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (place1[i] - place1[j]) * (place2[i] - place2[j]) < 0:
                discordant += 1
    total_pairs = n * (n - 1) / 2
    return discordant / total_pairs if total_pairs > 0 else 0.0


def main():
    print("=" * 60)
    print("Problem 2a: Rank vs Percent Methods (Forward Simulation)")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(DATA_PATH)
    weeks = parse_week_cols(df)
    seasons = sorted(df["season"].unique())
    
    # Part 1: Rank vs Percent difference metrics
    print("\n[Part 1] Running rank vs percent on all seasons...")
    part1_rows = []
    
    for season in seasons:
        df_s = df[df["season"] == season].copy()
        names = df_s["celebrity_name"].tolist()
        n = len(names)
        
        J = build_judge_matrix(df_s, weeks)
        W = J.shape[0]
        schedule = elim_schedule_from_judge(J)
        s_hist = load_fan_shares_for_season(season, names, W)
        
        if s_hist is None:
            print(f"  ⚠ Season {season}: No fan shares, skipping")
            continue
        
        # Run rank (simple: no bottom-two logic)
        elim_rank, place_rank = forward_simulate_simple(
            J, s_hist, schedule, regime="rank"
        )
        
        # Run percent (simple: no bottom-two logic)
        elim_pct, place_pct = forward_simulate_simple(
            J, s_hist, schedule, regime="percent"
        )
        
        # Metrics
        kendall = kendall_tau_distance(place_rank, place_pct)
        disp = mean_displacement(place_rank, place_pct)
        
        # Elimination weeks differ
        weeks_differ = np.sum(np.array(elim_rank) != np.array(elim_pct)) / n
        
        # Top 4 same?
        top4_rank = set([i for i, p in enumerate(place_rank) if p <= 4])
        top4_pct = set([i for i, p in enumerate(place_pct) if p <= 4])
        top4_same = (top4_rank == top4_pct)
        
        # Winner same?
        winner_rank = place_rank.index(1) if 1 in place_rank else None
        winner_pct = place_pct.index(1) if 1 in place_pct else None
        winner_same = (winner_rank == winner_pct)
        
        part1_rows.append({
            "season": season,
            "n_contestants": n,
            "kendall_tau_distance": kendall,
            "mean_displacement": disp,
            "frac_weeks_differ": weeks_differ,
            "top4_same": top4_same,
            "winner_same": winner_same,
        })
        
        print(f"  ✓ Season {season}: kendall={kendall:.3f}, disp={disp:.2f}, winner_same={winner_same}")
    
    df_part1 = pd.DataFrame(part1_rows)
    df_part1.to_csv(OUTPUT_DIR / "problem2a_part1_rank_vs_percent.csv", index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR / 'problem2a_part1_rank_vs_percent.csv'}")
    
    # Part 2: Which input dominates?
    print("\n[Part 2] Running 3-regime analysis (just judges, combined, just fans)...")
    part2_rows = []
    
    for season in seasons:
        df_s = df[df["season"] == season].copy()
        names = df_s["celebrity_name"].tolist()
        n = len(names)
        
        J = build_judge_matrix(df_s, weeks)
        W = J.shape[0]
        schedule = elim_schedule_from_judge(J)
        s_hist = load_fan_shares_for_season(season, names, W)
        
        if s_hist is None:
            continue
        
        # === RANK REGIME ===
        # Just judges (rank by judge score only)
        elim_j_rank, place_j_rank = forward_simulate_judge_only(J, schedule)
        
        # Combined (judge rank + fan rank, no bottom-two)
        elim_c_rank, place_c_rank = forward_simulate_simple(
            J, s_hist, schedule, regime="rank"
        )
        
        # Just fans (rank by fan share only)
        elim_f_rank, place_f_rank = forward_simulate_fan_only(s_hist, schedule)
        
        disp_fans_rank = mean_displacement(place_f_rank, place_c_rank)
        disp_judges_rank = mean_displacement(place_j_rank, place_c_rank)
        fan_advantage_rank = disp_judges_rank - disp_fans_rank  # Positive = fans closer
        
        # === PERCENT REGIME ===
        # Just judges
        elim_j_pct, place_j_pct = forward_simulate_judge_only(J, schedule)
        
        # Combined (judge pct + fan share, no bottom-two)
        elim_c_pct, place_c_pct = forward_simulate_simple(
            J, s_hist, schedule, regime="percent"
        )
        
        # Just fans
        elim_f_pct, place_f_pct = forward_simulate_fan_only(s_hist, schedule)
        
        disp_fans_pct = mean_displacement(place_f_pct, place_c_pct)
        disp_judges_pct = mean_displacement(place_j_pct, place_c_pct)
        fan_advantage_pct = disp_judges_pct - disp_fans_pct
        
        # Effect size: how much more does rank favor fans than percent? (in displacement units)
        rank_favor_magnitude = fan_advantage_rank - fan_advantage_pct  # positive = rank favors fans more

        part2_rows.append({
            "season": season,
            "n_contestants": n,
            "disp_fans_to_combined_rank": disp_fans_rank,
            "disp_judges_to_combined_rank": disp_judges_rank,
            "fan_advantage_rank": fan_advantage_rank,
            "fan_dominates_rank": fan_advantage_rank > 0,
            "disp_fans_to_combined_percent": disp_fans_pct,
            "disp_judges_to_combined_percent": disp_judges_pct,
            "fan_advantage_percent": fan_advantage_pct,
            "fan_dominates_percent": fan_advantage_pct > 0,
            "rank_favors_fans_more": fan_advantage_rank > fan_advantage_pct,
            "rank_favor_magnitude": rank_favor_magnitude,
        })
        
        print(f"  ✓ Season {season}: fan_adv_rank={fan_advantage_rank:.3f}, fan_adv_pct={fan_advantage_pct:.3f}")
    
    df_part2 = pd.DataFrame(part2_rows)
    df_part2.to_csv(OUTPUT_DIR / "problem2a_part2_input_dominance.csv", index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR / 'problem2a_part2_input_dominance.csv'}")

    # Part 3: Bottom-2 judge-save effect
    print("\n[Part 3] Bottom-2: fan advantage by method (rank vs percent) and tie-break (no B2 vs judge-save)...")
    part3_rows = []
    for season in seasons:
        season = int(season)
        df_s = df[df["season"] == season].copy()
        names = df_s["celebrity_name"].tolist()
        n = len(names)
        J = build_judge_matrix(df_s, weeks)
        W = J.shape[0]
        schedule = elim_schedule_from_judge(J)
        s_hist = load_fan_shares_for_season(season, names, W)
        if s_hist is None:
            continue

        _, place_j = forward_simulate_judge_only(J, schedule)
        _, place_f = forward_simulate_fan_only(s_hist, schedule)

        _, place_rank_no = forward_simulate(season, names, J, s_hist, schedule, regime_override="rank", force_no_bottom2=True)
        _, place_rank_js = forward_simulate(season, names, J, s_hist, schedule, regime_override="rank", judge_save=True, force_bottom2=True)
        disp_j_rno = mean_displacement(place_j, place_rank_no)
        disp_f_rno = mean_displacement(place_f, place_rank_no)
        fan_adv_rank_no = disp_j_rno - disp_f_rno
        disp_j_rjs = mean_displacement(place_j, place_rank_js)
        disp_f_rjs = mean_displacement(place_f, place_rank_js)
        fan_adv_rank_js = disp_j_rjs - disp_f_rjs

        _, place_pct_no = forward_simulate(season, names, J, s_hist, schedule, regime_override="percent", force_no_bottom2=True)
        _, place_pct_js = forward_simulate(season, names, J, s_hist, schedule, regime_override="percent", judge_save=True, force_bottom2=True)
        disp_j_pno = mean_displacement(place_j, place_pct_no)
        disp_f_pno = mean_displacement(place_f, place_pct_no)
        fan_adv_pct_no = disp_j_pno - disp_f_pno
        disp_j_pjs = mean_displacement(place_j, place_pct_js)
        disp_f_pjs = mean_displacement(place_f, place_pct_js)
        fan_adv_pct_js = disp_j_pjs - disp_f_pjs

        part3_rows.append({
            "season": season,
            "fan_adv_rank_no": fan_adv_rank_no,
            "fan_adv_rank_js": fan_adv_rank_js,
            "fan_adv_pct_no": fan_adv_pct_no,
            "fan_adv_pct_js": fan_adv_pct_js,
            "judge_save_decreases_rank": fan_adv_rank_js < fan_adv_rank_no,
            "judge_save_decreases_pct": fan_adv_pct_js < fan_adv_pct_no,
        })
    df_part3 = pd.DataFrame(part3_rows) if part3_rows else pd.DataFrame()
    if not df_part3.empty:
        df_part3.to_csv(OUTPUT_DIR / "problem2a_part3_bottom2_effect.csv", index=False)
        print(f"✓ Saved: {OUTPUT_DIR / 'problem2a_part3_bottom2_effect.csv'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n[Part 1] Rank vs Percent:")
    print(f"  Mean Kendall tau distance: {df_part1['kendall_tau_distance'].mean():.3f}")
    print(f"  Mean displacement: {df_part1['mean_displacement'].mean():.2f}")
    print(f"  Seasons with same winner: {df_part1['winner_same'].sum()}/{len(df_part1)}")
    print(f"  Seasons with same top 4: {df_part1['top4_same'].sum()}/{len(df_part1)}")
    
    print("\n[Part 2] Input Dominance:")
    print(f"  Seasons where fans dominate under rank: {df_part2['fan_dominates_rank'].sum()}/{len(df_part2)}")
    print(f"  Seasons where fans dominate under percent: {df_part2['fan_dominates_percent'].sum()}/{len(df_part2)}")
    print(f"  Seasons where rank favors fans more: {df_part2['rank_favors_fans_more'].sum()}/{len(df_part2)}")
    print(f"  Mean fan advantage (rank): {df_part2['fan_advantage_rank'].mean():.3f}")
    print(f"  Mean fan advantage (percent): {df_part2['fan_advantage_percent'].mean():.3f}")
    mag = df_part2['rank_favor_magnitude']
    print(f"  Rank favor magnitude (mean ± SD): {mag.mean():.3f} ± {mag.std():.3f} displacement units")
    if not df_part3.empty:
        print("\n[Part 3] Bottom-2 Effect:")
        print(f"  Judge-save decreases fan adv (Rank): {df_part3['judge_save_decreases_rank'].sum()}/{len(df_part3)} seasons")
        print(f"  Judge-save decreases fan adv (Percent): {df_part3['judge_save_decreases_pct'].sum()}/{len(df_part3)} seasons")
    
    # Generate plots and tables
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    summary_lines = [
        "# Part 1: Rank vs Percent Outcome Differences (Forward Simulation)\n",
        "| Season | Kendall τ | Displacement | Winner Same | Top4 Same | Weeks Differ |",
        "|--------|-----------|--------------|-------------|-----------|--------------|",
    ]
    for _, row in df_part1.iterrows():
        winner = "Yes" if row['winner_same'] else "No"
        top4 = "Yes" if row['top4_same'] else "No"
        summary_lines.append(
            f"| {int(row['season'])} | {row['kendall_tau_distance']:.3f} | {row['mean_displacement']:.2f} | {winner} | {top4} | {row['frac_weeks_differ']:.2f} |"
        )
    summary_lines.extend([
        "",
        f"**Mean Kendall τ:** {df_part1['kendall_tau_distance'].mean():.3f}",
        f"**Mean Displacement:** {df_part1['mean_displacement'].mean():.2f} positions",
        f"**Same Winner:** {df_part1['winner_same'].sum()}/34 ({df_part1['winner_same'].sum()/34*100:.0f}%)",
        f"**Same Top 4:** {df_part1['top4_same'].sum()}/34 ({df_part1['top4_same'].sum()/34*100:.0f}%)",
    ])
    with open(OUTPUT_DIR / "problem2a_part1_table.md", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"✓ Saved: {OUTPUT_DIR / 'problem2a_part1_table.md'}")

    mag = df_part2["rank_favor_magnitude"]
    part2_lines = [
        "# Part 2: Which Input Dominates? (Fan Advantage Summary)\n",
        "| Method | Mean Fan Advantage | Fans Dominate | Judges Dominate |",
        "|--------|-------------------|---------------|-----------------|",
        f"| Rank | {df_part2['fan_advantage_rank'].mean():.2f} | {(df_part2['fan_dominates_rank']).sum()}/{len(df_part2)} ({100*(df_part2['fan_dominates_rank']).mean():.0f}%) | {(~df_part2['fan_dominates_rank']).sum()}/{len(df_part2)} ({100*(~df_part2['fan_dominates_rank']).mean():.0f}%) |",
        f"| Percent | {df_part2['fan_advantage_percent'].mean():.2f} | {(df_part2['fan_dominates_percent']).sum()}/{len(df_part2)} ({100*(df_part2['fan_dominates_percent']).mean():.0f}%) | {(~df_part2['fan_dominates_percent']).sum()}/{len(df_part2)} ({100*(~df_part2['fan_dominates_percent']).mean():.0f}%) |",
        "",
        "**Fan Advantage** = (Judges displacement from combined) − (Fans displacement from combined). Positive = fans dominate.",
        "",
        "**Rank Favor Magnitude:** fan_advantage_rank − fan_advantage_percent. ",
        f"Mean = {mag.mean():.3f} ± {mag.std():.3f} displacement units.",
        "",
        f"| Seasons where rank favors fans more | {(df_part2['rank_favors_fans_more']).sum()}/{len(df_part2)} |",
    ]
    if not df_part3.empty:
        part2_lines.extend([
            "",
            "## Part 3: Bottom-2 Judge-Save Effect",
            f"Judge-save decreases fan adv: Rank {(df_part3['judge_save_decreases_rank']).sum()}/{len(df_part3)}, Percent {(df_part3['judge_save_decreases_pct']).sum()}/{len(df_part3)} seasons.",
        ])
    with open(OUTPUT_DIR / "problem2a_part2_table.md", "w") as f:
        f.write("\n".join(part2_lines))
    print(f"✓ Saved: {OUTPUT_DIR / 'problem2a_part2_table.md'}")
    
    # Part 1 Plot
    fig, ax = plt.subplots(figsize=(11, 3.5))
    x = df_part1["season"]
    colors = ['#C1292E' if not w else '#029E73' for w in df_part1['winner_same']]
    bars = ax.bar(x, df_part1['mean_displacement'], color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=0.5, width=0.7)
    for i, (bar, t4) in enumerate(zip(bars, df_part1['top4_same'])):
        if not t4:
            bar.set_hatch('///')
            bar.set_edgecolor('black')
    ax2 = ax.twinx()
    ax2.plot(x, df_part1['kendall_tau_distance'], 'o-', color='#A23B72', 
             linewidth=2, markersize=4, alpha=0.9, markeredgewidth=0)
    ax2.set_ylabel('Kendall τ Distance', fontsize=10, color='#A23B72')
    ax2.tick_params(axis='y', labelcolor='#A23B72')
    ax2.set_ylim(0, df_part1['kendall_tau_distance'].max() * 1.15)
    ax.set_xlabel('Season', fontsize=11)
    ax.set_ylabel('Mean Displacement', fontsize=10)
    ax.set_title('Rank vs Percent: Outcome Differences (Forward Simulation)', fontsize=12)
    ax.set_xlim(0, 35)
    ax.grid(True, alpha=0.2, axis='y')
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#029E73', edgecolor='black', label='Same Winner'),
        Patch(facecolor='#C1292E', edgecolor='black', label='Different Winner'),
        Patch(facecolor='gray', edgecolor='black', hatch='///', label='Different Top 4'),
        Line2D([0], [0], color='#A23B72', marker='o', markersize=5, label='Kendall τ'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
              fancybox=False, edgecolor='black', framealpha=1, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "problem2a_part1_displacement.pdf", dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved: {FIG_DIR / 'problem2a_part1_displacement.pdf'}")
    plt.close()
    
    # Plot 2: Fan advantage over time
    fig, ax = plt.subplots(figsize=(11, 5))
    x = df_part2["season"]
    y_rank = df_part2["fan_advantage_rank"]
    y_pct = df_part2["fan_advantage_percent"]
    ax.plot(x, y_rank, 'o-', color='#0173B2', linewidth=2.5, markersize=5, 
            label='Rank', alpha=0.9, markeredgewidth=0.5, markeredgecolor='white')
    ax.plot(x, y_pct, 's-', color='#DE8F05', linewidth=2.5, markersize=5, 
            label='Percent', alpha=0.9, markeredgewidth=0.5, markeredgecolor='white')
    ax.fill_between(x, y_rank, y_pct, where=(y_rank > y_pct), 
                     color='#0173B2', alpha=0.15, interpolate=True)
    ax.fill_between(x, y_rank, y_pct, where=(y_rank <= y_pct), 
                     color='#DE8F05', alpha=0.15, interpolate=True)
    ax.axhline(0, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Fan Advantage', fontsize=11)
    ax.set_title('Fan Advantage Over Time', fontsize=12)
    ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black', framealpha=1, fontsize=10)
    ax.grid(True, alpha=0.25, axis='y', linewidth=0.8)
    ax.set_xlim(0, 35)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "problem2a_evolution.pdf", dpi=300, bbox_inches='tight', format='pdf')
    print(f"✓ Saved: {FIG_DIR / 'problem2a_evolution.pdf'}")
    plt.close()

    # Combined 2-panel
    if not df_part3.empty:
        x = df_part3["season"]
        effect_rank = df_part3["fan_adv_rank_js"] - df_part3["fan_adv_rank_no"]
        fig2, axes2 = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
        ax_a = axes2[0]
        ax_a.plot(df_part2["season"], df_part2["fan_advantage_rank"], "o-", color="#0173B2", linewidth=2.5, markersize=5, label="Rank")
        ax_a.plot(df_part3["season"], df_part3["fan_adv_rank_js"], "o--", color="#0173B2", linewidth=1.5, markersize=4, alpha=0.9, label="Rank, judge-save")
        ax_a.plot(df_part2["season"], df_part2["fan_advantage_percent"], "s-", color="#DE8F05", linewidth=2.5, markersize=5, label="Percent")
        ax_a.axhline(0, color="#333333", linestyle="--", linewidth=1, alpha=0.7)
        ax_a.set_ylabel("Fan Advantage", fontsize=11)
        ax_a.set_title("(a) Rank vs Percent: Fan advantage", fontsize=11)
        ax_a.legend(loc="lower left", fontsize=9)
        ax_a.grid(True, alpha=0.2)
        ax_a.set_xlim(0, 35)
        ax_b = axes2[1]
        bars_r = ax_b.bar(x, effect_rank, 0.5, color="#0173B2", alpha=0.85, edgecolor="black", linewidth=0.5)
        ax_b.axhline(0, color="#333333", linestyle="-", linewidth=1)
        ax_b.set_xlabel("Season", fontsize=11)
        ax_b.set_ylabel("Judge-save effect (Δ)", fontsize=11)
        ax_b.set_title("(b) Rank: Judge-save effect (Δ < 0 = favors judges)", fontsize=11)
        ax_b.grid(True, alpha=0.2, axis="y")
        ax_b.set_xlim(0, 35)
        plt.tight_layout()
        fig2.savefig(FIG_DIR / "problem2a_combined_evolution_bottom2.pdf", dpi=300, bbox_inches="tight", format="pdf")
        plt.close(fig2)
        print(f"✓ Saved: {FIG_DIR / 'problem2a_combined_evolution_bottom2.pdf'}")

    print("\n" + "=" * 60)
    print("✓ Problem 2a complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
