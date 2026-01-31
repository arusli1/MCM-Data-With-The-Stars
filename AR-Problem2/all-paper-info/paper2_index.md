# Problem 2: Paper Info Index

All figures, tables, and explanations for the paper live here.

## Regenerate

```bash
cd AR-Problem2
python3 problem2a.py
python3 problem2b_controversy.py
python3 sensitivity_analysis/run_threshold_sensitivity.py
python3 fetch_wiki_viewership.py   # → Data/dwts_viewership.csv
python3 viewership_controversy_analysis.py   # → outputs/, all-paper-info/
```

Outputs: `outputs/`, `figures/`, `sensitivity_analysis/outputs/`, `sensitivity_analysis/figures/`.

## Files

| File | Description |
|------|-------------|
| `paper_2a_explanation.md` | 2a methods, assumptions, results, limitations, conclusions, implications |
| `paper_2b_explanation.md` | 2b methods (incl. GMM), assumptions, results, limitations, conclusions |
| `paper_2c_recommendation.md` | 2c: Rank vs Percent recommendation, judge-save, creative rule proposals |
| `problem2_questions_coverage.md` | Maps problem questions → files; direct answers |
| `problem2a_part1_table.md` | 34 seasons: Kendall τ, displacement, winner same, top4 same |
| `problem2a_part2_table.md` | Fan advantage summary, rank favor magnitude, Part 3 bottom-2 |
| `problem2b_table.md` | Full list of 21 controversial contestants with regime placements |
| `threshold_sensitivity_summary.md` | Controversy count vs threshold (sensitivity) |
| `methods_overview.md` | Brief methods: rank, percent, fan advantage, bottom-2, GMM |
| `problem2a_part1_displacement.pdf` | Part 1: displacement bars + Kendall τ |
| `problem2a_evolution.pdf` | Part 2: fan advantage over time |
| `problem2a_combined_evolution_bottom2.pdf` | (a) Rank, Rank judge-save, Percent. (b) Rank judge-save effect Δ |
| `problem2b_controversy_cdf.pdf` | CDF of controversy score; GMM cutoff |
| `problem2b_controversy_scatter.pdf` | Judge vs placement percentile scatter |
| `problem2b_regime_controversy_by_type.pdf` | Mean simulated controversy by regime |
| `viewership_controversy_scatter.pdf` | Controversy count vs mean viewership (r ≈ 0) |
| `viewership_controversy_by_season.pdf` | Viewership + controversy bars by season |
| `outputs/viewership_controversy_summary.md` | Correlation stats, interpretation |

## Suggested Paper Wording

**2a — Combination method:**
> We applied both combination methods (rank and percent) to each season using the same judge scores and estimated fan-share trajectories from Problem 1. In most seasons the winner is the same under both rules. For each method, we ran 3 simulations (just judges, combined, just fans) and measured which input dominates; rank gives fans more advantage than percent (mean rank favor magnitude 0.75 displacement units). Part 3 applies bottom-2 to k=1 weeks for all 34 seasons (ignoring historical regimes); judge-save decreases fan advantage vs fan-decide in 22/34 seasons.

**2b — Controversy and judge-save:**
> We identified controversial contestants (extreme judge–placement disagreement) using a 2-component Gaussian mixture model. For each we ran 4 regimes: rank vs percent × judge-save vs fan-decide. Method choice has minimal impact; controversial outcomes are driven by large judge-fan disagreement, not the combination rule. Judge-save in bottom-two weeks changed who was eliminated in 6 of 44 such weeks.
