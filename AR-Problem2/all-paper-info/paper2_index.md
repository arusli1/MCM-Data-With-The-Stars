# Problem 2: Paper Info Index

All figures, tables, and explanations for the paper live here.

## Regenerate

```bash
cd AR-Problem2
python3 problem2a.py
python3 problem2a_model_A.py   # Model A: week-by-week, no simulation
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
| `paper_2a_explanation.md` | Forward simulation; phantom survivors use zeros; rank favors fans. |
| `paper_2b_explanation.md` | 2b methods (incl. GMM), assumptions, results, limitations, conclusions |
| `paper_2c_recommendation.md` | 2c: Rank vs Percent recommendation, judge-save, creative rule proposals |
| `problem2_questions_coverage.md` | Maps problem questions → files; direct answers |
| `problem2a_part1_table.md` | 34 seasons: Kendall τ, displacement, winner same, top4 same |
| `problem2a_part2_table.md` | Fan advantage summary, rank favor magnitude, Part 3 bottom-2 |
| `problem2b_table.md` | Full list of 21 controversial contestants with regime placements |
| *(moved to archive/)* `threshold_sensitivity_summary.md` | Controversy count vs threshold |
| `methods_overview.md` | Brief methods: rank, percent, fan advantage, bottom-2, GMM |
| *(moved to archive/)* `simulation_divergence_limitation.md` | Limitation: phantom survivors; brainstormed solutions |
| *(moved to archive/)* `alternative_models_2a_brainstorm.md` | Alternative models for 2a brainstorm |
| `problem2a_part1_displacement.pdf` | Part 1: displacement bars + Kendall τ |
| `problem2a_evolution.pdf` | Part 2: fan advantage over time |
| `problem2a_combined_evolution_bottom2.pdf` | (a) Rank vs Percent fan advantage. (b) Judge-save effect Δ on Rank |
| `problem2b_controversy_cdf.pdf` | CDF of controversy score; GMM cutoff |
| `problem2b_controversy_scatter.pdf` | Judge vs placement percentile scatter |
| `problem2b_regime_controversy_by_type.pdf` | Mean simulated controversy by regime |
| `viewership_controversy_scatter.pdf` | Mean controversy vs viewership scatter |
| `viewership_controversy_by_season.pdf` | Viewership + controversy over time |
| `paper_viewership_controversy.pdf` | **Paper-ready:** 2-panel (a) viewership+controversy over time, (b) residual scatter |
| `paper_viewership_controversy_scatter.pdf` | **Paper-ready:** single-panel scatter with partial r |
| `outputs/viewership_controversy_summary.md` | Correlation stats, interpretation |

## Suggested Paper Wording

**2a — Combination method:**
> We applied both combination methods (rank and percent) to each season using forward simulation and fan-share trajectories from Problem 1 (Data/estimate_votes.csv). Phantom survivors use zeros. Same winner in 88% of seasons; same top 4 in 71%. Rank favors fans more than percent (rank favor magnitude 0.81 displacement units; 29/34 seasons). Percent is heavily judge-dominated (fan advantage −0.97). Judge-save decreases fan adv in Rank 25/34 seasons.

**2b — Controversy and judge-save:**
> We identified controversial contestants (extreme judge–placement disagreement) using a 2-component Gaussian mixture model. For each we ran 4 regimes: rank vs percent × judge-save vs fan-decide. For the four named examples (Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones): method choice matters for Billy Ray, Bristol Palin, and Bobby Bones (percent matches actual outcome). Judge-save impacted 2 of 21 controversial contestants (Iman Shumpert, Whitney Leavitt); 6 of 44 bottom-two weeks.
