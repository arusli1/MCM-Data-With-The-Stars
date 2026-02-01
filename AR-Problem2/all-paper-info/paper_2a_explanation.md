# Problem 2a: Method Comparison — Full Explanation

## Question
Compare rank vs percent across seasons. Does one method favor fan votes more than the other?

---

## Methods (Forward Simulation)

We use **forward simulation**: week-by-week elimination under each method (rank vs percent). Phantom survivors—contestants who would have been eliminated under reality but survive under the simulated path—receive **zero** fan share (no data). See `simulation_divergence_limitation.md` for details and caveats.

### Part 1 — Rank vs Percent Outcome Differences
- For each season: simulate full elimination under **rank** (judge_rank + fan_rank) and **percent** (judge_pct + fan_share).
- Compare placements: Kendall τ, mean displacement, same winner?, same top 4?

### Part 2 — Fan Advantage
- For each method: compare judges-only, combined, and fans-only orderings.
- **Fan Advantage** = disp(judges→combined) − disp(fans→combined). Positive = fans dominate.
- **Rank favor magnitude** = fan_adv_rank − fan_adv_percent. Positive = rank favors fans more than percent.

### Part 3 — Bottom-2 Judge-Save Effect
- Apply bottom-two logic to k=1 weeks for all seasons (regime override).
- **Judge-save:** Judges pick who of bottom 2 goes home (eliminate lower judge score).
- **Fan-decide:** Fans pick (eliminate lower fan share).
- Compare fan advantage: no bottom-2 vs judge-save.

---

## Assumptions

1. **Fan-share estimates:** `Data/estimate_votes.csv` (AR-Problem1-Base). Phantom survivors use zeros.
2. **Elimination schedule:** Judge-based k per week. Same schedule for all regimes.
3. **No strategic voting:** Fan shares reflect true preferences.

---

## Results (Forward Simulation; Data/estimate_votes.csv)

| Metric | Value |
|--------|-------|
| Mean Kendall τ | 0.078 |
| Mean displacement | 0.76 |
| Same winner | 30/34 (88%) |
| Same top 4 | 24/34 (71%) |
| Rank favors fans more | 29/34 (85%) |
| Fan advantage (rank) | −0.16 |
| Fan advantage (percent) | −0.97 |
| **Rank favor magnitude** | **0.81 ± 0.79** displacement units |

**Part 3:** Judge-save decreases fan adv: Rank 25/34 seasons, Percent 0/34.

---

## Limitations

1. **Phantom survivors:** When sim diverges from reality, we lack data for survivors. We use zeros—they are eliminated quickly. See `simulation_divergence_limitation.md`.
2. **Fan-share uncertainty:** Inferred shares; robustness to alternatives not fully explored.
3. **Schedule dependence:** Results depend on judge-derived elimination schedule.

---

## Conclusions

- **Rank favors fans more than percent** (rank favor magnitude 0.81; 29/34 seasons).
- Percent is heavily judge-dominated (fan advantage −0.97; 0/34 seasons fans dominate).
- Same winner in 88% of seasons; same top 4 in 71%.
- Judge-save reduces fan advantage under Rank in 25/34 seasons.

---

## Figures

| File | Description |
|------|-------------|
| `problem2a_part1_displacement.pdf` | Displacement bars; Kendall τ. |
| `problem2a_evolution.pdf` | Fan advantage over time. Blue=rank, orange=percent. |
| `problem2a_combined_evolution_bottom2.pdf` | (a) Rank vs Percent fan advantage. (b) Judge-save effect Δ. |
