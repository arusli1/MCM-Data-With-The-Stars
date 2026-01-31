# Problem 2a: Method Comparison — Full Explanation

## Question
Compare rank vs percent across seasons. Does one method favor fan votes more than the other?

---

## Methods

### Part 1 — Rank vs Percent Comparison
- **Rank:** Combined score = judge_rank + fan_rank (within active contestants each week). Eliminate contestant(s) with **largest** combined rank.
- **Percent:** Combined score = judge_pct + fan_share (both normalized to sum to 1 among active). Eliminate contestant(s) with **lowest** combined score.
- **Metrics:** Kendall τ (fraction of discordant pairs in final order), mean displacement (|placement_rank − placement_percent|), same winner, same top 4.

### Part 2 — Fan Advantage (Input Dominance)
- For each method, run 3 simulations: (1) judges only, (2) combined (normal rule), (3) fans only.
- **Fan Advantage** = disp(judges→combined) − disp(fans→combined).
- **disp(A,B)** = mean |placement_A − placement_B| over contestants. Smaller disp = A and B are more similar.
- Positive fan advantage = combined outcome is closer to fans-only than judges-only (fans dominate).
- **Rank favor magnitude** = fan_advantage_rank − fan_advantage_percent. Interpretable effect size: how much more fan advantage rank gives than percent (in displacement units).

### Part 3 — Bottom-2 Judge-Save Effect
- In weeks with k=1 elimination, the bottom two by combined score face elimination. A tie-break picks who goes home.
- **Judge-save:** Tie-break = eliminate the one with **lower judge score** (judges pick).
- **Fan-decide:** Tie-break = eliminate the one with **lower fan share** (fans pick).
- Apply bottom-2 to k=1 weeks for **all 34 seasons** (ignoring historical regimes). Compare fan advantage: no bottom-2 vs judge-save, for Rank and Percent separately.

**Key finding:** Percent judge-save has no effect (Δ = 0 for all seasons) because in every k=1 week, the contestant with lowest combined also has the lowest judge score among the bottom 2, so no-B2 and judge-save eliminate the same person. Rank judge-save decreases fan advantage in 21/34 seasons. Season 18 is an outlier: Rank judge-save increases fan advantage (Δ ≈ +0.83).

---

## Assumptions

1. **Fan-share estimates:** We use inferred fan shares from Problem 1 (base model). These are estimates, not observed votes.
2. **Elimination schedule:** We use the judge-based elimination schedule (k per week) for all simulations. The number eliminated each week is fixed; only *who* is eliminated varies by method.
3. **No strategic voting:** We assume fan shares reflect true preferences; no gaming of the vote.
4. **Static judge scores:** Judge scores per week are taken as given from the data.
5. **Same contestants:** We do not model withdrawals, injuries, or late additions beyond what the data encodes.

---

## Results

| Metric | Value |
|--------|-------|
| Mean Kendall τ | 0.082 |
| Mean displacement | 0.80 |
| Same winner | 26/34 (76%) |
| Same top 4 | 23/34 (68%) |
| Rank favors fans more | 24/34 (71%) |
| Fan advantage (rank) | −0.23 |
| Fan advantage (percent) | −0.97 |
| **Rank favor magnitude** | **0.75 ± 0.94** displacement units |

**Part 3:** Judge-save decreases fan adv (Rank): 21/34 seasons. Percent: 0/34 (no effect). Mean fan adv: Rank no-B2 −0.26, Rank judge-save −0.57; Percent −0.94 (identical with or without judge-save).

---

## Limitations

1. **Fan-share uncertainty:** Inferred fan shares have uncertainty; robustness to alternative share estimates is not fully explored.
2. **Schedule dependence:** Results depend on the judge-derived elimination schedule. A different schedule (e.g., fan-derived) could change outcomes.
3. **Single metric:** Fan advantage is one summary; other metrics (e.g., winner stability, distribution of placements) could yield different conclusions.
4. **Season 18 outlier:** Rank judge-save increases fan advantage in S18; some weeks have J=0 (withdrawn/missing data), which may affect tie-breaks.

---

## Conclusions

- **Rank favors fans more than percent.** Mean rank favor magnitude 0.75 displacement units; rank outcomes are on average ~0.75 positions closer to fan-only than percent outcomes.
- **Percent is more judge-dominated.** Fan advantage under percent (−0.97) is more negative than under rank (−0.23).
- **Judge-save tilts toward judges** for Rank (21/34 seasons); for Percent it has no effect because the worst-by-combined always has the worst judge score among the bottom 2.

---

## Implications and Discussion

- **For show design:** If the goal is to give fans more influence, the rank combination method is preferable to percent. Percent’s additive structure (judge_pct + fan_share) lets judge scores dominate when they are more variable.
- **For interpretation:** The 76% winner agreement and 0.80 mean displacement suggest moderate but non-trivial differences. In ~1 in 4 seasons, the winner would change under the other method.
- **Bottom-2:** Adding judge-save in bottom-two weeks strengthens judge influence under Rank but does nothing under Percent—a structural property of the data, not a design choice.

---

## Figures

| File | Description |
|------|-------------|
| `problem2a_part1_displacement.pdf` | Displacement bars by season; Kendall τ. |
| `problem2a_evolution.pdf` | Fan advantage over time. Blue=rank, orange=percent. |
| `problem2a_combined_evolution_bottom2.pdf` | (a) Rank, Rank judge-save, Percent. (b) Rank judge-save effect Δ. |

## Tables
- `problem2a_part1_table.md` — 34 seasons: Kendall τ, displacement, winner same, top4 same.
- `problem2a_part2_table.md` — Fan advantage summary, rank favor magnitude, Part 3 bottom-2.
