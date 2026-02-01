# Problem 2a: Method Comparison — Paper Explanation

## Question
Compare rank vs percent combination methods. Does one favor fan votes more?

---

## Methods

### Part 1: Rank vs Percent Outcome Differences

**Simulation:** For each season, run both methods (no bottom-two rule):
- **Rank:** Combined score = judge_rank + fan_rank. Eliminate highest (worst).
- **Percent:** Combined score = judge_pct + fan_share. Eliminate lowest.

**Metrics:**
- **Kendall τ distance:** Fraction of contestant pairs reordered between methods.
- **Mean displacement:** Average |placement_rank − placement_percent|.
- **Winner same:** Boolean.
- **Top 4 same:** Boolean.

### Part 2: Which Input Dominates?

**Simulation:** For each method, run 3 scenarios:
1. **Just judges:** Eliminate by judge score only.
2. **Combined:** Normal rule (judge + fan).
3. **Just fans:** Eliminate by fan vote only.

**Metric:** Displacement from combined outcome.
- Smaller displacement = that input dominates (combined is closer to that input alone).

**Fan Advantage** = disp(judges→combined) − disp(fans→combined).
- Positive = fans dominate.
- Negative = judges dominate.

**Comparison:** If fan advantage higher under rank than percent → rank favors fans more.

---

## Key Results

### Part 1
- Mean Kendall τ: **0.082** (8.2% of pairs reordered)
- Mean displacement: **0.80** positions
- Same winner: **26/34 (76%)**
- Same top 4: **23/34 (68%)**

### Part 2
| Method | Mean Fan Advantage | Fans Dominate |
|--------|-------------------|---------------|
| Rank | −0.23 | 8/34 (24%) |
| Percent | −0.97 | 1/34 (3%) |

**Rank favors fans more:** 24/34 seasons (71%).

---

## Interpretation

**Why does rank favor fans more?**

- **Rank:** Normalizes both inputs to same scale (ranks 1 to N). Equal contribution.
- **Percent:** Uses raw scores. Judge scores are concentrated and predictive; fan shares are dispersed. Judges dominate the sum.

**Practical impact:** Both methods usually agree on winner (76%), but rank gives fans more influence in determining the full ranking.

---

## Figures

1. **problem2a_part1_displacement.pdf:** Displacement bars (green=same winner, red=different) + Kendall τ line (purple).
2. **problem2a_evolution.pdf:** Fan advantage time series. Blue=rank, orange=percent. Filled regions show which favors fans.

## Tables

1. **problem2a_part1_table.md:** All 34 seasons with metrics.
2. **problem2a_part2_table.md:** Fan advantage summary.
