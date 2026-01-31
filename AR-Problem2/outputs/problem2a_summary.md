# Problem 2a Summary: Rank vs Percent Methods

## Key Findings

### Part 1: Rank vs Percent Outcome Differences

**Method:** Simulate **pure rank** (judge_rank + fan_rank) and **pure percent** (judge_pct + fan_share) on all 34 seasons, **without bottom-two logic**. Compare outcomes.

**Across 34 seasons:**
- Mean Kendall tau distance: **0.082** (8.2% of contestant pairs reordered)
- Mean placement displacement: **0.80** positions
- Same winner: **26/34** seasons (76.5%)
- Same top 4: **23/34** seasons (67.6%)

**Pattern:** Differences are **small and relatively uniform** across all seasons. No dramatic increase in S28+ because we're not using the bottom-two rule—just comparing the two basic combination methods.

**Seasons with largest differences:**
- S21: Kendall tau = 0.205, displacement = 1.85 (different winner)
- S18: Kendall tau = 0.197, displacement = 1.83 (same winner)
- S8: Kendall tau = 0.192, displacement = 2.00 (same winner)

**Seasons with no difference:**
- S1, 2, 6, 7, 26: Kendall tau = 0.000 (identical outcomes)

---

### Part 2: Which Input Dominates? (Fans vs Judges)

**Method:** For each method (rank/percent), run 3 simulations:
1. **Just judges** — Eliminate by judge score/rank only
2. **Combined** — Normal rule (judge + fan)
3. **Just fans** — Eliminate by fan vote only

**Measure:** Displacement from combined outcome. **Smaller displacement = that input dominates** (combined is closer to that input alone).

**Fan advantage** = `disp(judges→combined)` − `disp(fans→combined)`. Positive = fans dominate (combined closer to "just fans").

#### Results:

| Metric | Rank | Percent |
|--------|------|---------|
| **Mean fan advantage** | **−0.23** | **−0.97** |
| **Fans dominate** | 8/34 seasons (23.5%) | 1/34 seasons (2.9%) |
| **Judges dominate** | 26/34 seasons (76.5%) | 33/34 seasons (97.1%) |
| **Rank favors fans more** | **24/34 seasons (70.6%)** | — |

**Interpretation:**
- **Judges dominate both methods**, but **judges dominate percent MUCH MORE** (mean fan adv = −0.97 vs −0.23).
- **Rank favors fans more** in **70.6% of seasons** (24/34): fan advantage is higher under rank than under percent.
- In **percent**, the combined outcome is almost always closer to "just judges" than to "just fans" (only 1 season where fans dominate).
- In **rank**, fans dominate in 23.5% of seasons (8/34), and the mean fan advantage is 4× higher than in percent.

#### Why does percent favor judges more?

**Rank:** Judge rank + fan rank. Both inputs are normalized to the same scale (1 to N), so they contribute equally.

**Percent:** Judge % + fan %. 
- Judge scores are typically **more concentrated** (e.g., top dancers get 35/40, bottom get 25/40 → after normalization, judge % ranges from ~0.20 to ~0.30).
- Fan votes are **more dispersed** (fan favorites can get 30–40% share, others get 5–10% → fan % ranges from 0.05 to 0.40).

When you add two distributions, you might expect the **higher-variance input to dominate the sum**. But here, **judges dominate percent** because:
1. **Judge scores are more predictive of elimination risk** — Low judge score = almost certainly eliminated early (judges agree on who's a bad dancer).
2. **Fan shares are noisier and more variable** — Fan favorites can survive despite low judge scores, but this is rare. Most eliminations follow judge rankings.
3. **The combined score (judge % + fan %) is still driven by judges** because judge scores are more consistent week-to-week, while fan shares fluctuate.

In **rank**, both inputs are normalized to the same scale, so fans have more relative influence. A contestant ranked 10th by judges and 2nd by fans gets rank sum = 12. A contestant ranked 5th by judges and 7th by fans gets rank sum = 12. Fans can "pull" a contestant up more easily under rank.

In **percent**, a contestant with judge % = 0.20 and fan % = 0.35 gets combined = 0.55. A contestant with judge % = 0.28 and fan % = 0.15 gets combined = 0.43. The judge % matters less (only 0.08 difference) compared to the fan % (0.20 difference), but in practice, **judges dominate** because low judge scores are more predictive of elimination (contestants with low judge scores rarely have high enough fan shares to compensate).

---

## Conclusion

**Does one method favor fan votes more?**

**Yes: Rank favors fan votes more than percent.**

- **Rank:** Fans dominate in 23.5% of seasons; mean fan advantage = −0.23.
- **Percent:** Fans dominate in 2.9% of seasons; mean fan advantage = −0.97.
- **Rank favors fans more** in 70.6% of seasons (24/34).

**Why?** Rank normalizes both inputs to the same scale (ranks 1 to N), giving fans equal weight. Percent uses raw scores/shares, where judge scores are more concentrated and predictive, so judges drive outcomes more.

**Practical impact:** 
- The two methods produce the **same winner in 76.5%** of seasons (26/34).
- The two methods produce the **same top 4 in 67.6%** of seasons (23/34).
- Mean placement displacement is only **0.80 positions** (small practical difference).
- **Rank gives fans more influence**, but judges still dominate both methods overall (judges dominate in 76.5% of rank seasons vs 97.1% of percent seasons).
