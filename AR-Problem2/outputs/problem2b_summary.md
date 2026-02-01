# Problem 2b Summary: Per-Contestant Outcomes Under 2×2 Regimes

**Controversy type** (by diagonal): **above line** = **fan_favored** (placement > judge — fans kept them further than judges rated); **below line** = **judge_favored** (placement < judge — judges liked them, eliminated early). Of the 20: **13 fan_favored**, **8 judge_favored**. See `controversy_type` in problem2b_controversy_classified.csv and problem2b_2x2_scenarios.csv; scatter plot colors by type.

For each **controversial** contestant we run 2×2 = 4 regimes: rank vs percent × judge_save (bottom-2: eliminate lower judge) vs fan_decide (eliminate lower fan share). We ask, **for that individual**:

## 1. Would the choice of method (rank vs percent) have led to the same result for this contestant?

- **For 10 of 21** controversial contestants, **yes**: their placement (and elim week) would be the same under rank as under percent (holding judge_save vs fan_decide fixed).
- For the other **11**, **no**: their outcome would differ under rank vs percent.
- **10** of 21 had the **same outcome in all 4** regimes (placement and elim week identical across rank/percent and judge_save/fan_decide).

## 2. How would including judge-save (judges choose which of bottom two to eliminate) impact the results for this contestant?

- **For 1 of 21** controversial contestants, **judge-save vs fan-decide would have changed that contestant's outcome**: under rank and/or under percent, their placement or elimination week differs when the bottom-two tie-break is judge vs fan.
- For the other **20**, judge-save vs fan-decide would **not** have changed their outcome (same placement and elim week under judge_save and fan_decide).

Per-contestant details: placement and elim_week in each of the 4 regimes, plus `same_result_rank_vs_percent`, `judge_save_impacted_this_contestant`, `same_outcome_all_4`, in **problem2b_2x2_scenarios.csv**.

## Which regime reduces vs increases controversy? (fan_favored vs judge_favored)

For each contestant we compute **simulated controversy** in each of the 4 regimes: \|judge_percentile − placement_percentile_in_that_regime\|. Lower = outcome closer to what judges would suggest.

- **Fan_favored** (above diagonal): Rank (fan-heavy) tends to give them *better* placement than percent (judge-heavy), so **percent regime *increases* controversy** for fan_favored (pushes placement down toward judges). **Rank regime *reduces* controversy** for fan_favored in 13 of 13 cases (best at minimizing simulated controversy).

- **Judge_favored** (below diagonal): Percent (judge-heavy) tends to give them *better* placement than rank, so **percent regime *reduces* controversy** for judge_favored (pushes placement up toward judges). **Percent regime *reduces* controversy** for judge_favored in 1 of 8 cases (best at minimizing simulated controversy).

- **Judge-save** (bottom-two): When judges choose which of the bottom two to eliminate, they eliminate the *lower* judge score → **judge_save systematically helps judge_favored** (high judge score) and **hurts fan_favored** (low judge score) when in the bottom two. The one controversial contestant in a bottom-two season (Whitney Leavitt s34, judge_favored) had judge_save *improve* her outcome (placement 6 with judge_save vs 10 with fan_decide under percent).

- **Sensitivity** (mean placement range across 4 regimes): fan_favored 2.15, judge_favored 0.50. Higher = more sensitive to which regime is used.

**Conclusion:** Rank (fan-heavy) favors fan_favored contestants and reduces their controversy; percent (judge-heavy) favors judge_favored contestants and reduces their controversy. Judge-save in bottom-two weeks further tilts outcomes toward judge_favored. See **problem2b_regime_controversy_by_type.csv** for mean simulated controversy and mean placement by regime and type.
