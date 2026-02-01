# Problem 2a Model A: Week-by-Week Results (No Simulation)

## Method
Model A avoids the simulation-divergence limitation by **never leaving the observed data**. Each week, we use the actual active set (contestants who competed that week). We compare who rank vs percent *would* eliminate—no phantoms, no forward simulation. Fan shares from `Data/estimate_votes.csv` (AR-Problem1-Base, s_share). Fallback: AA-Prob1 `fan_vote_estimates.csv`.

- **Agreement:** Both methods would eliminate the same person(s).
- **Disagreement:** They would eliminate different people. When that happens, we ask: which method eliminated the judge-favored contestant (higher judge score)? That method "favors fans."
- **NEW METRIC — Fan-favor margin:** When they disagree, mean J(rank_elim) − J(pct_elim). Positive = rank eliminates higher-judge person on average → rank favors fans. Quantifies *how much* one method tilts toward fans.

## Regenerate
```bash
cd AR-Problem2
python3 problem2a_model_A.py
```
(Ensure AR-Problem1-Base `base.py` has been run to produce `Data/estimate_votes.csv`, or use AA-Prob1 as fallback.)

## Results (34 seasons, 261 elimination weeks)

| Metric | Value |
|--------|-------|
| Agreement (same person eliminated) | 256 (98.1%) |
| Disagreement | 5 |

**When they disagree — by judge score** (eliminated higher J = favored fans):
| Rank favors fans | Percent favors fans | Tie (J_A = J_B) |
|------------------|---------------------|-----------------|
| 2 (40%) | 0 | 3 |

**When they disagree — by fan share** (eliminated lower S = favored fans):
| Rank favors fans | Percent favors fans | Tie (S_A = S_B) |
|------------------|---------------------|-----------------|
| 2 (40%) | 0 | 3 |

**NEW METRIC — Fan-favor margin (when disagree):**
| Mean J(rank_elim) − J(pct_elim) |
|----------------------------------|
| 18.0 (positive → rank favors fans) |

## Conclusion
**Rank favors fan votes more than percent** under both definitions. When they disagree (5 weeks across seasons 16, 18, 28, 33), rank eliminates the judge-favored person 2 times; percent 0; tie 3. The fan-favor margin of 18.0 indicates that when rank and percent diverge, rank tends to eliminate someone with substantially higher judge scores—consistent with rank giving more weight to fan preferences.

High agreement (98.1%) indicates AA-Prob1’s fan-share estimates are consistent with both combination methods. The few disagreements occur in specific weeks where the methods diverge.

This aligns with the original 2a forward-simulation finding but is **more robust** because it uses only observed data and avoids the phantom-survivor bias.

## Outputs
- `outputs/problem2a_model_A_summary.csv` — Per-season: agreement rate, disagreement count, rank/percent favors fans, mean_judge_margin, mean_fan_margin, first_disagree_week
- `outputs/problem2a_model_A_week_detail.csv` — Per-week detail: who each method would eliminate, judge scores
