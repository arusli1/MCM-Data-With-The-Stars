# Problem 2b: Controversy Summary

## Controversy Classification

| Type | Count | Definition |
|------|-------|------------|
| Fan Favored | 13 | Placement > Judge percentile (above diagonal) |
| Judge Favored | 8 | Placement < Judge percentile (below diagonal) |
| **Total Controversial** | **21** | GMM: P(high|x) > 0.8 |

**Controversy Score** = |judge_percentile − placement_percentile|

## All Controversial Contestants (GMM classification)

| Season | Name | Type | Score | Judge Pct | Place Pct | Actual | Rank | Percent | Same RvP? | JS impacted? |
|--------|------|------|-------|-----------|-----------|--------|------|---------|-----------|--------------|
| 1 | Kelly Monaco | fan | 0.40 | 0.60 | 1.00 | 1 | 1 | 1 | Yes | No |
| 1 | Rachel Hunter | judge | 0.40 | 0.80 | 0.40 | 4 | 4 | 4 | Yes | No |
| 2 | **Jerry Rice** | fan | 0.44 | 0.44 | 0.89 | 2 | 2 | 2 | Yes | No |
| 4 | Shandi Finnessey | judge | 0.40 | 0.50 | 0.10 | 10 | 10 | 10 | Yes | No |
| 4 | Paulina Porizkova | judge | 0.40 | 0.40 | 0.00 | 11 | 11 | 11 | Yes | No |
| 4 | **Billy Ray Cyrus** | fan | 0.40 | 0.20 | 0.60 | 5 | 7 | 5 | No | No |
| 5 | Sabrina Bryan | judge | 0.45 | 0.91 | 0.45 | 7 | 6 | 7 | No | No |
| 5 | Marie Osmond | fan | 0.36 | 0.45 | 0.82 | 3 | 7 | 3 | No | No |
| 7 | Cloris Leachman | fan | 0.42 | 0.08 | 0.50 | 7 | 7 | 7 | Yes | No |
| 11 | **Bristol Palin** | fan | 0.36 | 0.45 | 0.82 | 3 | 7 | 3 | No | No |
| 13 | Chynna Phillips | judge | 0.36 | 0.64 | 0.27 | 9 | 8 | 9 | No | No |
| 17 | Bill Engvall | fan | 0.45 | 0.27 | 0.73 | 4 | 7 | 4 | No | No |
| 19 | Michael Waltrip | fan | 0.42 | 0.08 | 0.50 | 7 | 6 | 7 | No | No |
| 21 | Alek Skarlatos | fan | 0.42 | 0.42 | 0.83 | 3 | 2 | 3 | No | No |
| 24 | Heather Morris | judge | 0.36 | 0.73 | 0.36 | 8 | 6 | 8 | No | No |
| 24 | David Ross | fan | 0.36 | 0.55 | 0.91 | 2 | 3 | 2 | No | No |
| 27 | **Bobby Bones** | fan | 0.58 | 0.42 | 1.00 | 1 | 3 | 1 | No | No |
| 27 | Tinashe | judge | 0.42 | 0.67 | 0.25 | 10 | 9 | 10 | No | No |
| 27 | Joe Amabile | fan | 0.42 | 0.17 | 0.58 | 6 | 13 | 6 | No | No |
| 30 | Iman Shumpert | fan | 0.36 | 0.64 | 1.00 | 1 | 1 | 15 (JS) / 1 (FD) | No | **Yes** |
| 34 | Whitney Leavitt | judge | 0.38 | 1.00 | 0.62 | 6 | 6 | 1 (JS) / 10 (FD) | No | **Yes** |

**Score** = controversy metric \|judge_percentile − placement_percentile\|. **Bold** = named examples. Same RvP? = same result under rank vs percent. JS impacted? = judge-save changed this contestant's outcome.

## Regime Effect on Controversy

| Regime | Fan Favored (mean) | Judge Favored (mean) |
|--------|-------------------|---------------------|
| Rank + Judge Save | 0.31 | 0.28 |
| Rank + Fan Decide | 0.31 | 0.28 |
| Percent + Judge Save | 0.29 | 0.26 |
| Percent + Fan Decide | 0.29 | 0.26 |

**Finding:** Percent reduces simulated controversy slightly more than rank for both types.

## Judge-Save Impact (Seasons 28+)

- Bottom-two weeks analyzed: 44
- Weeks where judge-save changed elimination: **6 (14%)**
