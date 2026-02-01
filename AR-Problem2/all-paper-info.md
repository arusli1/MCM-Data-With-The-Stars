# Problem 2: All Paper Materials

## Figures (4)

| File | Description |
|------|-------------|
| `figures/problem2a_part1_displacement.pdf` | Part 1: Displacement bars + Kendall τ line. Green=same winner, red=different, hatched=different top4. |
| `figures/problem2a_evolution.pdf` | Part 2: Fan advantage time series. Blue=rank, orange=percent. |
| `figures/problem2b_controversy_scatter.pdf` | Controversy scatter. Green=fan favored, red=judge favored. Labels: JR, BC, BP, BB. |
| `figures/problem2b_regime_controversy_by_type.pdf` | Regime effect on controversy by type. |

---

## Tables for Paper (4)

| File | Description |
|------|-------------|
| `outputs/problem2a_part1_table.md` | 34 seasons: Kendall τ, displacement, winner same, top4 same. |
| `outputs/problem2a_part2_table.md` | Fan advantage summary: rank vs percent comparison. |
| `outputs/problem2b_table.md` | Controversy summary: classification, examples, regime effects. |
| `outputs/problem2b_controversy_classified.csv` | List of 20 controversial contestants (if needed). |

---

## Explanation Files (2)

| File | Description |
|------|-------------|
| `paper_2a_explanation.md` | Full 2a methodology and results for paper writing. |
| `paper_2b_explanation.md` | Full 2b methodology and results for paper writing. |

---

## Key Numbers to Cite

### Problem 2a (Forward simulation — Data/estimate_votes.csv; phantom survivors use zeros)
- Mean Kendall τ: **0.078**
- Mean displacement: **0.76**
- Same winner: **30/34 (88%)**
- Same top 4: **24/34 (71%)**
- Rank favors fans more: **29/34** — rank favor magnitude **0.81** displacement units
- Fan advantage: rank = **−0.16**, percent = **−0.97**
- Judge-save decreases fan adv (Rank): **25/34** seasons

### Problem 2a Model A (archived — week-by-week, no simulation)
- Agreement: **256/261 (98.1%)** elimination weeks
- Disagreement: **5** weeks (seasons 16, 18, 28, 33)
- When disagree: rank favors fans **2** times, percent **0**, tie **3**
- **Fan-favor margin:** mean J(rank_elim) − J(pct_elim) = **18.0** when disagree (positive → rank favors fans)

### Problem 2b
- Controversial contestants: **20**
- Fan favored: **12**, Judge favored: **8**
- Controversy threshold: **0.36**
- Judge-save changed elimination: **6/44 weeks (14%)**

---

## Figure Captions (Draft)

**Figure X (2a Part 1):** Mean placement displacement between rank and percent methods by season. Green bars indicate seasons with the same winner under both methods; red bars indicate different winners. Hatched bars indicate different top-4 finalists. Purple line shows Kendall τ distance (right axis).

**Figure X (2a Part 2):** Fan advantage over time. Positive values indicate fans dominate the combined outcome; negative values indicate judges dominate. Blue = rank method, orange = percent method. Shaded regions highlight which method favors fans more.

**Figure X (2b Scatter):** Judge percentile vs placement percentile for all contestants. Diagonal line represents perfect agreement. Green points = fan favored (above diagonal), red points = judge favored (below diagonal). Labeled examples: JR = Jerry Rice (S2), BC = Billy Ray Cyrus (S4), BP = Bristol Palin (S11), BB = Bobby Bones (S27).

**Figure X (2b Regime):** Mean simulated controversy by voting regime and controversy type. Lower values indicate outcomes closer to judge rankings.
