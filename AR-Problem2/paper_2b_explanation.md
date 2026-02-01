# Problem 2b: Controversy Analysis — Paper Explanation

## Question
Examine voting methods for "controversial" celebrities (judge-fan disagreement). Would method choice or judge-save rule change outcomes?

---

## Methods

### Defining Controversy

**Data used:** Judge scores and final placement only (no inferred fan votes).

**Metrics:**
- **Judge percentile:** Rank by total judge score (1 = best).
- **Placement percentile:** Final placement (1 = winner).
- **Controversy score:** |judge_percentile − placement_percentile|.

**Classification:** Controversial if score ≥ 0.36 (captures known examples).

**Types:**
- **Fan Favored:** Placement > judge percentile (above diagonal). Low judge score but high placement — fans kept them.
- **Judge Favored:** Placement < judge percentile (below diagonal). High judge score but low placement — eliminated early despite judges' approval.

### 2×2 Regime Simulation

For each controversial contestant, simulate 4 regimes:
1. Rank + Judge Save (bottom-two: judges eliminate lower judge score)
2. Rank + Fan Decide (bottom-two: fans eliminate lower fan share)
3. Percent + Judge Save
4. Percent + Fan Decide

**Metrics per contestant:**
- Placement in each regime
- Elimination week in each regime
- Simulated controversy = |judge_pct − simulated_placement_pct|
- Same result across regimes?

---

## Key Results

### Controversy Classification
- **20 controversial contestants** identified across 34 seasons.
- 12 fan favored, 8 judge favored.

### Known Examples (all fan favored)
| Name | Season | Judge Pct | Placement Pct | Note |
|------|--------|-----------|---------------|------|
| Jerry Rice | 2 | 0.11 | 0.56 | Runner-up despite low scores |
| Billy Ray Cyrus | 4 | 0.00 | 0.40 | 5th despite worst scores |
| Bristol Palin | 11 | 0.00 | 0.36 | 3rd with lowest scores 12× |
| Bobby Bones | 27 | 0.42 | 1.00 | Won despite low scores |

### Method Impact
- **Same result (rank vs percent):** Most controversial contestants have same outcome regardless of method.
- **Judge-save impact:** In 6/44 bottom-two weeks (14%), judge-save changed who was eliminated.

### Regime Effect on Controversy
- Percent slightly reduces simulated controversy for both types.
- Difference is small (~0.02 in mean simulated controversy).

---

## Interpretation

**Fan favored contestants** (e.g., Bobby Bones): Low judge scores but high fan support. They survive under any regime because fan votes are strong enough.

**Judge favored contestants:** High judge scores but eliminated early. Less common; fans override judges.

**Method choice matters less than expected:** Controversial outcomes are driven by large judge-fan disagreement, not the combination method.

**Judge-save has limited impact:** Only affects ~14% of bottom-two decisions.

---

## Figures

1. **problem2b_controversy_scatter.pdf:** Judge percentile vs placement percentile. Diagonal = agreement. Green = fan favored, red = judge favored. Labels: JR, BC, BP, BB.
2. **problem2b_regime_controversy_by_type.pdf:** Mean simulated controversy by regime and type.

## Tables

1. **problem2b_table.md:** Controversy classification, known examples, regime effects, judge-save impact.
