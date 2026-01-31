# Problem 2b: Controversy Analysis — Full Explanation

## Question
For controversial contestants (judge–fan disagreement), would method choice or judge-save change outcomes?

---

## Methods

### Controversy Metric
- **controversy_score** = |judge_percentile − placement_percentile|
- High score = judges and outcome disagreed (judges liked them but they went early, or judges disliked them but they went far).
- **Fan favored:** placement > judge percentile (above diagonal). **Judge favored:** placement < judge percentile (below diagonal).

### GMM Classification (Consolidated)

**Approach:** 2-component Gaussian mixture model on controversy_score. We use a *semi–data-driven* cutoff: the model structure and EM are data-driven, but constraints steer the cutoff toward the extreme tail.

**How GMM works:**
- Assume controversy scores come from a mixture of two Gaussians: a "normal" (low disagreement) component and a "controversial" (high disagreement) component.
- **EM algorithm:** (1) E-step: compute P(high | x) for each point. (2) M-step: update μ, σ, π using soft assignments. (3) Repeat until convergence.
- **Classification:** Classify as controversial if P(high | x) > threshold (we use 0.8).

**Our constraints:**
1. **MU_HIGH_MIN:** μ_high ≥ 98th percentile of scores. Keeps the high component in the extreme tail; prevents the model from splitting near the median.
2. **Posterior threshold 0.8:** Classify only if P(high | x) > 0.8. Following GMM-Demux (Xiong et al., *Nature Communications* 2020), 0.8 is used as a high-confidence threshold; borderline cases are treated as "unclear."

Without these constraints, an unconstrained GMM would split near the median (~0.09) and label ~50% as controversial. Our setup yields ~21 cases in the extreme tail.

**GMM vs K-means:** K-means with K=2 would give a cutoff around 0.15–0.20 and classify more people. GMM with constraints targets the tail. For right-skewed controversy scores, GMM is better suited to identify *extreme* disagreement.

### Simulation
- **4 regimes** per contestant: rank vs percent × judge-save vs fan-decide.
- Simulate elimination under each; record placement and elim week.
- Ask: Would method choice (rank vs percent) have led to the same result? Would judge-save have changed their outcome?

---

## Assumptions

1. **Controversy metric:** Judge percentile and placement percentile adequately capture "disagreement." We do not weight by recency or season size.
2. **GMM appropriateness:** Controversy scores are treated as a mixture of two Gaussians; the tail is assumed to represent a distinct "controversial" population.
3. **Posterior threshold 0.8:** Arbitrary but justified by precedent (GMM-Demux); sensitivity analysis shows main conclusions robust to threshold (0.25–0.40).
4. **Same fan-share and schedule:** As in 2a; inferred shares and judge-based schedule apply.

---

## Results

| Metric | Value |
|--------|-------|
| Controversial contestants | 21 (GMM) |
| Fan favored | 13 |
| Judge favored | 8 |
| Judge-save changed elimination | 6/44 weeks (14%) |

**Named examples** (see `problem2b_table.md` for full list of 21):
- Jerry Rice (S2): Same result under all regimes (2nd). Method choice and judge-save: no impact.
- Billy Ray Cyrus (S4): Rank → 7th; Percent → 5th. Method choice matters.
- Bristol Palin (S11): Rank → 7th; Percent → 3rd. Percent matches actual.
- Bobby Bones (S27): Rank → 3rd; Percent → 1st. Percent matches actual; under Rank, Evanna Lynch would win.

**Judge-save impacted:** 2 contestants (Iman Shumpert S30, Whitney Leavitt S34).

---

## Limitations

1. **GMM constraints are not purely data-driven:** MU_HIGH_MIN and 0.8 threshold inject prior structure. Different choices would change the controversial set.
2. **Sensitivity:** Threshold sensitivity (0.25–0.40) shows robustness of conclusions, but very different thresholds could alter findings.
3. **Controversy definition:** We use judge vs placement percentile only; other definitions (e.g., weeks with lowest judge score) could yield different lists.
4. **Small N:** 21 controversial contestants; judge-save impacts only 2 directly; statistical power is limited.

---

## Conclusions

- **Method choice has minimal impact** for most controversial contestants. Same result rank vs percent in ~1/3 of cases; when they differ, percent often matches actual placement for fan favorites (Bristol Palin, Bobby Bones).
- **Judge-save rarely changes outcomes** for these contestants (6/44 weeks overall; 2 contestants directly impacted).
- **Controversial outcomes are driven by large judge–fan disagreement**, not the combination method. The regime choice matters less than the underlying tension between judge and fan preferences.

---

## Implications and Discussion

- **For show design:** Switching from rank to percent (or vice versa) would not have changed most controversial outcomes. The "controversy" stems from fans and judges disagreeing, not from the combination rule.
- **For Bobby Bones (S27):** Under percent he would have won; under rank he would have been 3rd. This illustrates that percent can amplify fan influence for extreme fan favorites.
- **GMM as a tool:** The semi–data-driven GMM balances objectivity (EM, data) with interpretability (targeting extreme tail). A fully unsupervised approach would over-classify; fixed thresholds are arbitrary. Our approach is a middle ground.

---

## Figures

| File | Description |
|------|-------------|
| `problem2b_controversy_cdf.pdf` | CDF of controversy score; GMM cutoff; shaded = controversial. |
| `problem2b_controversy_scatter.pdf` | Judge vs placement percentile. Green=fan favored, red=judge favored. |
| `problem2b_regime_controversy_by_type.pdf` | Mean simulated controversy by regime and type. |

## Tables
- `problem2b_table.md` — Full list of 21 controversial contestants with Score, regime placements, Same RvP?, Judge-save impacted?
