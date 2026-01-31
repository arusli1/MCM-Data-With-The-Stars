# Arbitrary Choices in Problem 2a/2b — and Data-Driven Alternatives

## UPDATE: GMM Classification (Implemented)

**Main analysis now uses a data-driven 2-component Gaussian mixture model:**
- GMM constrained: high component mean ≥ 98th percentile (extreme tail).
- Classify as controversial if P(high component | score) > 0.8.
- Yields ~21 controversial; all four known examples included.
- CDF plot (`problem2b_controversy_cdf.pdf`) shows natural break.

**Sensitivity analysis:** See `sensitivity_analysis/` for threshold sensitivity (0.25–0.40).

---

## Historical: How was 0.36 decided?

**Short answer: It wasn’t data-driven.** The threshold was chosen **backward** so the four problem-statement examples are all classified as controversial.

From `problem2b_controversy.py` line 64:

```python
CONTROVERSY_JUDGE_PLACEMENT_CUTOFF = 0.36   # classify if controversy_score >= this. 0.36 gets all four known examples.
```

**Controversy scores of the four canonical examples:**

| Contestant       | Season | Controversy Score |
|------------------|--------|-------------------|
| Jerry Rice       | 2      | 0.444             |
| Billy Ray Cyrus  | 4      | 0.40              |
| Bristol Palin    | 11     | **0.364**         |
| Bobby Bones      | 27     | 0.583             |

So **0.36 is the smallest threshold that keeps Bristol Palin in**. At 0.37 we’d exclude her and keep only 15 contestants.

---

## Other arbitrary choices in 2a and 2b

### Problem 2b
| Choice | Current | Why it’s arbitrary |
|--------|---------|--------------------|
| Controversy threshold | 0.36 | Chosen to include all 4 named examples, not from the data distribution |
| Fan vs judge favored | Diagonal (placement > judge pct) | Natural geometric split, but the same could be done with other axes |
| Judge percentile | Mean judge score rank over weeks | Could use median, weighted by recency, or week-by-week variation |
| Placement percentile | 1 − (placement−1)/(N−1) | Linear; could use rank-based or other scalings |
| Top-K option | `USE_TOP_K=False`, K=2 | Alternative exists but is not used |

### Problem 2a
| Choice | Current | Why it’s arbitrary |
|--------|---------|--------------------|
| Fan advantage | disp(judges→combined) − disp(fans→combined) | Reasonable but not unique; other measures possible |
| “Rank favors fans more” | fan_adv_rank > fan_adv_percent | Binary; could use effect size or a continuous metric |
| Part 1 simulation | No bottom-two rule | Simplifies comparison; could add bottom-two as robustness check |
| Fan share source | Base inferred shares (Problem 1) | Depends on Problem 1 assumptions |

### `problem2_utils.py`
| Choice | Current | Why it’s arbitrary |
|--------|---------|--------------------|
| Season regime cutoffs | s≤2: rank; 3–27: percent; 28+: rank_bottom2 | Based on DWTS rules; cutoffs are fixed, not estimated |
| Uniform fallback | 1/N when fan shares missing | Neutral prior; other priors (e.g., proportional to judges) could be used |

---

## Data-driven alternatives for controversy classification

### 1. Percentile-based threshold
Use the distribution of `controversy_score`:
- **Top 5%** ≈ 21 contestants (score ~0.35) — close to current 20
- **Top 10%** ≈ 42 contestants (score ~0.22)
- **Top 2.5%** ≈ 10–11 contestants (score ~0.40)

Advantage: No reliance on specific examples; fixed share of contestants.

### 2. Top-K per season
Use `USE_TOP_K=True, K=2` (already implemented):  
Top 2 most controversial per season.  
- Uses relative comparison within season, not a global threshold.  
- Produces a fixed number per season, not per data-set.

### 3. Elbow / gap method
Inspect histogram or CDF of `controversy_score` for a natural break (e.g., where density drops sharply).  
Current distribution:  
- Mean 0.11, median ~0.09, max 0.58  
- Clear right tail above ~0.35  
- Gap between ~0.33 and ~0.36 is relatively small; elbow is soft.

### 4. Mixture / clustering
Model scores as a mixture of “normal” vs “controversial” components.  
Classify using posterior probability of belonging to the high-controversy component.

### 5. Validation against known examples
- Keep the 4 examples as a validation set.
- Choose threshold to maximize recall (all 4 included).
- Then check robustness: how many false positives at that threshold, and how sensitive conclusions are to small changes in the cutoff.

### 6. Sensitivity analysis (recommended)
Report results for several thresholds: e.g., 0.30, 0.33, 0.36, 0.40.  
If main conclusions (e.g., method choice has limited impact, judge-save affects few cases) are similar, the choice of threshold is less critical.

---

## Distribution summary (from `problem2b_controversy_list.csv`)

| Metric | Value |
|--------|-------|
| N contestants | 421 |
| Mean controversy_score | 0.111 |
| Median | 0.091 |
| 75th %ile | 0.182 |
| 90th %ile | 0.273 |
| 95th %ile | 0.333 |
| Max | 0.583 |

| Threshold | Count | % of contestants |
|-----------|-------|------------------|
| ≥ 0.30 | 30 | 7.1% |
| ≥ 0.35 | 21 | 5.0% |
| **≥ 0.36** | **20** | **4.8%** |
| ≥ 0.37 | 15 | 3.6% |
| ≥ 0.40 | 13 | 3.1% |

---

## Recommended next steps

1. **Implement sensitivity analysis** — run the 2b analysis at 0.30, 0.33, 0.36, 0.40 and report how key metrics (e.g., judge-save impact, regime comparison) change.
2. **Add percentile-based option** — e.g., top 5% by `controversy_score` as an alternative classification.
3. **Document rationale** — in the paper, state that 0.36 was chosen to include the four named examples and that results are robust to nearby thresholds.
4. **Optional: Top-K analysis** — re-run with top 2 per season and compare outcomes to the threshold-based analysis.
