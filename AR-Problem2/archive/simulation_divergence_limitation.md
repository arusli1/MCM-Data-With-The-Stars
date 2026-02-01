# Simulation Divergence Limitation — Problem & Solutions

## The Problem

When we simulate eliminations under rank vs percent (or judge-save vs fan-decide), the simulation can **diverge** from the actual elimination order in week 1. For example:

- **Reality:** A, B, C active → B eliminated week 1 → week 2 active = {A, C}
- **Simulation (rank):** A, B, C active → C eliminated week 1 → week 2 active = {A, B}

**We do not have data for B in week 2.** Fan shares and judge scores are inferred from the *actual* elimination sequence. For B:
- Judge score week 2: 0 or missing (B didn't perform)
- Fan share week 2: 0 (inference assigns 0 to eliminated contestants)

So when the sim keeps B "alive," we use zeros. B will almost always rank worst and get eliminated in week 2. **The model effectively forces B to lose**—not because we believe B would have performed worst, but because we have no alternative data. We are not simulating counterfactual performance; we are defaulting to "no data → treat as worst."

### Why This Matters

1. **Bias:** Phantom survivors are systematically eliminated as soon as we lack data. The sim "snaps back" toward reality by construction.
2. **Invalid counterfactuals:** When divergence occurs, subsequent eliminations are driven by placeholder zeros, not real (or even modeled) preferences.
3. **Uncertainty in conclusions:** Our rank vs percent comparisons may be more reliable when the sim stays close to reality. High-divergence seasons are less trustworthy.

---

## Brainstormed Solutions

### 1. New Models for Fan Share (Imputation / Prediction)

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Predict from covariates** | Train a model (e.g., regression) to predict fan share from judge score, demographics, prior-week share. Use predicted share for phantom B in week 2. | Uses observable info; less arbitrary than 0 | Speculative; adds model assumptions; may not generalize |
| **Extrapolate from last known week** | Use B's week 1 fan share (or decayed version) as proxy for week 2. | Simple; assumes continuity | Ignores week 2 performance; may overstate B's share |
| **Uniform among unknown** | Give phantom B share = 1 / |active| (assume average among active). | Not punitive; symmetric | May understate variance; assumes no info |
| **Bayesian prior** | Place weakly informative prior (e.g., Dirichlet with small concentration) over phantom shares. | Principled uncertainty | Complex; still needs a prior |
| **Inference conditional on sim path** | Re-run inverse optimization *assuming* the sim's elimination order (e.g., C out week 1). Infer shares for A, B in week 2 under that path. | Theoretically coherent | Computationally heavy; requires new inverse problem per divergent path |

---

### 2. New Simulation Approaches

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Stay-aligned / bounded divergence** | Only allow divergence when methods disagree within a "close call" (e.g., bottom-2 have nearly identical combined scores). Otherwise, follow reality. | Reduces phantom survivors | Arbitrary threshold; may miss interesting disagreements |
| **Re-sync on divergence** | When sim would need data we don't have, mark "divergence occurred" and **re-sync** to reality's active set. Treat phantom as eliminated that week; continue with reality's pool. | Avoids using zeros | Mixes sim + reality; placement logic becomes inconsistent (who "won" under sim?) |
| **Stop at divergence** | Halt simulation at first divergence. Report partial metrics (e.g., "agree on eliminations through week 3"). | Honest; no fake data | Incomplete; many seasons may diverge early |
| **Single-week counterfactual** | Only analyze weeks where rank vs percent would eliminate *different* people from the *same* bottom-2. No need to simulate past that week. | Focused on controversial moments | Narrow scope; doesn't give full placement comparison |
| **Rollback / branch** | When phantom appears, don't use 0. Instead, record "cannot simulate past week w with this order" and report confidence-weighted metrics. | Transparent | Requires new metric design |

---

### 3. New Metrics

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Divergence-weighted metrics** | Weight results by inverse divergence: seasons where sim matches reality get full weight; high-divergence seasons get down-weighted or excluded. | Downplays unreliable cases | Need to define divergence measure; may shrink sample |
| **First-disagreement analysis** | Focus on the *first* week where rank and percent would eliminate different people. No need to simulate beyond that. | Avoids phantom problem | Doesn't yield full placement comparison |
| **Week-by-week agreement** | Count: in how many weeks do rank and percent eliminate the *same* person? Metric = fraction of weeks in agreement. | Uses less counterfactual structure | Different from "final placement" comparison |
| **Controversy-focused** | Restrict to seasons/weeks with "controversial" contestants (2b list). These are where methods might differ; non-controversial seasons likely agree anyway. | Targets policy-relevant cases | Excludes quiet seasons |
| **Bootstrap / sensitivity** | Re-run with different imputations (0 vs uniform vs extrapolate). Report range of conclusions. | Shows robustness | Doesn't fix underlying issue |

---

### 4. Structural / Procedural Changes

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Report divergence rate** | For each season, compute week-by-week agreement between sim and reality. Flag high-divergence seasons in tables/figures. | Transparent; readers can judge | Doesn't fix simulation |
| **Exclude high-divergence seasons** | Drop seasons where sim diverges heavily. Report rank vs percent only on "clean" subset. | Cleaner inference | Smaller N; possible selection bias |
| **Two-tier analysis** | Tier 1: Full simulation (current approach) with explicit caveat. Tier 2: Conservative analysis (e.g., first-disagreement only) with stronger claims. | Balances breadth and rigor | More work; two narratives |

---

## Recommendation (Preliminary)

1. **Short term:** Add the limitation to the paper (2a, methods_overview) and report a **divergence statistic** (e.g., fraction of weeks where sim elimination order matches reality, by season). Down-weight or flag high-divergence seasons in interpretation.
2. **Medium term:** Implement **sensitivity analysis** with alternative imputation (e.g., uniform for phantoms) and report whether conclusions hold.
3. **Longer term:** Explore **first-disagreement** or **week-by-week agreement** metrics that avoid simulating past divergence, and/or **imputation models** for phantom fan shares if we want full placement comparison with less bias.

---

## Implementation Notes

- **Divergence measure:** For each season, week w: `diverged[w] = 1` if sim's week-w elimination ≠ reality's. Season divergence rate = mean(diverged) or first-week divergence.
- **Phantom detection:** Contestant i is "phantom" in week w if: elim_week_sim[i] > w and elim_week_true[i] ≤ w. When we use s_hist[w,i] or J[w,i] for such i, we're using zeros.
- **Data dependency:** Fan shares come from `Data/estimate_votes.csv` (Problem 2 canonical source). The shares are inferred and condition on the actual elimination order. No counterfactual shares exist without re-running inference or imputation.
