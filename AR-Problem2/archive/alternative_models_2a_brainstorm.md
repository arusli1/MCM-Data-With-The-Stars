# Alternative Models for 2a: Rank vs Percent Comparison

## The Question (from the problem)
> Use your fan vote estimates with the rest of the data to compare and contrast the results produced by the two approaches used by the show to combine judge and fan votes (i.e. rank and percentage) across seasons. Apply both approaches to each season. If differences in outcomes exist, does one method seem to favor fan votes more than the other?

## Core Flaw in Current Approach
The current model **forward-simulates** eliminations week-by-week. Fan shares are inferred *conditional on the actual elimination order*. When the sim diverges from reality, we lack data for "phantom survivors" and default to zeros, forcing them to lose. This biases comparisons and invalidates counterfactual placement.

---

## Alternative Models (Fundamentally Different Approaches)

### Model A: **Week-by-Week, Same Active Set (No Forward Simulation)**
**Idea:** Never leave the observed data. Each week, the active set is fixed (reality). We have judge + fan data for everyone. Compare who rank vs percent *would* eliminate—no phantoms.

**Procedure:**
1. For each (season, week w): active set = contestants with elim_week_true > w.
2. Compute combined rank and combined percent for each active contestant.
3. Who would rank eliminate? (worst combined rank) Who would percent eliminate? (lowest combined score)
4. Compare: same person or different?

**Metrics:**
- **Agreement rate:** Fraction of (season, week) where both methods eliminate the same person.
- **Fan favor (when they disagree):** When methods disagree, which eliminates the judge-favored contestant? Rank does X% of the time, percent does Y%. If rank more often eliminates the judge-favored one → rank favors fans.
- **Margin of victory:** When they agree, how close was the "runner-up"? When they disagree, how large was the judge–fan gap for each eliminated person?

**Pros:** No simulation, no phantoms, uses only observed data. Directly answers "do the methods make different choices?"  
**Cons:** Doesn't yield full placement (who would have won). Doesn't capture cumulative / path-dependent effects.

---

### Model B: **Aggregate Score Re-Ranking (No Elimination Structure)**
**Idea:** Ignore the elimination process. For each contestant, compute their *average* combined score (rank-style and percent-style) over all weeks they competed. Re-rank all contestants by each. Compare the two orderings.

**Procedure:**
1. For each contestant i: over weeks w where i was active, compute judge_rank(w) + fan_rank(w) and judge_pct(w) + fan_pct(w) (within active each week).
2. Average across weeks (or sum, or use last-K-weeks). Get one "rank score" and one "percent score" per contestant.
3. Order contestants by each. Compute Kendall τ, displacement, same winner, etc.

**Pros:** No simulation, no phantoms. Simple. Answers "if we ranked by aggregate combined performance, how do rank vs percent differ?"  
**Cons:** Different interpretation—not "who would the show have crowned?" but "whose overall performance looks better under each metric?" Eliminees have fewer weeks, so early exits may be underweighted. Could weight by weeks survived or use percentile within tenure.

---

### Model C: **First-Disagreement + Marginal Analysis**
**Idea:** Focus on the *moments* where the methods would diverge. When rank and percent would eliminate different people from the *same* bottom-2, we have full data—both candidates competed.

**Procedure:**
1. For each (season, week): among active, find bottom 2 by combined rank and bottom 2 by combined percent.
2. If the sets overlap (same 2 people in bottom 2 under both) but elimination order differs → we have a "disagreement" with full data.
3. If the sets differ, we may still have both in the actual active set—we have data.
4. For each disagreement: who does rank eliminate? who does percent eliminate? Which eliminated person had higher judge score? Higher fan share? 
5. **Fan favor:** Count: when they disagree, rank eliminates the judge-favored one in X cases, percent in Y cases. Ratio or difference indicates which method tilts toward fans.

**Pros:** Targets the policy-relevant cases (method choice matters). No phantoms when we restrict to actual bottom-2.  
**Cons:** May be few such weeks. Doesn't give full placement.

---

### Model D: **Inverse / Fit-to-Data**
**Idea:** Flip the problem. We observe eliminations. Ask: which method (rank or percent) is more *consistent* with the observed elimination order, given judge scores and inferred fan shares?

**Procedure:**
1. For each week: compute who *would* be eliminated under rank and under percent (using actual active set).
2. Compare to who *was* actually eliminated.
3. **Consistency rate:** In what fraction of weeks does rank match reality? Percent match reality?
4. If the show used a mix or one method historically, we might see one fit better. Or: the inferred fan shares were tuned to match reality under one regime—so we're circular. Need to be careful.

**Alternative:** Treat observed eliminations as outcome. Fit: P(eliminate i | judge, fan, method). Which method's implied probabilities best predict the observed eliminations? 
**Pros:** Uses observed outcomes; could reveal which method the show (or data) aligns with.  
**Cons:** Regime varies by season (rank vs percent historically). Inference may be circular with Problem 1.

---

### Model E: **Finalists-Only Comparison**
**Idea:** Restrict to contestants who made final 4 (or final 2). We have many weeks of data for all of them. Few or no phantoms.

**Procedure:**
1. For each season, take the 4 (or 2) finalists.
2. For each, compute combined rank and combined percent over *all* weeks they competed (or weeks 1–K for everyone).
3. Order finalists by each method. Compare: same winner? Same order? Kendall τ among finalists.
4. **Fan favor:** For finalists, compute "fan dominance" = how much closer is ordering to fan-only vs judge-only under rank vs percent.

**Pros:** No phantoms in this subset. Focuses on who wins.  
**Cons:** Ignores how they got there. Smaller N. Doesn't capture early-season method effects.

---

### Model F: **Imputation Model + Full Simulation**
**Idea:** Keep the forward simulation, but fix the phantom problem by *imputing* fan shares for phantoms instead of using 0.

**Procedure:**
1. Train an imputation model on observed (season, week, contestant) → fan_share. Features: judge score that week, fan share prior week, judge score prior week, contestant/season fixed effects.
2. For phantom (B, week 2): predict fan_share from B's week 1 data + judge score week 2 (if available) or use prior-week share with decay.
3. Run current forward simulation with imputed values for phantoms.
4. Compare rank vs percent as before.

**Pros:** Preserves full placement comparison. Addresses the core flaw.  
**Cons:** Imputation is speculative. Adds model assumptions. Sensitivity to imputation choice.

---

### Model G: **Path-Conditional Inverse Optimization**
**Idea:** Re-run Problem 1's inverse optimization *under each hypothetical elimination path*. For "rank would have eliminated C in week 1," re-infer fan shares for the path (A, B survive). Then we have valid shares for B in week 2 under that path.

**Procedure:**
1. For each season, for each method (rank, percent): run inverse optimization assuming that method's elimination order.
2. The optimizer infers fan shares consistent with *that* path. No phantoms—shares are defined for everyone active under that path.
3. Compare placement under rank vs percent using their respective inferred shares.
4. Or: run one inverse per method per season; compare the two full trajectories.

**Pros:** Theoretically coherent. No zeros.  
**Cons:** Computationally heavy (new inverse problem per path). May have identification issues (multiple share trajectories can produce same order).

---

### Model H: **Sensitivity / Bounds Without New Data**
**Idea:** Acknowledge we don't know phantom shares. Instead of picking one value, compute *bounds* on the comparison. What's the range of rank vs percent outcomes if phantom shares can be anything in [0, 1]?

**Procedure:**
1. When a phantom appears, treat their share as unknown in [ε, 1] (or [0, 1]).
2. For each extreme (phantom gets min share, phantom gets max share), continue the sim.
3. Report: best-case and worst-case for "rank favors fans" under each scenario. Or: Monte Carlo over phantom shares, report distribution of Kendall τ, same winner, etc.

**Pros:** Honest about uncertainty. No fake imputation.  
**Cons:** Bounds may be wide. Complex to implement.

---

## Summary: Which to Try First?

| Model | Avoids phantoms? | Gives full placement? | Implementation effort | Directly answers "favors fans"? |
|-------|------------------|------------------------|------------------------|--------------------------------|
| A: Week-by-week | ✓ | No | Low | ✓ (when disagree) |
| B: Aggregate re-rank | ✓ | Different notion | Low | Partial |
| C: First-disagreement | ✓ | No | Low | ✓ |
| D: Inverse/fit | ✓ | No | Medium | Indirect |
| E: Finalists only | ✓ | Among finalists | Low | ✓ |
| F: Imputation + sim | No (fixed) | ✓ | Medium | ✓ |
| G: Path-conditional inverse | ✓ | ✓ | High | ✓ |
| H: Bounds | N/A | Bounded | High | With uncertainty |

**Recommended starting points:**
1. **Model A (week-by-week)** — ✅ IMPLEMENTED. `problem2a_model_A.py`. No phantoms, directly comparable. Answers "do the methods make different choices, and when they do, which favors fans?"
2. **Model C (first-disagreement)** — Complements A; focuses on controversial moments.
3. **Model B (aggregate re-rank)** — Quick check; different interpretation but avoids all simulation issues.
4. **Model F (imputation)** — If you want to keep full placement comparison; requires building a prediction model for fan share.
