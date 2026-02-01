# Problem 2c: Recommendations for Future Seasons

## Question
Based on your analysis, which method (rank vs percent) would you recommend for future seasons, and why? Would you suggest including the judge-save (judges choosing from the bottom two)?

---

## Framing: What Does "Better" Mean?

We evaluate methods along three criteria, motivated by theory and prior work:

| Criterion | Definition | Theory/Lit |
|-----------|------------|------------|
| **Procedural legitimacy** | Audiences perceive the outcome as fair; their votes matter | BBC/regulatory standards: participation TV must be honest, fair; viewers expect influence (e.g., "Dancing with the audience" on vote-in administration) |
| **Engagement/suspense** | Uncertainty and tension keep viewers watching | Suspense and surprise positively affect live TV audience figures; perceived uncertainty reduces probability of leaving a stream; later-stage uncertainty matters most (Wimbledon, sports demand literature) |
| **Narrative tension** | Judge–fan disagreement creates drama (but not alienation) | Underdog narratives + rooting interest → suspense; some controversy = engagement; excessive perceived "rigged" outcomes → trust erosion (2007 UK vote-in scandal) |

**Key tension:** More fan influence → higher legitimacy, but can reduce judge–fan "battle" narrative. Judge-save adds drama (judges vs. fans) but reduces perceived fairness. We seek a *balance*.

---

## Recommendation 1: Use Rank (Favors Fans More)

**Conclusion:** Recommend **Rank** for combining judge and fan inputs.

**Evidence from our analysis (Forward simulation; Data/estimate_votes.csv):**
- Rank favors fans more than percent (rank favor magnitude 0.81 displacement units; 29/34 seasons).
- Percent is heavily judge-dominated (fan advantage −0.97; 0/34 seasons fans dominate).
- Same winner in 88% of seasons under both; same top 4 in 71%.

**Rationale:**
- **Legitimacy:** Rank gives fans more influence than percent. Percent's structure lets judge scores dominate; rank normalizes inputs. Procedural fairness favors Rank.
- **Engagement:** Rank creates more swing potential; fan favorites can overcome bad judge weeks.
- **Controversy balance:** Our 21 controversial cases are mostly fan-favored (Bobby Bones, Bristol Palin). Rank allows fan influence to override judges; percent would reduce perceived fairness.

---

## Recommendation 2: Judge-Save vs. Fan-Decide in Bottom-Two

**Setup:** In weeks with a single elimination (k=1), the **bottom two** by combined score face elimination. One goes home. The rule for *who picks*:
- **Judge-save:** Eliminate the one with **lower judge score** (judges pick).
- **Fan-decide:** Eliminate the one with **lower fan share** (fans pick).

This applies in **every** bottom-two week, not only when scores are tied.

**Conclusion:** We recommend **fan-decide** for fairness; **judge-save** if maximizing drama is the priority.

**Evidence:**
- Judge-save decreases fan advantage in 25/34 seasons (Rank). Under Percent, judge-save has no effect (Δ = 0).

**Rationale:**
- **If legitimacy > drama:** Use **fan-decide**. In every bottom-two week, the contestant with lower fan share goes home—fans effectively choose. Aligns with "your vote matters."
- **If drama > legitimacy:** Use **judge-save**. Creates "judges vs. fans" tension—"judges saved X, fans wanted Y." Narrative payoff.
- **Potential issue:** Fan-decide every week could amplify extreme fan favorites (e.g., Bobby Bones) and frustrate judges. Judge-save balances that but may feel unfair to voters. The trade-off is real.

---

## Creative Proposals: Novel Rules to Stand Out

Below are **theory-motivated** rule changes that could differentiate the show and are justified by our criteria.

### 1. **Tension Arc: Alternating Regime by Phase**
- **Rule:** Early weeks (e.g., weeks 1–5): use **Percent** (judge-heavy). Late weeks (e.g., top 6 onward): use **Rank** (fan-heavy).
- **Rationale:** Early on, judges filter clearly weak dancers. Late in the season, viewers are invested—giving fans more influence increases engagement and legitimacy when stakes are highest. Literature: later-stage uncertainty matters most for viewership.
- **Stand-out:** Explicitly designed "arc" from judge curation to fan empowerment.

### 2. **Redemption Bottom-Two: Double-Chance Elimination**
- **Rule:** Bottom two don’t both leave. One is eliminated; the other goes to "redemption" status. Next week, if they’re in the bottom two again, they’re eliminated regardless. If not, they clear redemption.
- **Rationale:** Increases suspense—"can they claw back?" Underdog narrative. One bad week isn’t fatal. More episodes of tension.
- **Stand-out:** Gives contestants a second chance without removing stakes.

### 3. **Judge Blind Week: One Episode of Pure Fan Voice**
- **Rule:** One designated week per season, judges’ scores are **sealed until after the fan vote closes**. Combined score uses judge + fan, but fans vote without knowing judge rankings.
- **Rationale:** Reduces strategic/bandwagon voting; captures "pure" fan preference. Creates novelty and a one-time "fan power" moment. Transparency about the rule builds legitimacy.
- **Stand-out:** Simple, one-off experiment; easy to explain and market.

### 4. **Suspense Multiplier: Momentum Bonus**
- **Rule:** Contestants who **improve** week-over-week (by judge scores) get a small fan-vote multiplier (e.g., 1.1×) for that week.
- **Rationale:** Rewards growth; creates "improvement" narrative. Aligns judges (they like improvement) with fans (viewers love comeback stories). Increases outcome uncertainty—late improvers can surge.
- **Stand-out:** Incentivizes both artistic growth and fan engagement.

### 5. **Controversy Cap: One Judge Override per Season**
- **Rule:** If a contestant has been in the bottom two by judges **3+ times** but keeps surviving (fan saves), judges get **one** "override" per season: they can send that contestant home despite fan support.
- **Rationale:** Addresses extreme cases (e.g., "Bobby Bones" scenario) where judges feel the outcome is absurd. One override limits abuse. Creates clear "judges vs. fans" moment with a cap.
- **Stand-out:** Balances fan power with judge credibility; prevents perceived "joke" outcomes.

---

## Do We Need More Lit Review, Data Analysis, or Plots?

**What we have:** Our recommendations rest on (1) empirical results from 2a/2b (fan advantage, controversy, regime effects), (2) theory from regulatory/participation-TV and sports-engagement literature, and (3) motivated reasoning (fairness vs. drama trade-offs).

**What would strengthen 2c (optional):**
- **Lit review:** A short paragraph citing 2–3 studies on reality-TV voting, procedural fairness, or suspense/viewership would add academic weight. We reference concepts; formal citations would deepen credibility.
- **Data/plots:** Viewership data was fetched from Wikipedia (see `Data/dwts_viewership.csv`). **Controversy–viewership analysis done:** we use a continuous metric (mean |judge % − placement %| per contestant per season). Raw correlation with viewership is positive (r ≈ 0.33) but confounded by time. **Partial correlation (controlling season) is negative** (r ≈ −0.54, *p* ≈ 0.002): within-era, higher controversy tends to associate with *lower* viewership. **Paper figures:** `paper_viewership_controversy.pdf` (2-panel: time series + residual scatter), `paper_viewership_controversy_scatter.pdf`.
- **Simulation of proposed rules:** We could implement "Tension Arc" or "Redemption B2" and show simulated outcomes. This would make proposals more concrete and data-driven.

**Bottom line:** The current 2c is defensible without additional analysis—recommendations are grounded in our existing results and theory. Additional work would strengthen but is not strictly required.

---

## Optional: Additional Analysis to Strengthen 2c

If time permits, these would add empirical support:

1. **Suspense index by season:** Correlate rank–percent displacement with a proxy for "tension" (e.g., number of contestants whose placement differs by ≥3 under the two methods). Seasons with high displacement = more potential narrative swing. Plot: "Suspense potential by season."
2. **Controversy vs. volatility:** For controversial contestants, compute variance in placement across the 4 regimes. High volatility = more "what if" tension. Could support "controversy creates engagement" claim.
3. **Viewership analysis (done):** Fetched Wikipedia ratings; correlated with mean controversy (|judge % − placement %|). Raw r ≈ 0.33 (confounded by time); partial r controlling season ≈ −0.54 (*p* ≈ 0.002). See "Viewership–Controversy: Data Limitations" below.
---

## Viewership–Controversy: Data Limitations

We cannot robustly test whether controversy drives engagement (e.g., viewership) with our data. Issues:

| Issue | Explanation |
|-------|-------------|
| **Confounding** | Viewership declines sharply over time (cord-cutting). Raw correlation is confounded: early seasons had both high viewership and different controversy levels. Partial correlation (controlling season) suggests the opposite of the raw trend. |
| **Small N** | Only 30 seasons with Wikipedia viewership; 4 missing (12, 18, 19, 31). Low power; sensitivity to outliers. |
| **Data quality** | Wikipedia ratings tables vary by season (format, completeness). No fallback for missing seasons. Season 31 (Disney+) has no comparable Nielsen data. |
| **Metric choice** | We use mean |judge % − placement %| per contestant. Alternative metrics (count above threshold, max in season) give different results. Max controversy shows no correlation. |
| **Causal inference** | Even significant correlations would not imply causation. Competitors, scheduling, and external events drive viewership. |

**Bottom line:** We present the viewership–controversy figures as exploratory. We do not claim controversy predicts viewership; we acknowledge the analysis is limited and emphasize theory over this empirical result for 2c recommendations.

---

## Summary Table

| Decision | Recommendation | Primary justification |
|----------|----------------|----------------------|
| **Combination method** | **Rank** | Fan influence, legitimacy, rank favor magnitude 0.81 |
| **Who picks in bottom-two** (every k=1 week) | **Fan-decide** (fairness) or **Judge-save** (drama) | Trade-off; fan-decide can amplify extreme favorites |
| **Creative rules** | Tension arc, Redemption B2, Judge Blind Week, Momentum bonus, Controversy cap | Theory-motivated; differentiate from generic suggestions |

---

## Suggested Paper Wording (2c)

> We recommend **Rank** over Percent for future seasons. Rank gives fans more influence (rank favor magnitude 0.81 displacement units; 29/34 seasons). Percent is heavily judge-dominated (fan advantage −0.97). Same winner in 88% of seasons—switching would not radically alter most outcomes but would improve legitimacy where it matters.
>
> For the bottom-two rule (in every single-elimination week, who picks which of the bottom two goes home): **fan-decide** favors legitimacy; **judge-save** favors drama. The choice depends on whether the show prioritizes "your vote matters" or "judges vs. fans" tension. Transparency about the rule is essential either way.
>
> We further propose five novel rule designs—a tension arc (Percent early, Rank late), redemption bottom-two, judge blind week, momentum bonus, and controversy cap—each motivated by theory on suspense, engagement, and procedural fairness. These could differentiate the show and be tested in future seasons.
