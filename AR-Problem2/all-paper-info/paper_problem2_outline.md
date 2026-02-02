# Problem 2: Voting System Comparison — Paper Outline

## 4.1 Rank vs Percent: Combination Methods and Outcome Comparison

- **Inputs:** Judge scores (weekly totals) and fan-share trajectories from Problem 1; elimination schedule (k per week) from data.
- **Outputs:** Predicted placement order under **rank** (judge_rank + fan_rank) and under **percent** (judge_pct + fan_share) for each season.
- **Metric:** Kendall τ (discordant pairs in final order), mean displacement (|placement_rank − placement_percent|), same winner?, same top 4?
- **Baseline:** Forward simulation under both methods; treat percent as the “judge-heavy” reference and rank as the alternative.
- **Assumptions:** Phantom survivors (contestants who survive in one path but not the other) receive zero fan share; same schedule for both regimes.

---

## 4.2 Fan Advantage and Judge-Save Effect

- **Improved framing:** Define **fan advantage** = disp(judges→combined) − disp(fans→combined). Positive ⇒ combined outcome is closer to “just fans” than “just judges.”
- **Rank favor magnitude:** fan_adv_rank − fan_adv_percent; positive ⇒ rank favors fans more than percent.
- **Why it matters:** Quantifies which method gives fans more influence; percent is judge-dominated (fan advantage ≈ −0.97), rank gives fans more swing (rank favor magnitude ≈ 0.81).
- **Judge-save (bottom-two):** In k=1 weeks, bottom two by combined score; **judge-save** = eliminate lower judge score; **fan-decide** = eliminate lower fan share. Compare fan advantage with vs without judge-save.
- **Key result:** Judge-save decreases fan advantage under Rank in most seasons; no effect under Percent. Feeds recommendation (4.3).

---

## 4.3 Controversy, Regime Comparison, and Recommendations

- **Controversy metric:** |judge_percentile − placement_percentile|; high ⇒ judges and outcome disagreed (fan favored or judge favored).
- **Classification:** 2-component GMM on controversy_score; constrained so high component lies in extreme tail; classify as controversial if P(high | x) > threshold (e.g. 0.8). Yields a set of controversial contestants.
- **2×2 regimes:** For each controversial contestant, simulate four regimes: rank vs percent × judge-save vs fan-decide; record placement and elim week under each.
- **Derivations (if needed):** GMM E-step/M-step or cutoff interpretation only if it aids understanding; otherwise state constraints and threshold.
- **Key outputs for later sections:** (1) **Recommendation:** Rank over percent for fan influence; fan-decide vs judge-save depending on fairness vs drama. (2) **Controversy set and regime table:** Which contestants are controversial and how method choice / judge-save would have changed their outcome (feeds discussion and implications).
