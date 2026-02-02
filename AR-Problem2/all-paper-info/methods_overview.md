# Problem 2: Methods Overview (Less Black-Box)

Brief explanation of the main methods so the structure is clear.

---

## 2a: Fan Advantage and Regime Comparison

### Part 1 — Rank vs Percent
- **Rank:** Combined = judge_rank + fan_rank. Eliminate contestant(s) with **largest** combined rank each week.
- **Percent:** Combined = judge_pct + fan_share (both sum to 1). Eliminate contestant(s) with **lowest** combined score.
- **Metrics:** Kendall τ (discordant pairs in final order), mean displacement (|placement_rank − placement_percent|), winner same, top 4 same.

### Part 2 — Which Input Dominates?
- Run **3 simulations** per method: (1) just judges, (2) combined (normal rule), (3) just fans.
- **Fan Advantage** = disp(judges→combined) − disp(fans→combined). Positive = combined outcome is closer to "just fans" than to "just judges" (fans dominate).F
- **Disp(A,B)** = mean |placement_A − placement_B| over contestants. Smaller disp = A and B are more similar.
- **Rank favor magnitude** = fan_adv_rank − fan_adv_percent. Positive = rank gives fans more advantage. Forward simulation; phantom survivors use zeros (see simulation_divergence_limitation.md).

### Part 3 — Bottom-2 Judge-Save Effect
- **Bottom-2:** In weeks with k=1 elimination, the bottom two by combined score face elimination. A tie-break picks who goes home.
- **Judge-save:** Tie-break = eliminate the one with **lower judge score** (judges pick).
- **Fan-decide:** Tie-break = eliminate the one with **lower fan share** (fans pick).
- Apply bottom-2 to k=1 weeks for **all seasons** (ignore historical rules). Compare fan advantage: no bottom-2 vs judge-save, for **Rank** and **Percent** separately.
- **Note:** Percent judge-save has no effect (Δ = 0 for all seasons) because in every k=1 week, the contestant with lowest combined (judge_pct + fan_share) also has the lowest judge score among the bottom 2—so no-B2 and judge-save eliminate the same person. Rank judge-save decreases fan advantage in 21/34 seasons (judges gain). Season 18 is an outlier: Rank judge-save increases fan advantage (Δ ≈ +0.83).

---

## 2b: Controversy Classification and 2×2 Regimes

### Controversy Metric
- **controversy_score** = |judge_percentile − placement_percentile|. High = judges and outcome disagreed (judges liked them but went early, or judges disliked them but went far).

### GMM Classification
- **2-component Gaussian mixture** on controversy_score. High-mean component = "controversial."
- **EM algorithm:** E-step (posterior P(high|x)), M-step (update μ, σ, π). Stop when parameters change little (convergence).
- **Constraints:** μ_high ≥ 98th percentile; classify if P(high|x) > 0.8 (GMM-Demux precedent).
- **Fan favored:** placement > judge percentile (above diagonal). **Judge favored:** placement < judge percentile (below diagonal).

### 2×2 Regimes
- For each controversial contestant: 4 regimes = rank vs percent × judge-save vs fan-decide.
- Simulate elimination under each; record placement and elim week. Ask: would method choice or judge-save change their outcome?

---

## Key Notation

| Symbol | Meaning |
|--------|---------|
| disp(A,B) | Mean \|placement_A − placement_B\| |
| Fan Advantage | disp(judges→combined) − disp(fans→combined) |
| judge_percentile | 1 = best with judges, 0 = worst |
| placement_percentile | 1 = winner, 0 = last |
