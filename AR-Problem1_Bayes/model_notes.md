## Bayesian Model Architecture (Fan Vote Shares)

**Goal.** Estimate weekly fan vote shares `s_{i,w}` that explain eliminations while quantifying uncertainty. The model is intentionally minimal: only a latent dynamics prior and outcome likelihoods tied to the show rules.

### Data & preprocessing
- Source: `Data/2026_MCM_Problem_C_Data.csv`.
- Weekly judge total `J_{i,w}` = sum of `weekX_judgeY_score` (ignoring `N/A`).
- Active set `A_w`: contestants with `J_{i,w} > 0` in week `w`.
- Elimination week: parsed from `results` (withdrawals use last active week).

### Latent state
- Latent fan support `p_{i,w}` evolves weekly with a smoothness prior.
- Fan vote shares `s_w = softmax(p_w)` on active contestants (masked softmax).

### Priors / regularization
1) **Latent dynamics prior (core).**  
Smoothness with a small judge‑performance drift:
```
p_{i,w} = p_{i,w-1} + γ·J^z_{i,w} + ε_{i,w},  ε_{i,w} ~ N(0, σ^2)
```
We penalize deviations from this dynamics (Gaussian prior on ε).

2) **Logit scale control.**  
L2 penalty on `p` to prevent extreme softmax concentration.

### Likelihood (regime‑specific)
We use a soft elimination likelihood plus a hard‑constraint penalty.  
Define:
```
q_{i,w} = judge percentile (or rank percentile)
C_{i,w} = a·z(q_{i,w}) − b·z(log s_{i,w})
P(elim = i | w) ∝ exp(λ · C_{i,w})
```
- `z(·)` denotes within‑week standardization so judge and vote terms are on comparable scales.
- **Percent seasons:** `q_{i,w}` = judge percent.  
- **Rank/Bottom seasons:** `q_{i,w}` = judge rank percentile.  

For finals, we add a **Plackett–Luce** likelihood over a season‑strength score:
```
S_i = α Σ_w (1 − q_{i,w}) + β Σ_w log(s_{i,w})
```
and apply Plackett–Luce to the final placement order.

### Fan‑power indicators (explicit linkage)
- **Indicator 2 (weekly survival despite weak judges):**  
  This is the core signal in the weekly likelihood via `C_{i,w}`. If a contestant is
  consistently low in `q_{i,w}` but not eliminated, the model must increase `s_{i,w}`
  (thus `p_{i,w}`) to make the observed elimination likely.
- **Indicator 1 (season‑level discrepancy):**  
  Captured as a *validation/constraint* through the optional finals likelihood on `S_i`.
  We treat this as a weaker, season‑level check to avoid double‑counting weekly evidence.

Soft likelihood: `P(elim | C) ∝ exp(C / T)` (higher `C` = worse).  
`a` and `b` are learned (positive via softplus); `γ` is learned (bounded via tanh).
Hard penalty: add hinge loss if any eliminated has *lower* `C` than a survivor.

### Objective (MAP)
Minimize:
- `λ_dyn · ‖p_w − (p_{w-1} + γ·J^z_w)‖²`
- `λ_p · ‖p_w‖²` (logit scale control)
- `−λ_ent · H(s_w)` (entropy regularization)
- `−log likelihood` (soft)
- `λ_hard · hinge` (elimination ordering)

### Optimization
- MAP via Adam on `p_{w}` and scalar weights `a,b,γ,α,β`.
- SGLD samples enabled for uncertainty.

### Outputs
- `AR-Problem1_Bayes/inferred_shares_bayes.csv` (MAP `s_map`)
- `AR-Problem1_Bayes/inferred_shares_bayes_unc.csv` (`s_mean`, `s_p50`, `s_std`, `s_p10`, `s_p90`)
- `AR-Problem1_Bayes/elimination_match_bayes.csv` (per‑season match rates)

### Hyperparameters (default)
`TEMP = 0.7`, `TEMP_PLACEMENT = 0.6`, `λ_dyn = 0.5`, `λ_p = 0.1`,  
`λ_hard = 150`, `λ_final = 10`, `λ_ent = 0.05`, `LR = 0.05`, `N_STEPS = 250`  
SGLD: `SGLD_STEPS = 80`, `BURNIN = 20`, `INTERVAL = 5`

### Assumptions & limitations
- Rank/bottom regimes remain under‑identified; the hard penalty enforces eliminations.
- Soft likelihood can trade off fit vs prior; `λ_hard` controls strictness.
