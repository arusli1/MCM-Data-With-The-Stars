# Summary: Bayesian Latent-Fan-Support Model (concise)

## What the model does
It infers weekly fan vote shares by modeling a latent, time-evolving popularity score that responds to judges’ performance and contestant covariates, maps to vote shares via softmax, and generates eliminations through regime-specific likelihoods (percent, rank, bottom-two + judge save). It outputs MAP shares plus uncertainty and diagnostic fit metrics.

## Inputs (data used)
From the provided CSV:
- `season`, `celebrity_name`, `ballroom_partner`, `celebrity_age_during_season`,
  `celebrity_industry`, `celebrity_home*`, `placement`, `results`
- Week-by-week judge scores: `weekX_judgeY_score`
- Season-level popularity proxy `pop_{si}` from Google Trends:
  a season-window time series is pulled once per season and averaged for
  each contestant, normalized against a fixed anchor term. This is a strong
  driver of baseline support and avoids sparse weekly zeros.
- Derived: weekly active set, eliminated set, judge percent, standardized judge z-score

Optional / external (used only if you choose):
- Episode viewership for totals scaling
- External popularity priors (social followers)

Not used: actual vote counts, judge identities beyond scores.

## Outputs
Per season-week-contestant (active only):
- MAP vote share
- Uncertainty: posterior sd and 10–90% (or 2.5–97.5%) interval
- Optional scaled totals if you provide weekly turnout
- Diagnostics: elimination hit rate, bottom-two inclusion rate, elimination margin

## Model (compact math)
For each season s, week w, contestant i:

1) State dynamics
```
p_swi = rho * p_s,w-1,i
       + (1-rho) * (alpha_s + X_i beta + beta_pop * pop_i
                    + gamma * Jz_swi + u_pro(i) + eta_w)
       + eps_swi,   eps_swi ~ N(0, sigma_p^2)
```

2) Shares (softmax)
```
s_swi = exp(p_swi / tau) / sum_{j in A_sw} exp(p_sw,j / tau)
```
Use weekly centering of p to fix gauge freedom.

3) Combined risk
- Percent seasons: R_swi = judge_pct_swi + s_swi
- Rank seasons: R_swi = rJ_swi + rF_swi (rF from soft rank of s)
- Bottom-two seasons: use rank proxy to define bottom-two, then judge-choice logit

4) Likelihood
Sequential Plackett–Luce on logits l_i = R_i / kappa for the observed eliminated set
(multi-elim uses random permutations with log-mean-exp). Add a hard ordering
penalty so eliminated contestants have worse combined scores than survivors.
For judge-save weeks: marginalize judge choice over possible bottom-two pairs.

## Priors / regularization (defaults)
- rho ~ Beta(4,2)
- tau ~ LogNormal(0, 0.25^2), clamped to [0.7, 2]
- beta, gamma, alpha_s, eta_w, u_pro ~ Normal(0, sd^2) with weak sd
- sigma_p ~ HalfNormal
- Entropy penalty on shares and L2 penalty on centered p to avoid over-peaking

## Edge cases
- k=0 weeks: omit likelihood
- k>1 weeks: bottom-k Plackett–Luce
- Carryover weeks: treat as no-elim if unknown
- Regime uncertainty (S28+): run both assumptions and report sensitivity

## Inference plan
1) MAP fit with Adam (optionally L-BFGS refinement)
2) Uncertainty:
   - Laplace (diag Hessian), or
   - SGLD / bootstrap re-fits
3) Diagnostics: posterior predictive eliminations vs observed

## Practical implementation notes
- Mask inactive contestants; use stable softmax; center p weekly to fix gauge
- Keep tau > 0.6 to avoid extreme peaks
- Use logistic rank proxy with tau_r around 20–50
- Output CSV fields:
  `season,week,celebrity,s_map,s_mean,s_sd,s_p10,s_p90,notes`

## Defensibility and limits
- Explicitly encodes show rules and performance-to-popularity link
- Shares are identifiable; totals are not without external anchors
- Production interventions and carryovers are sources of uncertainty
