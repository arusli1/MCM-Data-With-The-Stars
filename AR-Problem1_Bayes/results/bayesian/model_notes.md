# Bayesian Model Notes (legacy: bayesian.py)

This folder contains outputs from `AR-Problem1_Bayes/bayesian.py`.

Key characteristics:
- Latent fan support `p_{i,w}` with smoothness and judge drift.
- Shares `s_w = softmax(p_w)`.
- Regime-specific elimination likelihoods (percent, rank, bottom-two).
- Optional finals likelihood to anchor winners.
- SGLD samples for uncertainty.

For full details, see `AR-Problem1_Bayes/bayesian.py` and commit history.
