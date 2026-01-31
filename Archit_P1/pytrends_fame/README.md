# PyTrends-based fame vs outcome (DWTS)

This folder computes a Google Trends–based fame proxy for each contestant-season using `pytrends`, then evaluates how that fame relates to final placement while controlling for judges' scores.

## What it produces
- `outputs/pytrends_fame_merged.csv`: merged dataset with placement, judge summary, and pytrends fame scores
- `outputs/correlations.csv`: correlation summary
- `outputs/plots/*.png`: diagnostic plots
- `pytrends_fame_vs_outcomes_report.md`: paper-style writeup

## How fame is computed (important)
Google Trends values are **relative** within each request. To make values comparable across batches, we use an **anchor term**:
- For each season, we query groups of contestants plus a fixed anchor (default: `"Barack Obama"`).
- For each contestant we compute: `fame_ratio = mean(contestant_interest) / mean(anchor_interest)` over a preseason window.
This stitches batches together (as long as anchor interest is non-zero).

## How to run
From repo root:

```bash
python3 -m pip install pytrends
python3 Archit_P1/pytrends_fame/pytrends_fame_vs_outcomes.py
```

If Google blocks requests, try:
- re-running later,
- adding a longer sleep via `--sleep-seconds`,
- running from a different network,
- or using a proxy (see `pytrends` docs).


