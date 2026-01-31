# Problem 2: Voting System Comparison

Compare rank vs percent combination methods; analyze controversy cases (judge–fan disagreement). **All paper info (figures, tables, explanations) is in `all-paper-info/`.**

## Scripts

- **`problem2a.py`** — Rank vs percent; fan advantage; rank favor magnitude; Part 3 bottom-2 effect.
- **`problem2b_controversy.py`** — Controversy (GMM classification); 2×2 regimes; judge-save impact.
- **`sensitivity_analysis/run_threshold_sensitivity.py`** — Threshold sensitivity.

## Outputs

- `outputs/` — CSVs and tables (part1, part2, part3, controversy list, classified, 2×2 scenarios, etc.).
- `figures/` — PDFs for paper.
- `sensitivity_analysis/outputs/` and `sensitivity_analysis/figures/` — Threshold sensitivity.

## Usage

```bash
python3 AR-Problem2/problem2a.py
python3 AR-Problem2/problem2b_controversy.py
python3 AR-Problem2/sensitivity_analysis/run_threshold_sensitivity.py
```

## Regime rules

- **Rank (s1–2):** R = judge_rank + fan_rank; eliminate largest R.
- **Percent (s3–27):** C = judge_pct + fan_share; eliminate lowest C.
- **Bottom-two (s28+):** Bottom two by R; judge-save eliminates lower judge score; else lower fan share.
