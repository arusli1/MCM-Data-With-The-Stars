# Sensitivity Analysis: Problem 2b

## Threshold Sensitivity

Controversy classification sensitivity to threshold choice. Main analysis uses GMM (data-driven); this folder reports results at fixed thresholds for robustness.

**Run:**
```bash
python3 run_threshold_sensitivity.py
```

**Outputs:**
- `outputs/threshold_sensitivity.csv` — N controversial, fan/judge favored, known examples at each threshold.
- `outputs/threshold_sensitivity_summary.md` — Summary table and conclusion.
- `figures/threshold_sensitivity.pdf` — N controversial vs threshold.

**Conclusion:** Main findings (method choice minimal impact, judge-save affects few) robust across thresholds 0.30–0.40.
