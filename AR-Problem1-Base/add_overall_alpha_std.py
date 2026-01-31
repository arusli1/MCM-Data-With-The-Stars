#!/usr/bin/env python3
"""Add overall_alpha_std to base_overall_metrics.csv using existing base_metrics and placement/shares for weights. No full rerun."""
import os
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = os.path.join(REPO, "AR-Problem1-Base")
RESULTS = os.path.join(BASE, "base_results")
FINAL = os.path.join(BASE, "final_results")

def main():
    # Use final_results if present, else base_results
    metrics_path = os.path.join(FINAL, "base_metrics.csv")
    overall_path = os.path.join(FINAL, "base_overall_metrics.csv")
    placement_path = os.path.join(FINAL, "base_placement_orders.csv")
    if not os.path.isfile(metrics_path):
        metrics_path = os.path.join(RESULTS, "base_metrics.csv")
        overall_path = os.path.join(RESULTS, "base_overall_metrics.csv")
        placement_path = os.path.join(RESULTS, "base_placement_orders.csv")

    metrics = pd.read_csv(metrics_path)
    overall = pd.read_csv(overall_path)

    # Weights = number of contestants per season (same as base.py)
    if os.path.isfile(placement_path):
        placement = pd.read_csv(placement_path)
        weights = placement.groupby("season").size().reindex(metrics["season"]).fillna(1).values
    else:
        weights = np.ones(len(metrics))

    weights = weights.astype(float)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(metrics)) / len(metrics)

    alphas = metrics["alpha"].values
    alpha_mean = float(np.average(alphas, weights=weights))
    alpha_var = float(np.average((alphas - alpha_mean) ** 2, weights=weights))
    alpha_std = float(np.sqrt(alpha_var))

    if "overall_alpha_std" not in overall.columns:
        overall["overall_alpha_std"] = alpha_std
    else:
        overall["overall_alpha_std"] = alpha_std
    # Keep mean in sync
    overall["overall_alpha_mean"] = alpha_mean

    overall.to_csv(overall_path, index=False)
    print(f"Updated {overall_path} with overall_alpha_mean={alpha_mean:.6f} overall_alpha_std={alpha_std:.6f}")

    # Also update the other copy if different
    other = FINAL if overall_path.startswith(RESULTS) else RESULTS
    other_path = os.path.join(other, "base_overall_metrics.csv")
    if other_path != overall_path and os.path.isfile(other_path):
        overall.to_csv(other_path, index=False)
        print(f"Updated {other_path}")

if __name__ == "__main__":
    main()
