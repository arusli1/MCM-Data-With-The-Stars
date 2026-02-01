#!/usr/bin/env python3
"""
Problem 2a Model A: Week-by-week comparison (no forward simulation).

Uses only the ACTUAL active set each week. No phantoms, no divergence.
Compares who rank vs percent would eliminate each week. When they disagree,
tracks which method favors fans (eliminates the judge-favored contestant).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from problem2_utils import (
    DATA_PATH,
    build_judge_matrix,
    compute_elim_week_from_judge,
    elim_schedule_from_judge,
    judge_pct,
    load_fan_shares_for_season,
    parse_week_cols,
    rank_order,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def week_by_week_compare(
    J: np.ndarray,
    s_hist: np.ndarray,
    schedule: list,
    elim_week_true: list,
    names: list,
) -> dict:
    """
    For each week with eliminations, compare who rank vs percent would eliminate.
    Uses only the actual active set (no phantoms).

    Returns dict with:
      - agreement_count, disagreement_count, agreement_rate
      - when_disagree_rank_favors_fans_by_judge/fan, when_disagree_percent_favors_fans_by_judge/fan
      - mean_judge_margin: when disagree, mean J(rank_elim) - J(pct_elim); >0 = rank favors fans
      - mean_fan_margin: when disagree, mean S(pct_elim) - S(rank_elim); >0 = rank eliminated lower-fan
      - first_disagree_week: earliest week where they disagree (or None)
      - rows: list of per-week details
    """
    W, N = J.shape
    rows = []
    agreement_count = 0
    disagreement_count = 0
    rank_favors_fans_by_judge = 0
    percent_favors_fans_by_judge = 0
    rank_favors_fans_by_fan = 0
    percent_favors_fans_by_fan = 0
    judge_margins = []
    fan_margins = []
    first_disagree_week = None

    for w in range(W):
        k = schedule[w + 1] if w + 1 < len(schedule) else 0
        if k <= 0:
            continue

        # Active = contestants who performed in week w (reality)
        active = np.array([elim_week_true[i] >= w + 1 for i in range(N)])
        active_idx = np.where(active)[0]

        if len(active_idx) <= k:
            continue

        J_w = J[w]
        s_w = s_hist[w]

        # Rank: worst combined rank (highest rank) eliminated
        rJ = rank_order(-J_w, active)
        rF = rank_order(-s_w, active)
        R = rJ + rF
        order_rank = active_idx[np.argsort(R[active_idx])]
        elim_rank_idx = order_rank[-k:]

        # Percent: lowest combined score eliminated
        jp = judge_pct(J_w, active)
        combined_pct = jp + s_w
        order_pct = active_idx[np.argsort(combined_pct[active_idx])]
        elim_pct_idx = order_pct[:k]

        # Agreement?
        set_rank = set(elim_rank_idx)
        set_pct = set(elim_pct_idx)
        agree = set_rank == set_pct

        if agree:
            agreement_count += 1
            rank_elim = elim_rank_idx[0] if k == 1 else None
            pct_elim = elim_pct_idx[0] if k == 1 else None
            rows.append({
                "week": w + 1,
                "k": k,
                "agree": True,
                "rank_eliminee": names[elim_rank_idx[0]] if k == 1 else ";".join(names[i] for i in elim_rank_idx),
                "pct_eliminee": names[elim_pct_idx[0]] if k == 1 else ";".join(names[i] for i in elim_pct_idx),
                "rank_elim_judge": J_w[elim_rank_idx[0]] if k == 1 else np.mean(J_w[elim_rank_idx]),
                "pct_elim_judge": J_w[elim_pct_idx[0]] if k == 1 else np.mean(J_w[elim_pct_idx]),
                "rank_elim_fan": s_w[elim_rank_idx[0]] if k == 1 else np.mean(s_w[elim_rank_idx]),
                "pct_elim_fan": s_w[elim_pct_idx[0]] if k == 1 else np.mean(s_w[elim_pct_idx]),
                "rank_favors_fans_by_judge": None,
                "rank_favors_fans_by_fan": None,
            })
        else:
            disagreement_count += 1
            if first_disagree_week is None:
                first_disagree_week = w + 1
            # For k=1: which method favors fans?
            # Judge-based: favors fans = eliminated judge-favored (higher J)
            # Fan-based: favors fans = eliminated fan-unfavored (lower S) — more direct
            if k == 1:
                r_elim = elim_rank_idx[0]
                p_elim = elim_pct_idx[0]
                j_r = J_w[r_elim]
                j_p = J_w[p_elim]
                s_r = s_w[r_elim]
                s_p = s_w[p_elim]
                by_judge = j_r > j_p  # rank eliminated higher judge → rank favors fans
                by_fan = s_r < s_p    # rank eliminated lower fan share → rank favored fans (eliminated who fans wanted out)
                if by_judge:
                    rank_favors_fans_by_judge += 1
                elif j_p > j_r:
                    percent_favors_fans_by_judge += 1
                if by_fan:
                    rank_favors_fans_by_fan += 1
                elif s_p < s_r:
                    percent_favors_fans_by_fan += 1
                # New metric: margins when they disagree
                judge_margins.append(float(j_r - j_p))  # >0 = rank eliminated higher judge
                fan_margins.append(float(s_p - s_r))    # >0 = rank eliminated lower fan
                rows.append({
                    "week": w + 1,
                    "k": k,
                    "agree": False,
                    "rank_eliminee": names[r_elim],
                    "pct_eliminee": names[p_elim],
                    "rank_elim_judge": float(j_r),
                    "pct_elim_judge": float(j_p),
                    "rank_elim_fan": float(s_r),
                    "pct_elim_fan": float(s_p),
                    "rank_favors_fans_by_judge": by_judge,
                    "rank_favors_fans_by_fan": by_fan,
                })
            else:
                rows.append({
                    "week": w + 1,
                    "k": k,
                    "agree": False,
                    "rank_eliminee": ";".join(names[i] for i in elim_rank_idx),
                    "pct_eliminee": ";".join(names[i] for i in elim_pct_idx),
                    "rank_elim_judge": float(np.mean(J_w[elim_rank_idx])),
                    "pct_elim_judge": float(np.mean(J_w[elim_pct_idx])),
                    "rank_elim_fan": None,
                    "pct_elim_fan": None,
                    "rank_favors_fans_by_judge": None,
                    "rank_favors_fans_by_fan": None,
                })

    total_weeks = agreement_count + disagreement_count
    agreement_rate = agreement_count / total_weeks if total_weeks > 0 else 0
    mean_judge_margin = np.mean(judge_margins) if judge_margins else np.nan
    mean_fan_margin = np.mean(fan_margins) if fan_margins else np.nan

    return {
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "total_weeks": total_weeks,
        "agreement_rate": agreement_rate,
        "when_disagree_rank_favors_fans_by_judge": rank_favors_fans_by_judge,
        "when_disagree_percent_favors_fans_by_judge": percent_favors_fans_by_judge,
        "when_disagree_rank_favors_fans_by_fan": rank_favors_fans_by_fan,
        "when_disagree_percent_favors_fans_by_fan": percent_favors_fans_by_fan,
        "mean_judge_margin": mean_judge_margin,
        "mean_fan_margin": mean_fan_margin,
        "first_disagree_week": first_disagree_week,
        "rows": rows,
    }


def main():
    print("=" * 60)
    print("Problem 2a Model A: Week-by-Week (No Simulation)")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    weeks = parse_week_cols(df)
    seasons = sorted(df["season"].unique())

    summary_rows = []
    all_detail_rows = []

    for season in seasons:
        df_s = df[df["season"] == season].copy()
        names = df_s["celebrity_name"].tolist()

        J = build_judge_matrix(df_s, weeks)
        W = J.shape[0]
        schedule = elim_schedule_from_judge(J)
        elim_week_true = compute_elim_week_from_judge(J)
        s_hist = load_fan_shares_for_season(season, names, W)

        if s_hist is None:
            print(f"  ⚠ Season {season}: No fan shares, skipping")
            continue

        res = week_by_week_compare(J, s_hist, schedule, elim_week_true, names)

        summary_rows.append({
            "season": season,
            "n_contestants": len(names),
            "total_weeks_with_elim": res["total_weeks"],
            "agreement_count": res["agreement_count"],
            "disagreement_count": res["disagreement_count"],
            "agreement_rate": round(res["agreement_rate"], 4),
            "rank_favors_by_judge": res["when_disagree_rank_favors_fans_by_judge"],
            "pct_favors_by_judge": res["when_disagree_percent_favors_fans_by_judge"],
            "rank_favors_by_fan": res["when_disagree_rank_favors_fans_by_fan"],
            "pct_favors_by_fan": res["when_disagree_percent_favors_fans_by_fan"],
            "mean_judge_margin": res["mean_judge_margin"],
            "mean_fan_margin": res["mean_fan_margin"],
            "first_disagree_week": res["first_disagree_week"],
        })

        for r in res["rows"]:
            all_detail_rows.append({"season": season, **r})

        d = res["disagreement_count"]
        rfj = res["when_disagree_rank_favors_fans_by_judge"]
        pfj = res["when_disagree_percent_favors_fans_by_judge"]
        rff = res["when_disagree_rank_favors_fans_by_fan"]
        pff = res["when_disagree_percent_favors_fans_by_fan"]
        print(f"  ✓ Season {season}: agree={res['agreement_rate']:.2%}, disagree={d} (by judge: R:{rfj} P:{pfj} | by fan: R:{rff} P:{pff})")

    # Save
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUTPUT_DIR / "problem2a_model_A_summary.csv", index=False)
    print(f"\n✓ Saved: {OUTPUT_DIR / 'problem2a_model_A_summary.csv'}")

    df_detail = pd.DataFrame(all_detail_rows)
    df_detail.to_csv(OUTPUT_DIR / "problem2a_model_A_week_detail.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'problem2a_model_A_week_detail.csv'}")

    # Overall fan-favor margin (from detail: k=1 disagreements only)
    dis_rows = [r for r in all_detail_rows if not r.get("agree", True) and r.get("k") == 1]
    judge_margins_all = [
        r["rank_elim_judge"] - r["pct_elim_judge"]
        for r in dis_rows
        if r.get("rank_elim_judge") is not None and r.get("pct_elim_judge") is not None
    ]
    overall_judge_margin = np.mean(judge_margins_all) if judge_margins_all else np.nan

    # Overall summary
    total_agree = sum(r["agreement_count"] for r in summary_rows)
    total_disagree = sum(r["disagreement_count"] for r in summary_rows)
    total_rfj = sum(r["rank_favors_by_judge"] for r in summary_rows)
    total_pfj = sum(r["pct_favors_by_judge"] for r in summary_rows)
    total_rff = sum(r["rank_favors_by_fan"] for r in summary_rows)
    total_pff = sum(r["pct_favors_by_fan"] for r in summary_rows)
    total_weeks = total_agree + total_disagree

    print("\n" + "=" * 60)
    print("Model A Overall Summary")
    print("=" * 60)
    print(f"Total elimination weeks analyzed: {total_weeks}")
    print(f"Agreement (same person eliminated): {total_agree} ({100*total_agree/total_weeks:.1f}%)")
    print(f"Disagreement: {total_disagree}")
    print(f"\nWhen they disagree — BY JUDGE (eliminated higher J = favors fans):")
    print(f"  Rank favors fans: {total_rfj} | Percent favors fans: {total_pfj} | Tie: {total_disagree - total_rfj - total_pfj}")
    if total_disagree > 0:
        print(f"  → Rank: {100*total_rfj/total_disagree:.1f}% | Percent: {100*total_pfj/total_disagree:.1f}%")
    print(f"\nWhen they disagree — BY FAN SHARE (eliminated lower S = favors fans):")
    print(f"  Rank favors fans: {total_rff} | Percent favors fans: {total_pff} | Tie: {total_disagree - total_rff - total_pff}")
    if total_disagree > 0:
        print(f"  → Rank: {100*total_rff/total_disagree:.1f}% | Percent: {100*total_pff/total_disagree:.1f}%")
        if total_rff > total_pff:
            print(f"\n  CONCLUSION: Rank favors fan votes more than percent (by fan-share definition).")
        elif total_pff > total_rff:
            print(f"\n  CONCLUSION: Percent favors fan votes more than rank (by fan-share definition).")
        else:
            print(f"\n  CONCLUSION: Tie by fan share.")
    if not np.isnan(overall_judge_margin):
        print(f"\nNEW METRIC — Fan-favor margin (when disagree, mean J(rank_elim) - J(pct_elim)): {overall_judge_margin:.3f}")
        print(f"  (positive = rank eliminates higher-judge person on average → rank favors fans)")


if __name__ == "__main__":
    main()
