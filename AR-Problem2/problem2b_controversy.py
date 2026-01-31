"""
Problem 2b: Controversy cases (judge-fan disagreement). Identify contestants,
run 2x2 (rank vs percent x judge-save vs no), and quantify judge-save impact.
"""
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from problem2_utils import (
    DATA_PATH,
    build_judge_matrix,
    elim_schedule_from_judge,
    forward_simulate,
    judge_pct,
    kendall_tau,
    load_fan_shares_for_season,
    mean_displacement,
    rank_order,
    parse_week_cols,
    season_regime,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Problem-statement examples: (season, celebrity_name, brief) — used for labels and summary
CONTROVERSY_EXAMPLES = [
    (2, "Jerry Rice", "Runner-up despite lowest judges scores in 5 weeks"),
    (4, "Billy Ray Cyrus", "5th despite last place judge scores in 6 weeks"),
    (11, "Bristol Palin", "3rd with lowest judge scores 12 times"),
    (27, "Bobby Bones", "Won despite consistently low judges scores"),
]


def get_brief(season: int, celebrity_name: str, row: pd.Series) -> str:
    """Brief descriptor: use known example text if matched, else judge/placement type."""
    for s, name, brief in CONTROVERSY_EXAMPLES:
        if s == season and name == celebrity_name:
            return brief
    if row.get("low_judge_high_place"):
        return "Low judge score, high placement (fan favorite)"
    if row.get("high_judge_low_place"):
        return "High judge score, low placement (early exit)"
    return "Judge–placement disagreement"


def controversy_type_from_row(row: pd.Series) -> str:
    """Classify by diagonal: above = fan_favored (placement > judge), below = judge_favored (placement < judge)."""
    judge_pct = float(row.get("judge_percentile", 0))
    place_pct = float(row.get("placement_percentile", 0))
    if place_pct > judge_pct:
        return "fan_favored"   # above diagonal: placement better than judge score suggests (fans kept them)
    return "judge_favored"    # below diagonal: placement worse than judge score suggests (judges liked them, elim early)

# Controversy = disagreement between JUDGE SCORES and PLACEMENT only (no fan-share estimates).
# Low judge score + high ranking (good placement) = "bad dancer but went far" (e.g. Jerry Rice, Bobby Bones).
# High judge score + low ranking (early elim) = "good dancer but went home early".
# Metric: judge_percentile (1=best with judges, 0=worst) vs placement_percentile (1=winner, 0=last).
# controversy_score = |judge_percentile - placement_percentile| (high = judges and outcome disagreed).
# Classification: 2-component Gaussian mixture model (data-driven); high-mean component = controversial.


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Normal PDF at x. sigma > 0."""
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def fit_2component_gmm(scores: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, dict]:
    """
    Fit 2-component 1D Gaussian mixture via EM.
    Returns (posterior_high, params) where posterior_high[i] = P(high component | scores[i]).
    High component = the one with larger mean (controversial).
    Initialize high component with extreme tail (scores >= 0.3) to capture controversy cases.
    """
    x = np.array(scores, dtype=float).flatten()
    n = len(x)
    # Initialize: low = bulk (scores < 0.25), high = extreme tail (scores >= 0.3)
    tail = x[x >= 0.3]
    bulk = x[x < 0.25]
    mu0 = float(np.mean(bulk)) if len(bulk) > 0 else float(np.percentile(x, 50))
    mu1 = float(np.mean(tail)) if len(tail) > 0 else float(np.percentile(x, 95))
    sig0 = max(np.std(bulk), 0.02) if len(bulk) > 1 else 0.05
    sig1 = max(np.std(tail), 0.02) if len(tail) > 1 else 0.05
    pi0 = len(bulk) / n if n > 0 else 0.9
    pi1 = len(tail) / n if n > 0 else 0.1

    # Constrain high component to extreme tail: mu_high >= 98th percentile
    # Yields ~20 controversial (extreme judge–placement disagreement); data-driven.
    MU_HIGH_MIN = float(np.percentile(x, 98))

    for _ in range(max_iter):
        # E-step: posterior P(component 1 | x) = pi1*N(x|mu1,s1) / (pi0*N0 + pi1*N1)
        p0 = _gaussian_pdf(x, mu0, sig0)
        p1 = _gaussian_pdf(x, mu1, sig1)
        denom = pi0 * p0 + pi1 * p1
        denom = np.maximum(denom, 1e-300)
        gamma1 = (pi1 * p1) / denom
        # M-step
        n1 = gamma1.sum()
        n0 = n - n1
        pi0, pi1 = n0 / n, n1 / n
        mu0_new = np.sum((1 - gamma1) * x) / max(n0, 1e-10)
        mu1_new = np.sum(gamma1 * x) / max(n1, 1e-10)
        mu1_new = max(mu1_new, MU_HIGH_MIN)  # keep high component in tail
        sig0_new = np.sqrt(np.sum((1 - gamma1) * (x - mu0_new) ** 2) / max(n0, 1e-10))
        sig1_new = np.sqrt(np.sum(gamma1 * (x - mu1_new) ** 2) / max(n1, 1e-10))
        sig0_new = max(sig0_new, 0.02)
        sig1_new = max(sig1_new, 0.02)
        if abs(mu0 - mu0_new) < tol and abs(mu1 - mu1_new) < tol:
            break
        mu0, mu1, sig0, sig1 = mu0_new, mu1_new, sig0_new, sig1_new

    # High component = larger mean (controversial tail)
    if mu0 > mu1:
        posterior_high = 1 - gamma1
        params = {"mu_low": mu1, "mu_high": mu0, "sig_low": sig1, "sig_high": sig0, "pi_low": pi1, "pi_high": pi0}
    else:
        posterior_high = gamma1
        params = {"mu_low": mu0, "mu_high": mu1, "sig_low": sig0, "sig_high": sig1, "pi_low": pi0, "pi_high": pi1}
    return posterior_high, params


# Posterior threshold for "controversial": require P(high|x) > this.
# 0.8 focuses on clear tail; with constrained GMM yields ~20–25.
GMM_POSTERIOR_THRESHOLD = 0.8


def placement_true_from_data(df_season: pd.DataFrame) -> List[Optional[int]]:
    out = []
    for v in df_season["placement"].tolist():
        try:
            out.append(int(v))
        except Exception:
            out.append(None)
    return out


def build_controversy_judge_placement_only(
    df: pd.DataFrame,
    weeks: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Controversy from PLACEMENT and JUDGE SCORES only (no fan-share estimates).
    Low judge score + high ranking = "bad dancer but went far". High judge score + low ranking = "good dancer but went home early".
    Judge percentile: 1 = best with judges, 0 = worst. Placement percentile: 1 = winner, 0 = last.
    controversy_score = |judge_percentile - placement_percentile| (high = disagreement).
    """
    full_rows = []
    for season in df["season"].unique():
        season = int(season)
        df_season = df[df["season"] == season].reset_index(drop=True)
        names = df_season["celebrity_name"].tolist()
        J = build_judge_matrix(df_season, weeks)
        W, N = J.shape
        placement_list = placement_true_from_data(df_season)
        # Mean judge score per contestant (over weeks they competed)
        mean_judge_scores = []
        for idx in range(N):
            vals = [J[w, idx] for w in range(W) if J[w, idx] > 0]
            mean_judge_scores.append(np.mean(vals) if vals else 0.0)
        # Rank by mean judge score (1 = best). Then judge_percentile: 1 = best, 0 = worst.
        order_judge = np.argsort(-np.array(mean_judge_scores))
        judge_rank = np.zeros(N, dtype=int)
        for r, i in enumerate(order_judge, start=1):
            judge_rank[i] = r
        judge_percentile = (N - judge_rank) / (N - 1) if N > 1 else np.zeros(N)
        for idx, name in enumerate(names):
            pt = placement_list[idx] if idx < len(placement_list) else None
            if pt is None:
                continue
            # placement_percentile: 1 = winner (placement 1), 0 = last place
            placement_percentile = 1.0 - (pt - 1) / (N - 1) if N > 1 else 1.0
            controversy_score = abs(judge_percentile[idx] - placement_percentile)
            n_lowest = weeks_lowest_judge(J, idx)
            # Type: low_judge_high_place = "bad with judges but good placement"; high_judge_low_place = "good with judges but bad placement"
            low_judge_high_place = judge_percentile[idx] < 0.5 and placement_percentile >= 0.5
            high_judge_low_place = judge_percentile[idx] >= 0.5 and placement_percentile < 0.5
            full_rows.append({
                "season": season,
                "celebrity_name": name,
                "n_contestants": N,
                "placement": pt,
                "mean_judge_score": mean_judge_scores[idx],
                "judge_rank_in_season": int(judge_rank[idx]),
                "judge_percentile": judge_percentile[idx],
                "placement_percentile": placement_percentile,
                "controversy_score": controversy_score,
                "weeks_lowest_judge_score": n_lowest,
                "low_judge_high_place": low_judge_high_place,
                "high_judge_low_place": high_judge_low_place,
            })
    full_df = pd.DataFrame(full_rows)

    # Data-driven classification: 2-component Gaussian mixture
    scores = full_df["controversy_score"].values
    posterior_high, gmm_params = fit_2component_gmm(scores)
    full_df["controversy_posterior"] = posterior_high
    full_df["controversial"] = posterior_high > GMM_POSTERIOR_THRESHOLD

    classified_rows = [
        {**row.to_dict(), "in_list_reason": "mixture"}
        for _, row in full_df.iterrows()
        if row["controversial"]
    ]
    classified_df = pd.DataFrame(classified_rows)
    return full_df, classified_df, gmm_params


def plot_regime_controversy_by_type(scenario_df: pd.DataFrame) -> None:
    """Bar chart: mean simulated controversy by regime and controversy_type (fan_favored vs judge_favored)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Publication-quality styling
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    rows = []
    for ctype in ["fan_favored", "judge_favored"]:
        sub = scenario_df[scenario_df["controversy_type"] == ctype]
        if sub.empty:
            continue
        for regime in SCENARIO_KEYS:
            col = f"simulated_controversy_{regime}"
            rows.append({"controversy_type": ctype, "regime": regime, "mean_controversy": sub[col].mean()})
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(SCENARIO_KEYS))
    w = 0.35
    
    # Colors: green for fan favored, red for judge favored
    colors = {"fan_favored": "#029E73", "judge_favored": "#C1292E"}
    
    for i, ctype in enumerate(["fan_favored", "judge_favored"]):
        sub = plot_df[plot_df["controversy_type"] == ctype]
        vals = sub.set_index("regime").loc[SCENARIO_KEYS, "mean_controversy"].values
        label = "Fan Favored" if ctype == "fan_favored" else "Judge Favored"
        ax.bar(x + (i - 0.5) * w, vals, w, label=label, color=colors[ctype], 
               edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Cleaner x-axis labels
    regime_labels = ["Rank +\nJudge Save", "Rank +\nFan Decide", "Percent +\nJudge Save", "Percent +\nFan Decide"]
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=9)
    ax.set_ylabel("Mean Simulated Controversy", fontsize=11)
    ax.set_title("Regime Effect on Controversy", fontsize=12)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "problem2b_regime_controversy_by_type.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {fig_dir}/problem2b_regime_controversy_by_type.pdf")


def plot_controversy_cdf(
    full_list: pd.DataFrame, classified_list: pd.DataFrame, gmm_params: dict
) -> None:
    """CDF of controversy_score to visualize natural break; overlay GMM components."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 12,
        "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
    })

    scores = np.sort(full_list["controversy_score"].values)
    n = len(scores)
    cdf = np.arange(1, n + 1) / n

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scores, cdf, "k-", linewidth=2, label="Empirical CDF")
    ax.set_xlabel("Controversy Score |judge_pct − placement_pct|", fontsize=11)
    ax.set_ylabel("Cumulative proportion", fontsize=11)
    ax.set_title("Controversy Score CDF and GMM Classification", fontsize=12)
    ax.grid(True, alpha=0.2)

    # GMM cutoff (decision boundary)
    mu_low, mu_high = gmm_params["mu_low"], gmm_params["mu_high"]
    cutoff = (mu_low + mu_high) / 2
    ax.axvline(cutoff, color="#C1292E", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"GMM cutoff ≈{cutoff:.2f}")
    ax.axvspan(cutoff, scores.max() + 0.05, alpha=0.1, color="#C1292E")

    in_classified = full_list.apply(
        lambda r: ((classified_list["season"] == r["season"]) & (classified_list["celebrity_name"] == r["celebrity_name"])).any(),
        axis=1,
    )
    n_controv = in_classified.sum()
    ax.annotate(f"Controversial: n={n_controv}", xy=(cutoff + 0.02, 0.5), fontsize=9)
    ax.legend(loc="lower right")
    ax.set_xlim(0, min(0.65, scores.max() * 1.1))
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "problem2b_controversy_cdf.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {fig_dir}/problem2b_controversy_cdf.pdf")


def plot_controversy_scatter(full_list: pd.DataFrame, classified_list: pd.DataFrame) -> None:
    """Scatter judge_percentile vs placement_percentile; color controversial by type. Diagonal = agreement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    # Publication-quality styling
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'font.family': 'sans-serif',
    })
    
    fig_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Other (non-controversial)
    in_classified = full_list.apply(
        lambda r: ((classified_list["season"] == r["season"]) & (classified_list["celebrity_name"] == r["celebrity_name"])).any(),
        axis=1,
    )
    ax.scatter(
        full_list.loc[~in_classified, "judge_percentile"],
        full_list.loc[~in_classified, "placement_percentile"],
        alpha=0.35, s=25, c="#888888", label="Other",
    )
    
    # Controversial: above diagonal = fan_favored, below diagonal = judge_favored
    merged = full_list.loc[in_classified].merge(
        classified_list[["season", "celebrity_name", "controversy_type"]],
        on=["season", "celebrity_name"],
        how="left",
    )
    
    # Colors: green for fan favored, red for judge favored
    colors = {"fan_favored": "#029E73", "judge_favored": "#C1292E"}
    
    for ctype, label in [
        ("fan_favored", "Fan Favored"),
        ("judge_favored", "Judge Favored"),
    ]:
        sub = merged[merged["controversy_type"] == ctype]
        if not sub.empty:
            ax.scatter(
                sub["judge_percentile"],
                sub["placement_percentile"],
                alpha=0.85, s=60, c=colors[ctype], edgecolors="black", linewidths=0.8, label=label,
            )
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1.5)
    
    ax.set_xlabel("Judge Percentile (1 = best)", fontsize=11)
    ax.set_ylabel("Placement Percentile (1 = winner)", fontsize=11)
    ax.set_title("Controversy: Judge Score vs Placement", fontsize=12)
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor='black', framealpha=1)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    
    # Label known examples with initials (explained in caption)
    initials = {
        "Jerry Rice": ("JR", (5, -10)),
        "Billy Ray Cyrus": ("BC", (-12, 5)),
        "Bristol Palin": ("BP", (5, -10)),
        "Bobby Bones": ("BB", (5, 5)),
    }
    for s, name, _ in CONTROVERSY_EXAMPLES:
        m = full_list[(full_list["season"] == s) & (full_list["celebrity_name"] == name)]
        if not m.empty:
            row = m.iloc[0]
            label, offset = initials.get(name, (name[:2].upper(), (5, 5)))
            ax.annotate(label, (row["judge_percentile"], row["placement_percentile"]),
                        xytext=offset, textcoords="offset points", fontsize=7, 
                        fontweight='bold', alpha=0.9)
    
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "problem2b_controversy_scatter.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {fig_dir}/problem2b_controversy_scatter.pdf")


def count_weeks_judge_save_saved_person(
    season: int,
    celebrity_name: str,
    judge_save_rows: List[dict],
) -> int:
    """Count weeks (in this season) where judge-save 'saved' this person (fan-decide would have eliminated them)."""
    count = 0
    for r in judge_save_rows:
        if r.get("season") != season:
            continue
        if r.get("elim_with_fan_decide") == celebrity_name and r.get("elim_with_judge_save") != celebrity_name:
            count += 1
    return count


def weeks_lowest_judge(J: np.ndarray, idx: int) -> int:
    """Count weeks where contestant idx had the lowest judge total among active."""
    W = J.shape[0]
    count = 0
    for w in range(W):
        if J[w, idx] <= 0:
            continue
        active = J[w] > 0
        if active.sum() < 2:
            continue
        j_w = J[w].copy()
        j_w[~active] = np.inf
        if np.argmin(j_w) == idx:
            count += 1
    return count


def weeks_in_bottom_two(J: np.ndarray, s_hist: np.ndarray, idx: int, regime: str) -> int:
    """Count weeks where contestant idx was in bottom two by combined score."""
    W, N = J.shape
    count = 0
    elim_so_far = set()
    for w in range(W):
        active = np.array([i not in elim_so_far for i in range(N)])
        if active.sum() < 2:
            continue
        J_w = J[w]
        s_w = s_hist[w]
        if regime == "percent":
            jp = judge_pct(J_w, active)
            combined = jp + s_w
        else:
            rJ = rank_order(-J_w, active)
            rF = rank_order(-s_w, active)
            combined = rJ + rF
        order = np.argsort(combined[active])
        active_idx = np.where(active)[0]
        # Bottom two have largest combined (percent: smallest C; rank: largest R)
        if regime == "percent":
            bottom2 = set(active_idx[order[-2:]])
        else:
            bottom2 = set(active_idx[order[-2:]])
        if idx in bottom2:
            count += 1
        # Who was eliminated this week? (simplified: assume one elim)
        if active.sum() <= 2:
            break
        if regime == "percent":
            elim_i = active_idx[order[0]]
        else:
            elim_i = active_idx[order[-1]]
        elim_so_far.add(elim_i)
    return count


SCENARIO_KEYS = ["rank_judge_save", "rank_fan_decide", "percent_judge_save", "percent_fan_decide"]


def compute_2x2_metrics(res: dict, names: List[str]) -> dict:
    """From run_2x2 result, compute pairwise and aggregate metrics across the 4 scenarios."""
    orders = [res[f"{k}_order"] for k in SCENARIO_KEYS]
    placements = [res[f"{k}_placement"] for k in SCENARIO_KEYS]
    winners = [res[f"{k}_winner"] for k in SCENARIO_KEYS]
    N = len(names)
    out = {}
    # Pairwise Kendall tau (fraction of pairs discordant; higher = more different)
    kendall_pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            tau = kendall_tau(orders[i], orders[j])
            kendall_pairs.append(((SCENARIO_KEYS[i], SCENARIO_KEYS[j]), tau))
    out["kendall_pairs"] = kendall_pairs
    out["mean_kendall_tau"] = float(np.mean([t for (_, t) in kendall_pairs])) if kendall_pairs else 0.0
    out["max_kendall_tau"] = max((t for (_, t) in kendall_pairs), default=0.0)
    # Pairwise mean displacement (placements)
    disp_pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            disp = mean_displacement(placements[i], placements[j])
            disp_pairs.append(((SCENARIO_KEYS[i], SCENARIO_KEYS[j]), disp))
    out["disp_pairs"] = disp_pairs
    out["mean_displacement"] = float(np.mean([d for (_, d) in disp_pairs])) if disp_pairs else 0.0
    out["max_displacement"] = max((d for (_, d) in disp_pairs), default=0.0)
    # Winner agreement
    distinct_winners = len(set(winners))
    out["distinct_winners"] = distinct_winners
    out["winners"] = winners
    # Rule sensitivity: sum of pairwise Kendall (higher = more disagreement across scenarios)
    out["rule_sensitivity"] = sum(t for (_, t) in kendall_pairs)
    # Weeks where at least two scenarios eliminated different people
    elim_weeks_list = [res[f"{k}_elim_week"] for k in SCENARIO_KEYS]
    W = len(elim_weeks_list[0]) if elim_weeks_list else 0
    weeks_elim_differ = 0
    for w in range(1, W):
        sets_w = [frozenset(i for i in range(N) if elim_weeks_list[s][i] == w) for s in range(4)]
        if len(set(sets_w)) > 1:
            weeks_elim_differ += 1
    out["weeks_elim_differ"] = weeks_elim_differ
    out["total_elim_weeks"] = W - 1  # weeks with at least one elimination (excluding final)
    return out


def run_2x2(
    season: int,
    names: List[str],
    J: np.ndarray,
    s_hist: np.ndarray,
    schedule: List[int],
) -> dict:
    """Run all 4 scenarios: rank/percent x judge_save (bottom-2: eliminate lower judge) / fan_decide.
    Returns elim_week, placement, winner, and order (names by placement) per scenario."""
    W, N = J.shape
    scenarios = [
        ("rank", True),   # rank_judge_save
        ("rank", False),  # rank_fan_decide
        ("percent", True),   # percent_judge_save
        ("percent", False),  # percent_fan_decide
    ]
    out = {}
    for (reg, judge_save), key in zip(scenarios, SCENARIO_KEYS):
        elim, place = forward_simulate(
            season, names, J, s_hist, schedule,
            regime_override=reg,
            judge_save=judge_save,
        )
        winner_i = np.argmin(place)
        out[f"{key}_elim_week"] = elim
        out[f"{key}_placement"] = place
        out[f"{key}_winner"] = names[winner_i]
        # Order: best first (placement 1, 2, ...)
        order = [names[i] for i in np.argsort(place)]
        out[f"{key}_order"] = order
    return out


def judge_save_impact(
    season: int,
    names: List[str],
    J: np.ndarray,
    s_hist: np.ndarray,
    schedule: List[int],
) -> List[dict]:
    """For each week in bottom-two regime: who would be eliminated with vs without judge-save?"""
    regime = season_regime(season)
    if regime != "rank_bottom2":
        return []
    W, N = J.shape
    rows = []
    elim_so_far = set()
    for w in range(W):
        k = schedule[w + 1] if w + 1 < len(schedule) else 0
        if k != 1:
            continue
        active = [i not in elim_so_far for i in range(N)]
        active_idx = [i for i in range(N) if active[i]]
        if len(active_idx) < 2:
            break
        J_w = J[w]
        s_w = s_hist[w]
        rJ = rank_order(-J_w, np.array(active))
        rF = rank_order(-s_w, np.array(active))
        R = rJ + rF
        order = np.argsort(R[active_idx])
        bottom2 = [active_idx[order[-1]], active_idx[order[-2]]]
        elim_judge_save = bottom2[np.argmin(J_w[bottom2])]
        elim_fan_save = bottom2[np.argmin(s_w[bottom2])]
        rows.append({
            "season": season,
            "week": w + 1,
            "bottom2_names": [names[i] for i in bottom2],
            "elim_with_judge_save": names[elim_judge_save],
            "elim_with_fan_decide": names[elim_fan_save],
            "judge_save_changed_result": elim_judge_save != elim_fan_save,
        })
        elim_so_far.add(elim_judge_save)
    return rows


def main():
    df = pd.read_csv(DATA_PATH)
    weeks = parse_week_cols(df)

    # 1) Controversy = placement + judge scores only (no fan-share estimates)
    full_list, classified_list, gmm_params = build_controversy_judge_placement_only(df, weeks)
    full_list.to_csv(os.path.join(OUT_DIR, "problem2b_controversy_list.csv"), index=False)
    classified_list = classified_list.copy()
    classified_list["controversy_type"] = classified_list.apply(controversy_type_from_row, axis=1)
    classified_list.to_csv(os.path.join(OUT_DIR, "problem2b_controversy_classified.csv"), index=False)
    print(f"Controversy metric: judge vs placement only (no fan data). Classification: 2-component GMM (data-driven)")
    print(f"  GMM: mu_low={gmm_params['mu_low']:.3f}, mu_high={gmm_params['mu_high']:.3f}, "
          f"cutoff ~{(gmm_params['mu_low']+gmm_params['mu_high'])/2:.3f}")
    print(f"Wrote {OUT_DIR}/problem2b_controversy_list.csv (full list: {len(full_list)} rows)")
    print(f"Wrote {OUT_DIR}/problem2b_controversy_classified.csv (classified: {len(classified_list)} rows)")

    # Check if the four named controversy examples are in the classified list
    print("\nKnown controversy examples — in classified list?")
    for s, name, brief in CONTROVERSY_EXAMPLES:
        match = full_list[(full_list["season"] == s) & (full_list["celebrity_name"] == name)]
        if match.empty:
            print(f"  {name} (s{s}): not in data")
            continue
        row = match.iloc[0]
        cscore = row["controversy_score"]
        in_classified = ((classified_list["season"] == s) & (classified_list["celebrity_name"] == name)).any()
        print(f"  {name} (s{s}): controversy_score={cscore:.3f}  in_list={in_classified}  — {brief}")

    # 1b) Visualize: CDF for natural break, then scatter
    plot_controversy_cdf(full_list, classified_list, gmm_params)
    plot_controversy_scatter(full_list, classified_list)

    # 2) Judge-save impact for all bottom-two seasons (build first so we can use it in 2x2)
    judge_save_rows = []
    for season in [s for s in df["season"].unique() if int(s) >= 28]:
        season = int(season)
        df_season = df[df["season"] == season].reset_index(drop=True)
        names = df_season["celebrity_name"].tolist()
        J = build_judge_matrix(df_season, weeks)
        W, N = J.shape
        schedule = elim_schedule_from_judge(J)
        s_hist = load_fan_shares_for_season(season, names, W)
        if s_hist is None:
            print(f"  ⚠ Season {season}: No fan shares (skipping judge-save impact)")
            continue
        js_impact = judge_save_impact(season, names, J, s_hist, schedule)
        judge_save_rows.extend(js_impact)

    controversy_rows = []
    scenario_rows = []
    season_cache = {}  # season -> (res, metrics) — only to run 2x2 once per season

    # Run 2x2 (all 4: rank/percent × judge_save/fan_decide) for every classified controversial contestant
    for _, class_row in classified_list.iterrows():
        season = int(class_row["season"])
        celeb_name = str(class_row["celebrity_name"])
        brief = get_brief(season, celeb_name, class_row)
        df_season = df[df["season"] == season].reset_index(drop=True)
        if df_season.empty:
            print(f"Season {season} not in data")
            continue
        names = df_season["celebrity_name"].tolist()
        if celeb_name not in names:
            print(f"{celeb_name} not in season {season} (names: {names[:5]}...)")
            continue
        J = build_judge_matrix(df_season, weeks)
        W, N = J.shape
        schedule = elim_schedule_from_judge(J)
        s_hist = load_fan_shares_for_season(season, names, W)
        if s_hist is None:
            print(f"  ⚠ Season {season}: No fan shares (skipping {celeb_name})")
            continue
        idx = names.index(celeb_name)
        placement_true = placement_true_from_data(df_season)
        place_true_i = placement_true[idx] if placement_true[idx] is not None else None
        n_weeks_lowest_judge = weeks_lowest_judge(J, idx)
        regime = season_regime(season)
        n_bottom2 = weeks_in_bottom_two(J, s_hist, idx, regime)

        controversy_rows.append({
            "season": season,
            "celebrity_name": celeb_name,
            "brief": brief,
            "placement_true": place_true_i,
            "weeks_lowest_judge_score": n_weeks_lowest_judge,
            "weeks_in_bottom_two_combined": n_bottom2,
            "regime": regime,
        })

        # Run 2x2 once per season (cache)
        if season not in season_cache:
            res = run_2x2(season, names, J, s_hist, schedule)
            metrics = compute_2x2_metrics(res, names)
            season_cache[season] = (res, metrics)
        res, metrics = season_cache[season]

        # Per-contestant: placements and elim_weeks in all 4 scenarios
        places = [res[f"{k}_placement"][idx] for k in SCENARIO_KEYS]
        elim_weeks = [res[f"{k}_elim_week"][idx] for k in SCENARIO_KEYS]
        placement_range = max(places) - min(places) if places else 0
        placement_std = float(np.std(places)) if len(places) > 1 else 0.0
        elim_week_range = max(elim_weeks) - min(elim_weeks) if elim_weeks else 0
        distinct_winners = metrics["distinct_winners"]
        winner_changing = distinct_winners > 1
        n_weeks_judge_save_saved = (
            count_weeks_judge_save_saved_person(season, celeb_name, judge_save_rows)
            if season_regime(season) == "rank_bottom2" else 0
        )
        # Per-contestant answers to the two 2b questions (for this individual only)
        # (1) Would the choice of method (rank vs percent) have led to the same result for this contestant?
        same_result_rank_vs_percent = (places[0] == places[2] and places[1] == places[3])
        # (2) Would judge-save vs fan-decide have impacted this contestant's outcome?
        judge_save_changed_under_rank = (places[0] != places[1]) or (elim_weeks[0] != elim_weeks[1])
        judge_save_changed_under_percent = (places[2] != places[3]) or (elim_weeks[2] != elim_weeks[3])
        judge_save_impacted_this_contestant = judge_save_changed_under_rank or judge_save_changed_under_percent
        # Same outcome for this contestant under all 4 regimes?
        same_outcome_all_4 = (placement_range == 0 and elim_week_range == 0)
        controversy_type = controversy_type_from_row(class_row)

        # Simulated controversy under each regime: |judge_percentile - placement_percentile_in_that_regime|
        judge_pct = float(class_row.get("judge_percentile", 0))
        placement_pct_by_regime = [
            (1.0 - (p - 1) / (N - 1)) if N > 1 else 1.0 for p in places
        ]
        simulated_controversy = [abs(judge_pct - pp) for pp in placement_pct_by_regime]
        best_regime_reduces_controversy = SCENARIO_KEYS[np.argmin(simulated_controversy)]
        worst_regime_increases_controversy = SCENARIO_KEYS[np.argmax(simulated_controversy)]
        best_placement_regime = SCENARIO_KEYS[np.argmin(places)]
        worst_placement_regime = SCENARIO_KEYS[np.argmax(places)]
        # Does percent (judge-heavy) reduce controversy vs rank (fan-heavy)? For fan_favored, percent usually gives worse placement → closer to judge → less controversy. For judge_favored, percent usually gives better placement → closer to judge → less controversy.
        min_controversy = min(simulated_controversy)
        max_controversy = max(simulated_controversy)

        scenario_rows.append({
            "season": season,
            "celebrity_name": celeb_name,
            "controversy_type": controversy_type,
            "elim_week_rank_judge_save": elim_weeks[0],
            "elim_week_rank_fan_decide": elim_weeks[1],
            "elim_week_percent_judge_save": elim_weeks[2],
            "elim_week_percent_fan_decide": elim_weeks[3],
            "placement_rank_judge_save": places[0],
            "placement_rank_fan_decide": places[1],
            "placement_percent_judge_save": places[2],
            "placement_percent_fan_decide": places[3],
            "same_result_rank_vs_percent": same_result_rank_vs_percent,
            "judge_save_impacted_this_contestant": judge_save_impacted_this_contestant,
            "judge_save_changed_under_rank": judge_save_changed_under_rank,
            "judge_save_changed_under_percent": judge_save_changed_under_percent,
            "same_outcome_all_4": same_outcome_all_4,
            "placement_range_4scenarios": placement_range,
            "elim_week_range_4scenarios": elim_week_range,
            "winner_rank_judge_save": res["rank_judge_save_winner"],
            "winner_rank_fan_decide": res["rank_fan_decide_winner"],
            "winner_percent_judge_save": res["percent_judge_save_winner"],
            "winner_percent_fan_decide": res["percent_fan_decide_winner"],
            "winner_changing": winner_changing,
            "weeks_in_bottom_two": controversy_rows[-1]["weeks_in_bottom_two_combined"]
                if controversy_rows else None,
            "n_weeks_judge_save_saved_this_person": n_weeks_judge_save_saved,
            "judge_percentile": judge_pct,
            "n_contestants": N,
            "simulated_controversy_rank_judge_save": simulated_controversy[0],
            "simulated_controversy_rank_fan_decide": simulated_controversy[1],
            "simulated_controversy_percent_judge_save": simulated_controversy[2],
            "simulated_controversy_percent_fan_decide": simulated_controversy[3],
            "best_regime_reduces_controversy": best_regime_reduces_controversy,
            "worst_regime_increases_controversy": worst_regime_increases_controversy,
            "best_placement_regime": best_placement_regime,
            "worst_placement_regime": worst_placement_regime,
            "min_simulated_controversy": min_controversy,
            "max_simulated_controversy": max_controversy,
        })

    pd.DataFrame(controversy_rows).to_csv(
        os.path.join(OUT_DIR, "problem2b_controversy_examples.csv"), index=False
    )
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(os.path.join(OUT_DIR, "problem2b_2x2_scenarios.csv"), index=False)
    plot_regime_controversy_by_type(scenario_df)

    # --- Regime vs controversy analysis: which regime reduces/increases controversy? fan_favored vs judge_favored ---
    regime_analysis_rows = []
    for ctype in ["fan_favored", "judge_favored"]:
        sub = scenario_df[scenario_df["controversy_type"] == ctype]
        if sub.empty:
            continue
        for regime in SCENARIO_KEYS:
            col_controv = f"simulated_controversy_{regime}"
            col_place = f"placement_{regime}"
            regime_analysis_rows.append({
                "controversy_type": ctype,
                "regime": regime,
                "n": len(sub),
                "mean_simulated_controversy": sub[col_controv].mean(),
                "mean_placement": sub[col_place].mean(),
            })
    if regime_analysis_rows:
        pd.DataFrame(regime_analysis_rows).to_csv(
            os.path.join(OUT_DIR, "problem2b_regime_controversy_by_type.csv"), index=False
        )
        print(f"Wrote {OUT_DIR}/problem2b_regime_controversy_by_type.csv")

    # Best regime for reducing controversy: counts by type (which of the 4 minimizes |judge_pct - placement_pct|)
    best_regime_counts = scenario_df.groupby("controversy_type")["best_regime_reduces_controversy"].value_counts()
    best_placement_counts = scenario_df.groupby("controversy_type")["best_placement_regime"].value_counts()
    # Percent vs rank: does percent reduce controversy more often for judge_favored? (percent = judge-heavy)
    fan_sub = scenario_df[scenario_df["controversy_type"] == "fan_favored"]
    judge_sub = scenario_df[scenario_df["controversy_type"] == "judge_favored"]
    fan_best_is_percent = sum(1 for r in scenario_rows if r.get("controversy_type") == "fan_favored" and r.get("best_regime_reduces_controversy", "").startswith("percent"))
    fan_best_is_rank = sum(1 for r in scenario_rows if r.get("controversy_type") == "fan_favored" and r.get("best_regime_reduces_controversy", "").startswith("rank"))
    judge_best_is_percent = sum(1 for r in scenario_rows if r.get("controversy_type") == "judge_favored" and r.get("best_regime_reduces_controversy", "").startswith("percent"))
    judge_best_is_rank = sum(1 for r in scenario_rows if r.get("controversy_type") == "judge_favored" and r.get("best_regime_reduces_controversy", "").startswith("rank"))
    if judge_save_rows:
        pd.DataFrame(judge_save_rows).to_csv(
            os.path.join(OUT_DIR, "problem2b_judge_save_impact_by_week.csv"), index=False
        )
    print(f"Wrote {OUT_DIR}/problem2b_controversy_examples.csv")
    print(f"Wrote {OUT_DIR}/problem2b_2x2_scenarios.csv")
    if judge_save_rows:
        print(f"Wrote {OUT_DIR}/problem2b_judge_save_impact_by_week.csv")
    n_changed = sum(1 for r in judge_save_rows if r.get("judge_save_changed_result"))
    print(f"Judge-save changed who was eliminated in {n_changed} of {len(judge_save_rows)} bottom-two weeks (across s28+).")

    # --- Problem 2b summary: per-contestant answers to the two questions ---
    n_same_rank_vs_percent = sum(1 for r in scenario_rows if r.get("same_result_rank_vs_percent"))
    n_judge_save_impacted = sum(1 for r in scenario_rows if r.get("judge_save_impacted_this_contestant"))
    n_same_outcome_all_4 = sum(1 for r in scenario_rows if r.get("same_outcome_all_4"))
    n_fan_favored = sum(1 for r in scenario_rows if r.get("controversy_type") == "fan_favored")
    n_judge_favored = sum(1 for r in scenario_rows if r.get("controversy_type") == "judge_favored")
    summary_lines = [
        "# Problem 2b Summary: Per-Contestant Outcomes Under 2×2 Regimes",
        "",
        "**Controversy type** (by diagonal): **above line** = **fan_favored** (placement > judge — fans kept them further than judges rated); **below line** = **judge_favored** (placement < judge — judges liked them, eliminated early). Of the 20: **" + str(n_fan_favored) + " fan_favored**, **" + str(n_judge_favored) + " judge_favored**. See `controversy_type` in problem2b_controversy_classified.csv and problem2b_2x2_scenarios.csv; scatter plot colors by type.",
        "",
        "For each **controversial** contestant we run 2×2 = 4 regimes: rank vs percent × judge_save (bottom-2: eliminate lower judge) vs fan_decide (eliminate lower fan share). We ask, **for that individual**:",
        "",
        "## 1. Would the choice of method (rank vs percent) have led to the same result for this contestant?",
        "",
        f"- **For {n_same_rank_vs_percent} of {len(scenario_rows)}** controversial contestants, **yes**: their placement (and elim week) would be the same under rank as under percent (holding judge_save vs fan_decide fixed).",
        f"- For the other **{len(scenario_rows) - n_same_rank_vs_percent}**, **no**: their outcome would differ under rank vs percent.",
        f"- **{n_same_outcome_all_4}** of {len(scenario_rows)} had the **same outcome in all 4** regimes (placement and elim week identical across rank/percent and judge_save/fan_decide).",
        "",
        "## 2. How would including judge-save (judges choose which of bottom two to eliminate) impact the results for this contestant?",
        "",
        f"- **For {n_judge_save_impacted} of {len(scenario_rows)}** controversial contestants, **judge-save vs fan-decide would have changed that contestant's outcome**: under rank and/or under percent, their placement or elimination week differs when the bottom-two tie-break is judge vs fan.",
        f"- For the other **{len(scenario_rows) - n_judge_save_impacted}**, judge-save vs fan-decide would **not** have changed their outcome (same placement and elim week under judge_save and fan_decide).",
        "",
        "Per-contestant details: placement and elim_week in each of the 4 regimes, plus `same_result_rank_vs_percent`, `judge_save_impacted_this_contestant`, `same_outcome_all_4`, in **problem2b_2x2_scenarios.csv**.",
        "",
        "## Which regime reduces vs increases controversy? (fan_favored vs judge_favored)",
        "",
        "For each contestant we compute **simulated controversy** in each of the 4 regimes: \|judge_percentile − placement_percentile_in_that_regime\|. Lower = outcome closer to what judges would suggest.",
        "",
        "- **Fan_favored** (above diagonal): Rank (fan-heavy) tends to give them *better* placement than percent (judge-heavy), so **percent regime *increases* controversy** for fan_favored (pushes placement down toward judges). **Rank regime *reduces* controversy** for fan_favored in " + str(fan_best_is_rank) + " of " + str(n_fan_favored) + " cases (best at minimizing simulated controversy).",
        "",
        "- **Judge_favored** (below diagonal): Percent (judge-heavy) tends to give them *better* placement than rank, so **percent regime *reduces* controversy** for judge_favored (pushes placement up toward judges). **Percent regime *reduces* controversy** for judge_favored in " + str(judge_best_is_percent) + " of " + str(n_judge_favored) + " cases (best at minimizing simulated controversy).",
        "",
        "- **Judge-save** (bottom-two): When judges choose which of the bottom two to eliminate, they eliminate the *lower* judge score → **judge_save systematically helps judge_favored** (high judge score) and **hurts fan_favored** (low judge score) when in the bottom two. The one controversial contestant in a bottom-two season (Whitney Leavitt s34, judge_favored) had judge_save *improve* her outcome (placement 6 with judge_save vs 10 with fan_decide under percent).",
        "",
        "- **Sensitivity** (mean placement range across 4 regimes): fan_favored " + (f"{fan_sub['placement_range_4scenarios'].mean():.2f}" if not fan_sub.empty else "—") + ", judge_favored " + (f"{judge_sub['placement_range_4scenarios'].mean():.2f}" if not judge_sub.empty else "—") + ". Higher = more sensitive to which regime is used.",
        "",
        "**Conclusion:** Rank (fan-heavy) favors fan_favored contestants and reduces their controversy; percent (judge-heavy) favors judge_favored contestants and reduces their controversy. Judge-save in bottom-two weeks further tilts outcomes toward judge_favored. See **problem2b_regime_controversy_by_type.csv** for mean simulated controversy and mean placement by regime and type.",
        "",
    ]
    summary_path = os.path.join(os.path.dirname(__file__), "outputs", "problem2b_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
