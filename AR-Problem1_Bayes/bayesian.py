"""
AR-Problem1_Bayes/bayesian.py
-----------------------------
This script infers *latent* fan-vote shares for Dancing With The Stars using only:
- weekly judges' scores, and
- the observed elimination week (from the `results` field).

It does *not* fit a fully Bayesian probabilistic model with an explicit likelihood in a
probabilistic programming language (PyMC/Stan). Instead, it uses:

1) A differentiable objective ("loss") that combines:
   - soft elimination likelihood (via a softmax over a risk score),
   - temporal smoothness / dynamics between weeks (optional),
   - L2 regularization on latent logits (prevents extreme values),
   - entropy regularization on inferred shares (prevents degenerate shares),
   - optional finals placement constraint (very approximate).

2) MAP-style optimization (Adam) to find one best-fit set of latent vote-share logits `p`.

3) Optional SGLD (stochastic gradient Langevin dynamics) to obtain a crude uncertainty
   estimate over shares by sampling around the MAP solution.

Outputs
-------
- inferred_shares_bayes.csv: MAP fan share per (season, week, celebrity)
- inferred_shares_bayes_unc.csv: uncertainty summary (mean/std/quantiles) if SGLD enabled
- elimination_match_bayes.csv: per-season elimination \"match rate\" diagnostic

Important modeling assumption
-----------------------------
Active contestants in week w are inferred as those with judges' total score > 0 in that week.
This matches how the provided CSV uses zeros after elimination.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install torch") from exc


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1_Bayes/results/bayesian/inferred_shares_bayes.csv"
UNC_PATH = "AR-Problem1_Bayes/results/bayesian/inferred_shares_bayes_unc.csv"
MATCH_PATH = "AR-Problem1_Bayes/results/bayesian/elimination_match_bayes.csv"

EPS = 1e-8
TEMP = 0.7
RANK_TAU = 10.0
LAM_DYN = 0.5
LAM_ETA = 1.0
LAM_HARD = 150.0
LAM_FINAL = 10.0
LAM_COEF = 0.1

LR = 0.05  # Adam learning rate
N_STEPS = 250  # Adam optimization steps per season

RUN_SGLD = True
SGLD_STEPS = 80     # total SGLD iterations
SGLD_BURNIN = 20    # discard early SGLD samples
SGLD_INTERVAL = 5   # keep every k-th SGLD sample after burn-in


def parse_week_cols(df: pd.DataFrame) -> List[str]:
    """Return all `week{w}_judge{j}_score` columns present in the wide CSV."""
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df: pd.DataFrame, week: int, cols: List[str]) -> pd.Series:
    """
    Compute judges' *total points* for each contestant in a given week by summing
    that week's judge score columns.
    """
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def regime_for_season(season: int) -> str:
    """
    Determine scoring regime by season index (per problem statement assumptions).
    - seasons 1–2: rank-based era (treated as \"rank\")
    - seasons 3–27: percent-combination era (treated as \"percent\")
    - seasons 28+: bottom-two/judges-save era (treated as \"bottom\")
    """
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom"


def build_season_struct(df_season: pd.DataFrame, week_cols: List[str]) -> Dict:
    """
    Convert one season's wide rows into a dense season structure.

    Returns dict containing:
    - names: contestant names in season order
    - J: (W x N) matrix of weekly judges totals (zeros after elimination)
    - elim_week: list of elimination week numbers (1-indexed) per contestant
    - max_week: last week with any nonzero total across contestants
    """
    max_week = max(int(re.search(r"week(\d+)_", c).group(1)) for c in week_cols)
    names = df_season["celebrity_name"].tolist()
    n = len(names)
    J = np.zeros((max_week, n), dtype=float)

    for w in range(1, max_week + 1):
        J[w - 1] = week_score(df_season, w, week_cols).to_numpy()

    week_idx = np.arange(1, max_week + 1)[:, None]
    last_active = (np.where(J > 0, week_idx, 0)).max(axis=0)
    elim_week = []
    for i, row in df_season.iterrows():
        # Parse elimination week from free-text `results`.
        # Withdrew is treated as eliminated in the last active week for this implementation.
        if isinstance(row["results"], str) and "Eliminated Week" in row["results"]:
            elim_week.append(int(row["results"].split("Eliminated Week ")[1]))
        elif isinstance(row["results"], str) and "Withdrew" in row["results"]:
            elim_week.append(int(last_active[df_season.index.get_loc(i)]))
        else:
            elim_week.append(None)

    # Determine the number of active weeks in this season based on any positive total score.
    max_week_active = int(np.where(J.sum(axis=1) > 0)[0].max() + 1)
    return {
        "names": names,
        "J": J[:max_week_active],
        "elim_week": elim_week,
        "max_week": max_week_active,
    }


def compute_jz(J: np.ndarray) -> np.ndarray:
    """
    Compute a per-week z-score normalization of judges totals among active contestants.
    This is used in the temporal dynamics term so weeks are on comparable scales.
    """
    W, N = J.shape
    Jz = np.zeros_like(J)
    for w in range(W):
        mask = J[w] > 0
        if mask.sum() < 2:
            continue
        mean = J[w, mask].mean()
        std = J[w, mask].std()
        Jz[w, mask] = (J[w, mask] - mean) / (std + EPS)
    return Jz


def build_feature_indices(df_season: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    age = pd.to_numeric(
        df_season["celebrity_age_during_season"], errors="coerce"
    ).to_numpy()
    age_mean = np.nanmean(age)
    age_std = np.nanstd(age)
    age_z = (age - age_mean) / (age_std + EPS)

    industry = df_season["celebrity_industry"].fillna("Unknown").tolist()
    industry_ids = {v: i for i, v in enumerate(sorted(set(industry)))}
    industry_idx = np.array([industry_ids[v] for v in industry], dtype=int)

    partners = df_season["ballroom_partner"].fillna("Unknown").tolist()
    partner_ids = {v: i for i, v in enumerate(sorted(set(partners)))}
    partner_idx = np.array([partner_ids[v] for v in partners], dtype=int)

    return age_z, industry_idx, partner_idx


def elimination_match_rate(shares, struct, regime):
    """
    Diagnostic: how often does the inferred share matrix predict the observed elimination?

    For each week:
    - Determine active contestants (J>0).
    - Identify the true eliminated contestant(s) for that week.
    - Predict \"at-risk\" contestants according to the regime:
        * percent: lowest combined (judges percent + inferred fan share)
        * rank/bottom: use rank aggregation approximations
    Returns (hits, total_weeks_evaluated).
    """
    J = struct["J"]
    elim_week = struct["elim_week"]
    W, N = J.shape
    hits = 0
    total = 0

    for w in range(W):
        idx = np.where(J[w] > 0)[0]
        if len(idx) < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx]
        total += 1
        if not elim:
            hits += 1
            continue

        k_elim = len(elim)
        if regime == "percent":
            j_pct = J[w, idx] / J[w, idx].sum()
            C = np.full(N, np.inf)
            for k, i in enumerate(idx):
                C[i] = j_pct[k] + shares[w, i]
            pred = set(np.argsort(C)[:k_elim])
        else:
            rF = np.argsort(-shares[w, idx]).argsort() + 1
            rJ = (
                pd.Series(J[w, idx])
                .rank(ascending=False, method="first")
                .to_numpy()
            )
            R = rJ + rF
            if regime == "bottom" and k_elim == 1:
                worst = np.argsort(-R)[:2]
                pred = set(idx[worst])
            else:
                worst = np.argsort(-R)[:k_elim]
                pred = set(idx[worst])

        if set(elim).issubset(pred):
            hits += 1
    return hits, total


def build_mask(J: np.ndarray) -> np.ndarray:
    """Binary mask (W x N): 1 if contestant is active in week (judge total > 0), else 0."""
    mask = (J > 0).astype(float)
    return mask


def softmax_masked(p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Softmax over contestants *within a week*, ignoring inactive contestants by assigning
    them a very negative logit (-1e9).
    """
    neg_inf = torch.full_like(p, -1e9)
    masked = torch.where(mask > 0, p, neg_inf)
    return torch.softmax(masked, dim=1)


def loss_for_season(
    p: torch.Tensor,
    J: torch.Tensor,
    elim_week: List[Optional[int]],
    active_mask: torch.Tensor,
    Jz: torch.Tensor,
    placement_rank: torch.Tensor,
    age_z: torch.Tensor,
    industry_idx: torch.Tensor,
    partner_idx: torch.Tensor,
    beta0: torch.Tensor,
    beta_age: torch.Tensor,
    beta_judge: torch.Tensor,
    beta_industry: torch.Tensor,
    beta_partner: torch.Tensor,
    b_contestant: torch.Tensor,
    gamma_raw: torch.Tensor,
    regime: str,
) -> torch.Tensor:
    """
    Core training objective for one season.

    Parameters (high-level):
    - p: (W x N) unconstrained logits whose masked softmax gives weekly fan shares
    - J: (W x N) weekly judges totals (zeros after elimination)
    - elim_week: elimination week per contestant (1-indexed), inferred from `results`
    - active_mask: (W x N) 1 for active contestants
    - Jz: (W x N) z-scored judges totals within each week for dynamics regularization
    - regime: determines whether \"percent\" or \"rank\" or \"bottom\" rules apply

    The loss includes:
    - dynamics regularization (encourages p[w] to be predictable from p[w-1] + judges signal)
    - L2 regularization on p (prevents extreme logits)
    - per-week elimination likelihood + hard constraints
    - optional finals placement constraint
    """
    W, N = J.shape
    s = softmax_masked(p, active_mask)
    loss = torch.tensor(0.0, device=p.device)

    for w in range(W):
        shared = active_mask[w] > 0
        if not shared.any():
            continue
        eta = (
            beta0
            + beta_age * age_z
            + beta_judge * Jz[w]
            + beta_industry[industry_idx]
            + beta_partner[partner_idx]
            + b_contestant
        )
        loss = loss + LAM_ETA * torch.mean((p[w, shared] - eta[shared]) ** 2)

    gamma = torch.tanh(gamma_raw)
    for w in range(1, W):
        shared = (active_mask[w] > 0) & (active_mask[w - 1] > 0)
        if shared.any():
            pred = p[w - 1] + gamma * Jz[w]
            loss = loss + LAM_DYN * torch.mean((p[w, shared] - pred[shared]) ** 2)

    # L2 prior on latent states to prevent extreme logits
    loss = loss + LAM_P * torch.mean((p * active_mask) ** 2)

    # Likelihood (soft elimination) + hard-constraint penalties:
    # - compute a per-week \"risk\" score that should be high for eliminated contestants.
    # - add log-softmax loss so eliminated are likely under risk.
    # - add large hinge penalties so eliminated risk > non-eliminated risk (hard ordering).
    for w in range(W):
        idx = torch.where(active_mask[w] > 0)[0]
        if idx.numel() < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx.tolist()]
        if not elim:
            continue
        if regime == "percent":
            # Percent regime uses judges percent in that week.
            q = J[w, idx] / (J[w, idx].sum() + EPS)
            C = q + s[w, idx]
            logits = -C / TEMP
        else:
            # Rank-like regimes approximate judges rank and normalize it to [0,1].
            rJ = torch.from_numpy(
                pd.Series(J[w, idx].cpu().numpy())
                .rank(ascending=False, method="first")
                .to_numpy()
            ).to(p.device)
            q = (rJ - 1) / (rJ.numel() - 1 + EPS)

        # Normalize scales within week to reduce dominance by one term.
        q_z = (q - q.mean()) / (q.std() + EPS)
        log_s = torch.log(s[w, idx] + EPS)
        log_s_z = (log_s - log_s.mean()) / (log_s.std() + EPS)

        a = torch.nn.functional.softplus(a_raw)
        b = torch.nn.functional.softplus(b_raw)
        risk = a * q_z - b * log_s_z

        logits = risk / TEMP
        log_probs = torch.log_softmax(logits, dim=0)
        loss = loss - torch.sum(log_probs[[idx.tolist().index(e) for e in elim]])


        # Hard-constraint penalty: each eliminated contestant should have higher risk
        # than each non-eliminated contestant (pairwise hinge losses).
        elim_pos = [idx.tolist().index(e) for e in elim]
        non_elim_pos = [i for i in range(idx.numel()) if i not in elim_pos]
        for e_pos in elim_pos:
            for j_pos in non_elim_pos:
                if regime == "percent":
                    loss = loss + LAM_HARD * torch.relu(C[e_pos] - C[j_pos])
                else:
                    loss = loss + LAM_HARD * torch.relu(R[j_pos] - R[e_pos])

        # Finals placement likelihood (very approximate):
        # tries to match overall placement ordering using a sequential choice model.
    if LAM_FINAL > 0 and placement_rank.numel() == J.shape[1]:
        pr = placement_rank
        valid = torch.where(torch.isfinite(pr))[0]
        if valid.numel() >= 2:
            season_score = torch.zeros_like(pr)
            for w in range(W):
                idx = torch.where(active_mask[w] > 0)[0]
                if idx.numel() < 2:
                    continue
                q = J[w, idx] / (J[w, idx].sum() + EPS)
                C = q + s[w, idx]
                season_score[idx] = season_score[idx] + C
            order = valid[torch.argsort(pr[valid], descending=False)]
            remaining = valid.clone()
            for pos in order:
                logits = season_score[remaining] / TEMP
                log_probs = torch.log_softmax(logits, dim=0)
                choose_idx = torch.where(remaining == pos)[0][0]
                loss = loss - LAM_FINAL * log_probs[choose_idx]
                remaining = torch.tensor(
                    [r for r in remaining.tolist() if r != pos.item()],
                    device=p.device,
                    dtype=remaining.dtype,
                )
                if remaining.numel() < 2:
                    break
            # Hard ordering for placements (winner should have highest season_score)
            for i in valid.tolist():
                for j in valid.tolist():
                    if pr[i] < pr[j]:
                        loss = loss + LAM_FINAL * torch.relu(
                            season_score[j] - season_score[i]
                        )
    return loss


def fit_season(df_season, week_cols, regime, device):
    """
    Fit one season:
    - build season structure (J, elim_week)
    - optimize p (weekly logits) and a few global parameters using Adam
    - optionally run SGLD to sample around the MAP logits for uncertainty bands
    """
    struct = build_season_struct(df_season, week_cols)
    J = torch.tensor(struct["J"], dtype=torch.float32, device=device)
    active_mask = torch.tensor(build_mask(struct["J"]), dtype=torch.float32, device=device)
    W, N = J.shape

    Jz = torch.tensor(compute_jz(struct["J"]), dtype=torch.float32, device=device)
    age_z, industry_idx, partner_idx = build_feature_indices(df_season)
    age_z = torch.tensor(age_z, dtype=torch.float32, device=device)
    industry_idx = torch.tensor(industry_idx, dtype=torch.long, device=device)
    partner_idx = torch.tensor(partner_idx, dtype=torch.long, device=device)
    placement_rank = (
        pd.to_numeric(df_season["placement"], errors="coerce")
        .rank(ascending=True, method="average")
        .to_numpy()
    )
    placement_rank = torch.tensor(placement_rank, dtype=torch.float32, device=device)

    # p[w,i] are unconstrained logits; shares are softmax_masked(p[w], active_mask[w]).
    p = torch.zeros((W, N), dtype=torch.float32, requires_grad=True, device=device)
    beta0 = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    beta_age = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    beta_judge = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    beta_industry = torch.zeros(
        int(industry_idx.max().item() + 1),
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    beta_partner = torch.zeros(
        int(partner_idx.max().item() + 1),
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    b_contestant = torch.zeros(N, dtype=torch.float32, requires_grad=True, device=device)
    gamma_raw = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    opt = torch.optim.Adam(
        [
            p,
            beta0,
            beta_age,
            beta_judge,
            beta_industry,
            beta_partner,
            b_contestant,
            gamma_raw,
        ],
        lr=LR,
    )

    # MAP-like optimization (gradient descent on the differentiable loss).
    for _ in range(N_STEPS):
        opt.zero_grad()
        loss = loss_for_season(
            p,
            J,
            struct["elim_week"],
            active_mask,
            Jz,
            placement_rank,
            age_z,
            industry_idx,
            partner_idx,
            beta0,
            beta_age,
            beta_judge,
            beta_industry,
            beta_partner,
            b_contestant,
            gamma_raw,
            regime,
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        s_map = softmax_masked(p, active_mask).cpu().numpy()

    # SGLD samples for uncertainty: inject Gaussian noise into gradient steps
    # so we sample around a local mode (very approximate posterior).
    samples = []
    if RUN_SGLD:
        p_sgld = p.detach().clone().requires_grad_(True)
        for t in range(SGLD_STEPS):
            loss = loss_for_season(
                p_sgld,
                J,
                struct["elim_week"],
                active_mask,
                Jz,
                placement_rank.detach(),
                age_z.detach(),
                industry_idx.detach(),
                partner_idx.detach(),
                beta0.detach(),
                beta_age.detach(),
                beta_judge.detach(),
                beta_industry.detach(),
                beta_partner.detach(),
                b_contestant.detach(),
                gamma_raw.detach(),
                regime,
            )
            loss.backward()
            noise = torch.randn_like(p_sgld) * np.sqrt(2 * LR)
            p_sgld = (
                p_sgld - LR * p_sgld.grad + noise
            ).detach().clone().requires_grad_(True)
            if t >= SGLD_BURNIN and (t - SGLD_BURNIN) % SGLD_INTERVAL == 0:
                with torch.no_grad():
                    samples.append(softmax_masked(p_sgld, active_mask).cpu().numpy())

    return struct, s_map, samples


def main() -> None:
    """Run the full pipeline across seasons and write CSV outputs."""
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    week_cols = parse_week_cols(df)
    device = torch.device("cpu")

    records = []
    unc_rows = []
    match_rows = []

    for season in sorted(df["season"].unique()):
        df_season = df[df["season"] == season].reset_index(drop=True)
        regime = regime_for_season(season)
        print(f"Solving season {season} ({regime}) with Bayesian MAP...")

        struct, shares, samples = fit_season(df_season, week_cols, regime, device)
        hits, total = elimination_match_rate(shares, struct, regime)
        match_rows.append(
            {
                "season": season,
                "match_rate": hits / total if total else 1.0,
                "weeks": total,
            }
        )

        # Convert SGLD samples into uncertainty summaries.
        if samples:
            arr = np.stack(samples, axis=0)
            mean = arr.mean(axis=0)
            median = np.quantile(arr, 0.5, axis=0)
            std = arr.std(axis=0)
            p10 = np.quantile(arr, 0.1, axis=0)
            p90 = np.quantile(arr, 0.9, axis=0)
        else:
            mean = shares.copy()
            median = shares.copy()
            std = np.zeros_like(shares)
            p10 = shares.copy()
            p90 = shares.copy()

        # Emit one row per (season, week, contestant) for active contestants.
        for w in range(struct["max_week"]):
            for i, name in enumerate(struct["names"]):
                if shares[w, i] > 0:
                    records.append(
                        {
                            "season": season,
                            "week": w + 1,
                            "celebrity_name": name,
                            "s_map": shares[w, i],
                        }
                    )
                    unc_rows.append(
                        {
                            "season": season,
                            "week": w + 1,
                            "celebrity_name": name,
                            "s_mean": mean[w, i],
                            "s_p50": median[w, i],
                            "s_std": std[w, i],
                            "s_p10": p10[w, i],
                            "s_p90": p90[w, i],
                        }
                    )

    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    pd.DataFrame(unc_rows).to_csv(UNC_PATH, index=False)
    pd.DataFrame(match_rows).to_csv(MATCH_PATH, index=False)
    print(f"Wrote {OUT_PATH}, {UNC_PATH}, {MATCH_PATH}")


if __name__ == "__main__":
    main()
