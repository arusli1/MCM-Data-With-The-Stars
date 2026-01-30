import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install torch") from exc


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_SHARES = "AR-Problem1_Bayes/results/bayesian_2/inferred_shares.csv"
OUT_UNC = "AR-Problem1_Bayes/results/bayesian_2/inferred_uncertainty.csv"
OUT_MATCH = "AR-Problem1_Bayes/results/bayesian_2/match_rates.csv"
POP_PATH = "AR-Problem1_Bayes/popularity_trends_season.csv"
POP_WEIGHT = 1.0

EPS = 1e-8

# Training
LR = 0.03
N_STEPS = 300

# Likelihood temperatures
KAPPA_INIT = 1.0
RANK_TAU = 10.0

# Prior strengths
LAM_P = 1.0
LAM_SMOOTH = 0.05
LAM_COEF = 0.2
LAM_FINAL = 5.0
LAM_ENT = 0.05
LAM_PMAG = 0.05
LAM_HARD = 50.0
ELIM_PERM_SAMPLES = 12
FINAL_TOP_K = 4

# Uncertainty
RUN_SGLD = True
SGLD_STEPS = 80
SGLD_BURNIN = 20
SGLD_INTERVAL = 5
SGLD_STEP = 0.005


def parse_week_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df: pd.DataFrame, week: int, cols: List[str]) -> pd.Series:
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


def parse_elim_week(results: str) -> Optional[int]:
    if not isinstance(results, str):
        return None
    match = re.search(r"Eliminated Week (\d+)", results)
    if match:
        return int(match.group(1))
    if "Withdrew" in results:
        return -1
    return None


def regime_for_season(season: int) -> str:
    if season <= 2:
        return "rank"
    if season <= 27:
        return "percent"
    return "bottom"


def build_season_struct(df_season: pd.DataFrame, week_cols: List[str]) -> Dict:
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
        ew = parse_elim_week(row.get("results"))
        if ew == -1:
            ew = int(last_active[df_season.index.get_loc(i)])
        elim_week.append(ew)

    max_week_active = int(np.where(J.sum(axis=1) > 0)[0].max() + 1)
    return {
        "names": names,
        "J": J[:max_week_active],
        "elim_week": elim_week,
        "max_week": max_week_active,
    }


def compute_jz(J: np.ndarray) -> np.ndarray:
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


def build_mask(J: np.ndarray) -> np.ndarray:
    return (J > 0).astype(float)


def softmax_masked(p: torch.Tensor, mask: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    neg_inf = torch.full_like(p, -1e9)
    masked = torch.where(mask > 0, p / tau, neg_inf)
    return torch.softmax(masked, dim=1)


def center_p(p: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    p_centered = p.clone()
    for w in range(p.shape[0]):
        idx = torch.where(active_mask[w] > 0)[0]
        if idx.numel() > 0:
            mean = p_centered[w, idx].mean()
            p_centered[w, idx] = p_centered[w, idx] - mean
    return p_centered


def soft_rank_from_shares(s: torch.Tensor) -> torch.Tensor:
    diff = (s.unsqueeze(0) - s.unsqueeze(1)) * RANK_TAU
    rF = 1.0 + torch.sigmoid(diff).sum(dim=1)
    return rF


def pl_log_prob_ordered(logits: torch.Tensor, order: List[int]) -> torch.Tensor:
    remaining = logits
    idx = torch.arange(logits.numel(), device=logits.device)
    logp = torch.tensor(0.0, device=logits.device)
    for chosen in order:
        log_probs = torch.log_softmax(remaining, dim=0)
        choose_idx = torch.where(idx == chosen)[0][0]
        logp = logp + log_probs[choose_idx]
        keep = idx != chosen
        idx = idx[keep]
        remaining = remaining[keep]
        if remaining.numel() == 0:
            break
    return logp


def logmeanexp(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 1:
        return values[0]
    return torch.logsumexp(values, dim=0) - torch.log(
        torch.tensor(values.numel(), device=values.device, dtype=values.dtype)
    )


def observed_elim_log_prob(logits: torch.Tensor, elim_local: List[int]) -> torch.Tensor:
    if len(elim_local) == 1:
        return pl_log_prob_ordered(logits, elim_local)
    samples = []
    for _ in range(ELIM_PERM_SAMPLES):
        perm = torch.randperm(len(elim_local))
        order = [elim_local[i] for i in perm.tolist()]
        samples.append(pl_log_prob_ordered(logits, order))
    return logmeanexp(torch.stack(samples))


def bottom_k_order_from_logits(logits: torch.Tensor, k: int) -> List[int]:
    return torch.argsort(logits, descending=True)[:k].tolist()


def log_prob_in_top2(logits: torch.Tensor, target: int) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=0)
    # P(target first)
    logp = log_probs[target]
    # Sum over j != target: P(j first) * P(target second | j first)
    for j in range(logits.numel()):
        if j == target:
            continue
        remaining = torch.tensor([k for k in range(logits.numel()) if k != j], device=logits.device)
        log_probs_after = torch.log_softmax(logits[remaining], dim=0)
        target_idx = torch.where(remaining == target)[0][0]
        logp = torch.logaddexp(
            logp,
            log_probs[j] + log_probs_after[target_idx],
        )
    return logp


@dataclass
class SeasonPack:
    season: int
    names: List[str]
    J: np.ndarray
    Jz: np.ndarray
    active_mask: np.ndarray
    elim_week: List[Optional[int]]
    age_z: np.ndarray
    pop_z: np.ndarray
    industry_idx: np.ndarray
    partner_idx: np.ndarray
    max_week: int


def load_popularity_scores(path: str) -> Dict[Tuple[int, str], float]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing popularity file: {path}")
    pop_df = pd.read_csv(path)
    if (
        "celebrity_name" not in pop_df.columns
        or "pop_score" not in pop_df.columns
        or "season" not in pop_df.columns
    ):
        raise ValueError("Popularity file missing required columns")
    return dict(
        zip(
            zip(pop_df["season"].astype(int), pop_df["celebrity_name"]),
            pop_df["pop_score"],
        )
    )


def build_dataset(df: pd.DataFrame) -> Tuple[List[SeasonPack], Dict]:
    week_cols = parse_week_cols(df)
    age = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce").to_numpy()
    age_mean = np.nanmean(age)
    age_std = np.nanstd(age)
    age_z_all = (age - age_mean) / (age_std + EPS)

    industry = df["celebrity_industry"].fillna("Unknown").tolist()
    industry_ids = {v: i for i, v in enumerate(sorted(set(industry)))}
    partner = df["ballroom_partner"].fillna("Unknown").tolist()
    partner_ids = {v: i for i, v in enumerate(sorted(set(partner)))}

    pop_scores = load_popularity_scores(POP_PATH)

    packs = []
    for season, df_season in df.groupby("season"):
        df_season = df_season.reset_index(drop=True)
        struct = build_season_struct(df_season, week_cols)
        J = struct["J"]
        Jz = compute_jz(J)
        active_mask = build_mask(J)
        age_z = age_z_all[df_season.index].copy()
        W, N = J.shape
        pop = np.zeros((W, N), dtype=float)
        for i, name in enumerate(df_season["celebrity_name"].tolist()):
            pop[:, i] = pop_scores.get((int(season), name), 0.0)
        pop_z = np.zeros_like(pop)
        for w in range(W):
            mask = active_mask[w] > 0
            if mask.sum() < 2:
                continue
            mean = pop[w, mask].mean()
            std = pop[w, mask].std()
            pop_z[w, mask] = (pop[w, mask] - mean) / (std + EPS)
        industry_idx = np.array([industry_ids[v] for v in df_season["celebrity_industry"].fillna("Unknown")])
        partner_idx = np.array([partner_ids[v] for v in df_season["ballroom_partner"].fillna("Unknown")])
        packs.append(
            SeasonPack(
                season=season,
                names=struct["names"],
                J=J,
                Jz=Jz,
                active_mask=active_mask,
                elim_week=struct["elim_week"],
                age_z=age_z,
                pop_z=pop_z,
                industry_idx=industry_idx,
                partner_idx=partner_idx,
                max_week=struct["max_week"],
            )
        )

    meta = {
        "industry_ids": industry_ids,
        "partner_ids": partner_ids,
        "max_week": max(p.max_week for p in packs),
        "week_cols": week_cols,
    }
    return packs, meta


def season_loss(
    pack: SeasonPack,
    p: torch.Tensor,
    alpha_s: torch.Tensor,
    beta_age: torch.Tensor,
    beta_pop: torch.Tensor,
    beta_industry: torch.Tensor,
    u_pro: torch.Tensor,
    eta_w: torch.Tensor,
    gamma: torch.Tensor,
    rho: torch.Tensor,
    tau: torch.Tensor,
    kappa: torch.Tensor,
    lam_judge: torch.Tensor,
) -> torch.Tensor:
    J = torch.tensor(pack.J, dtype=torch.float32, device=p.device)
    Jz = torch.tensor(pack.Jz, dtype=torch.float32, device=p.device)
    active_mask = torch.tensor(pack.active_mask, dtype=torch.float32, device=p.device)
    age_z = torch.tensor(pack.age_z, dtype=torch.float32, device=p.device)
    pop_z = torch.tensor(pack.pop_z, dtype=torch.float32, device=p.device)
    industry_idx = torch.tensor(pack.industry_idx, dtype=torch.long, device=p.device)
    partner_idx = torch.tensor(pack.partner_idx, dtype=torch.long, device=p.device)

    W, N = J.shape
    p_centered = center_p(p, active_mask)
    s = softmax_masked(p_centered, active_mask, tau)

    # Prior: latent evolution
    loss = torch.tensor(0.0, device=p.device)
    for w in range(W):
        shared = active_mask[w] > 0
        if not shared.any():
            continue
        eta = (
            alpha_s
            + beta_age * age_z
            + beta_pop * POP_WEIGHT * pop_z[w]
            + beta_industry[industry_idx]
            + u_pro[partner_idx]
            + eta_w[w]
            + gamma * Jz[w]
        )
        if w == 0:
            pred = eta
        else:
            pred = rho * p_centered[w - 1] + (1 - rho) * eta
        loss = loss + LAM_P * torch.mean((p_centered[w, shared] - pred[shared]) ** 2)
        if w > 0:
            shared_prev = (active_mask[w] > 0) & (active_mask[w - 1] > 0)
            if shared_prev.any():
                loss = loss + LAM_SMOOTH * torch.mean(
                    (p_centered[w, shared_prev] - p_centered[w - 1, shared_prev]) ** 2
                )

        if shared.any():
            s_w = s[w, shared]
            ent = -torch.sum(s_w * torch.log(s_w + EPS))
            loss = loss - LAM_ENT * ent
            loss = loss + LAM_PMAG * torch.mean(p_centered[w, shared] ** 2)

    # Likelihood
    regime = regime_for_season(pack.season)
    for w in range(W):
        idx = torch.where(active_mask[w] > 0)[0]
        if idx.numel() < 2:
            continue
        elim = [i for i, ew in enumerate(pack.elim_week) if ew == (w + 1) and i in idx.tolist()]
        k_elim = len(elim)
        if k_elim == 0:
            continue

        if regime == "percent":
            j_pct = J[w, idx] / (J[w, idx].sum() + EPS)
            C = j_pct + s[w, idx]
            logits = -C / kappa
            elim_local = [idx.tolist().index(e) for e in elim]
            loss = loss - observed_elim_log_prob(logits, elim_local)
            # Hard ordering: eliminated should have <= C than survivors
            for e_pos in elim_local:
                for j_pos in range(idx.numel()):
                    if j_pos in elim_local:
                        continue
                    loss = loss + LAM_HARD * torch.relu(C[e_pos] - C[j_pos])
        else:
            rJ = torch.from_numpy(
                pd.Series(J[w, idx].cpu().numpy())
                .rank(ascending=False, method="first")
                .to_numpy()
            ).to(p.device).float()
            rF = soft_rank_from_shares(s[w, idx])
            R = rJ + rF
            logits = R / kappa
            if regime == "rank":
                elim_local = [idx.tolist().index(e) for e in elim]
                loss = loss - observed_elim_log_prob(logits, elim_local)
                for e_pos in elim_local:
                    for j_pos in range(idx.numel()):
                        if j_pos in elim_local:
                            continue
                        loss = loss + LAM_HARD * torch.relu(R[j_pos] - R[e_pos])
            else:
                # Bottom-two + judge save
                if k_elim != 1:
                    elim_local = [idx.tolist().index(e) for e in elim]
                    loss = loss - observed_elim_log_prob(logits, elim_local)
                    for e_pos in elim_local:
                        for j_pos in range(idx.numel()):
                            if j_pos in elim_local:
                                continue
                            loss = loss + LAM_HARD * torch.relu(R[j_pos] - R[e_pos])
                    continue
                elim_idx = elim[0]
                elim_local = idx.tolist().index(elim_idx)
                # Probability elim in bottom two
                loss = loss - log_prob_in_top2(logits, elim_local)
                for j_pos in range(idx.numel()):
                    if j_pos == elim_local:
                        continue
                    loss = loss + LAM_HARD * torch.relu(R[j_pos] - R[elim_local])
                # Judge save marginalized over other bottom candidate
                jz = Jz[w, idx]
                judge_logit_full = -lam_judge * jz
                logits_other = logits.clone()
                logits_other[elim_local] = -1e9
                prob_other = torch.softmax(logits_other, dim=0)
                log_terms = []
                for j in range(logits.numel()):
                    if j == elim_local:
                        continue
                    pair_logits = torch.stack(
                        [judge_logit_full[elim_local], judge_logit_full[j]]
                    )
                    pair_log_probs = torch.log_softmax(pair_logits, dim=0)
                    log_terms.append(torch.log(prob_other[j] + EPS) + pair_log_probs[0])
                loss = loss - logmeanexp(torch.stack(log_terms))

    # Finals placement likelihood (weak): Plackett-Luce over season-sum C (finalists only)
    if LAM_FINAL > 0:
        elim = np.array(
            [ew if ew is not None else (W + 1) for ew in pack.elim_week], dtype=float
        )
        if np.isfinite(elim).sum() >= 2:
            season_score = torch.zeros(J.shape[1], device=p.device)
            for w in range(W):
                idx = torch.where(active_mask[w] > 0)[0]
                if idx.numel() < 2:
                    continue
                j_pct = J[w, idx] / (J[w, idx].sum() + EPS)
                C = j_pct + s[w, idx]
                season_score[idx] = season_score[idx] + C
            order = np.argsort(-elim).tolist()[:FINAL_TOP_K]
            loss = loss - LAM_FINAL * pl_log_prob_ordered(season_score, order)

    return loss


def main():
    df = pd.read_csv(DATA_PATH)
    packs, meta = build_dataset(df)
    device = torch.device("cpu")

    num_seasons = len(packs)
    max_week = meta["max_week"]
    num_industry = len(meta["industry_ids"])
    num_partner = len(meta["partner_ids"])

    # Global parameters
    alpha_s = torch.zeros(num_seasons, dtype=torch.float32, requires_grad=True, device=device)
    beta_age = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    beta_pop = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    beta_industry = torch.zeros(num_industry, dtype=torch.float32, requires_grad=True, device=device)
    u_pro = torch.zeros(num_partner, dtype=torch.float32, requires_grad=True, device=device)
    eta_w = torch.zeros(max_week, dtype=torch.float32, requires_grad=True, device=device)
    rho_raw = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    tau_raw = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    gamma = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    kappa_raw = torch.tensor(KAPPA_INIT, dtype=torch.float32, requires_grad=True, device=device)
    lam_judge_raw = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=device)

    # Per-season p
    p_seasons = []
    for pack in packs:
        W, N = pack.J.shape
        p = torch.zeros((W, N), dtype=torch.float32, requires_grad=True, device=device)
        p_seasons.append(p)

    params = [
        alpha_s,
        beta_age,
        beta_pop,
        beta_industry,
        u_pro,
        eta_w,
        rho_raw,
        tau_raw,
        gamma,
        kappa_raw,
        lam_judge_raw,
    ] + p_seasons

    opt = torch.optim.Adam(params, lr=LR)

    for _ in range(N_STEPS):
        opt.zero_grad()
        rho = torch.sigmoid(rho_raw)
        tau = torch.clamp(torch.nn.functional.softplus(tau_raw), 0.7, 1.5)
        kappa = torch.nn.functional.softplus(kappa_raw) + EPS
        lam_judge = torch.nn.functional.softplus(lam_judge_raw) + EPS
        loss = torch.tensor(0.0, device=device)
        for s_idx, pack in enumerate(packs):
            loss = loss + season_loss(
                pack,
                p_seasons[s_idx],
                alpha_s[s_idx],
                beta_age,
                beta_pop,
                beta_industry,
                u_pro,
                eta_w,
                gamma,
                rho,
                tau,
                kappa,
                lam_judge,
            )
        # Coefficient priors
        loss = loss + LAM_COEF * (
            beta_age**2
            + beta_pop**2
            + torch.mean(beta_industry**2)
            + torch.mean(u_pro**2)
            + torch.mean(alpha_s**2)
            + torch.mean(eta_w**2)
            + gamma**2
        )
        loss.backward()
        opt.step()

    # Outputs
    shares_rows = []
    unc_rows = []
    match_rows = []

    for s_idx, pack in enumerate(packs):
        rho = torch.sigmoid(rho_raw).detach()
        tau = torch.clamp(torch.nn.functional.softplus(tau_raw), 0.7, 1.5).detach()
        kappa = torch.nn.functional.softplus(kappa_raw).detach() + EPS
        lam_judge = torch.nn.functional.softplus(lam_judge_raw).detach() + EPS
        p = p_seasons[s_idx].detach()
        active_mask = torch.tensor(pack.active_mask, dtype=torch.float32, device=device)
        p_centered = center_p(p, active_mask)
        s_map = softmax_masked(p_centered, active_mask, tau).cpu().numpy()

        # SGLD for uncertainty (p only)
        samples = []
        if RUN_SGLD:
            p_sgld = p.clone().requires_grad_(True)
            for step in range(SGLD_STEPS):
                loss = season_loss(
                    pack,
                    p_sgld,
                    alpha_s[s_idx].detach(),
                    beta_age.detach(),
                    beta_pop.detach(),
                    beta_industry.detach(),
                    u_pro.detach(),
                    eta_w.detach(),
                    gamma.detach(),
                    rho,
                    tau,
                    kappa,
                    lam_judge,
                )
                grad = torch.autograd.grad(loss, p_sgld)[0]
                eps = torch.tensor(SGLD_STEP, device=p_sgld.device)
                noise = torch.randn_like(p_sgld)
                p_sgld = (
                    p_sgld - 0.5 * eps * grad + torch.sqrt(eps) * noise
                ).detach().requires_grad_(True)
                if step >= SGLD_BURNIN and (step - SGLD_BURNIN) % SGLD_INTERVAL == 0:
                    p_centered = center_p(p_sgld, active_mask)
                    s_sample = softmax_masked(p_centered, active_mask, tau).detach().cpu().numpy()
                    samples.append(s_sample)

        if samples:
            samples_arr = np.stack(samples, axis=0)
            s_mean = samples_arr.mean(axis=0)
            s_std = samples_arr.std(axis=0)
            s_p10 = np.percentile(samples_arr, 10, axis=0)
            s_p90 = np.percentile(samples_arr, 90, axis=0)
        else:
            s_mean = s_map
            s_std = np.zeros_like(s_map)
            s_p10 = s_map
            s_p90 = s_map

        # Match metrics
        hit = 0
        total = 0
        bottom2_hit = 0
        bottom2_total = 0
        margin_sum = 0.0
        margin_count = 0
        J = pack.J
        for w in range(pack.max_week):
            idx = np.where(J[w] > 0)[0]
            if len(idx) < 2:
                continue
            elim = [i for i, ew in enumerate(pack.elim_week) if ew == (w + 1) and i in idx]
            k_elim = len(elim)
            if k_elim == 0:
                continue
            total += 1
            if regime_for_season(pack.season) == "percent":
                j_pct = J[w, idx] / J[w, idx].sum()
                C = j_pct + s_map[w, idx]
                order = np.argsort(C)
                pred = set(idx[order[:k_elim]])
                if set(elim).issubset(pred):
                    hit += 1
                if len(order) > k_elim:
                    margin = C[order[k_elim]] - C[order[k_elim - 1]]
                    margin_sum += margin
                    margin_count += 1
            else:
                rJ = (
                    pd.Series(J[w, idx])
                    .rank(ascending=False, method="first")
                    .to_numpy()
                )
                rF = np.argsort(-s_map[w, idx]).argsort() + 1
                R = rJ + rF
                order = np.argsort(-R)
                pred = set(idx[order[:k_elim]])
                if set(elim).issubset(pred):
                    hit += 1
                if regime_for_season(pack.season) == "bottom" and k_elim == 1:
                    bottom2_total += 1
                    if elim[0] in idx[order[:2]]:
                        bottom2_hit += 1
                if len(order) > k_elim:
                    margin = R[order[k_elim - 1]] - R[order[k_elim]]
                    margin_sum += margin
                    margin_count += 1

        match_rows.append(
            {
                "season": pack.season,
                "regime": regime_for_season(pack.season),
                "hit_rate": hit / total if total else np.nan,
                "bottom2_rate": bottom2_hit / bottom2_total if bottom2_total else np.nan,
                "avg_margin": margin_sum / margin_count if margin_count else np.nan,
            }
        )

        for i, name in enumerate(pack.names):
            for w in range(pack.max_week):
                if pack.J[w, i] <= 0:
                    continue
                shares_rows.append(
                    {
                        "season": pack.season,
                        "week": w + 1,
                        "celebrity_name": name,
                        "s_map": s_map[w, i],
                    }
                )
                unc_rows.append(
                    {
                        "season": pack.season,
                        "week": w + 1,
                        "celebrity_name": name,
                        "s_mean": s_mean[w, i],
                        "s_std": s_std[w, i],
                        "s_p10": s_p10[w, i],
                        "s_p90": s_p90[w, i],
                    }
                )

    pd.DataFrame(shares_rows).to_csv(OUT_SHARES, index=False)
    pd.DataFrame(unc_rows).to_csv(OUT_UNC, index=False)
    pd.DataFrame(match_rows).to_csv(OUT_MATCH, index=False)
    print(f"Wrote {OUT_SHARES}, {OUT_UNC}, {OUT_MATCH}")


if __name__ == "__main__":
    main()
