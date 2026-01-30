import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install torch") from exc


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1_Bayes/inferred_shares_bayes.csv"
UNC_PATH = "AR-Problem1_Bayes/inferred_shares_bayes_unc.csv"
MATCH_PATH = "AR-Problem1_Bayes/elimination_match_bayes.csv"

EPS = 1e-8
TEMP = 0.7
TEMP_PLACEMENT = 0.6
LAM_DYN = 0.5
LAM_P = 0.1
LAM_HARD = 150.0
LAM_FINAL = 10.0
LAM_ENT = 0.05

LR = 0.05
N_STEPS = 250

RUN_SGLD = True
SGLD_STEPS = 80
SGLD_BURNIN = 20
SGLD_INTERVAL = 5


def parse_week_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def week_score(df: pd.DataFrame, week: int, cols: List[str]) -> pd.Series:
    wcols = [c for c in cols if c.startswith(f"week{week}_")]
    return df[wcols].sum(axis=1, skipna=True)


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
        if isinstance(row["results"], str) and "Eliminated Week" in row["results"]:
            elim_week.append(int(row["results"].split("Eliminated Week ")[1]))
        elif isinstance(row["results"], str) and "Withdrew" in row["results"]:
            elim_week.append(int(last_active[df_season.index.get_loc(i)]))
        else:
            elim_week.append(None)

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


def elimination_match_rate(shares, struct, regime):
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
    mask = (J > 0).astype(float)
    return mask


def softmax_masked(p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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
    a_raw: torch.Tensor,
    b_raw: torch.Tensor,
    gamma_raw: torch.Tensor,
    alpha_final_raw: torch.Tensor,
    beta_final_raw: torch.Tensor,
    regime: str,
) -> torch.Tensor:
    W, N = J.shape
    s = softmax_masked(p, active_mask)
    loss = torch.tensor(0.0, device=p.device)

    gamma = torch.tanh(gamma_raw)
    for w in range(1, W):
        shared = (active_mask[w] > 0) & (active_mask[w - 1] > 0)
        if shared.any():
            pred = p[w - 1] + gamma * Jz[w]
            loss = loss + LAM_DYN * torch.mean((p[w, shared] - pred[shared]) ** 2)

    # L2 prior on latent states to prevent extreme logits
    loss = loss + LAM_P * torch.mean((p * active_mask) ** 2)

    # Likelihood (soft elimination) + hard-constraint penalties
    for w in range(W):
        idx = torch.where(active_mask[w] > 0)[0]
        if idx.numel() < 2:
            continue
        elim = [i for i, ew in enumerate(elim_week) if ew == (w + 1) and i in idx.tolist()]
        if not elim:
            continue
        if regime == "percent":
            q = J[w, idx] / (J[w, idx].sum() + EPS)
        else:
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

        # Entropy regularization (keeps shares plausible, not extreme)
        loss = loss - LAM_ENT * torch.sum(s[w, idx] * torch.log(s[w, idx] + EPS))

        # Hard-constraint penalty: eliminated should be among worst risk.
        elim_pos = [idx.tolist().index(e) for e in elim]
        non_elim_pos = [i for i in range(idx.numel()) if i not in elim_pos]
        for e_pos in elim_pos:
            for j_pos in non_elim_pos:
                loss = loss + LAM_HARD * torch.relu(risk[j_pos] - risk[e_pos])

        # Finals placement likelihood (only on last active week)
    if LAM_FINAL > 0 and placement_rank.numel() == J.shape[1]:
        pr = placement_rank
        valid = torch.where(torch.isfinite(pr))[0]
        if valid.numel() >= 2:
            alpha_final = torch.nn.functional.softplus(alpha_final_raw)
            beta_final = torch.nn.functional.softplus(beta_final_raw)
            S = torch.zeros_like(pr)
            for w in range(W):
                idx = torch.where(active_mask[w] > 0)[0]
                if idx.numel() < 2:
                    continue
                if regime == "percent":
                    q = J[w, idx] / (J[w, idx].sum() + EPS)
                else:
                    rJ = torch.from_numpy(
                        pd.Series(J[w, idx].cpu().numpy())
                        .rank(ascending=False, method="first")
                        .to_numpy()
                    ).to(p.device).float()
                    q = (rJ - 1) / (rJ.numel() - 1 + EPS)
                S[idx] = S[idx] + alpha_final * (1 - q) + beta_final * torch.log(
                    s[w, idx] + EPS
                )
            order = valid[torch.argsort(pr[valid], descending=False)]
            remaining = valid.clone()
            for pos in order:
                logits = S[remaining] / TEMP_PLACEMENT
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
    return loss


def fit_season(df_season, week_cols, regime, device):
    struct = build_season_struct(df_season, week_cols)
    J = torch.tensor(struct["J"], dtype=torch.float32, device=device)
    active_mask = torch.tensor(build_mask(struct["J"]), dtype=torch.float32, device=device)
    W, N = J.shape

    Jz = torch.tensor(compute_jz(struct["J"]), dtype=torch.float32, device=device)
    placement_rank = (
        pd.to_numeric(df_season["placement"], errors="coerce")
        .rank(ascending=True, method="average")
        .to_numpy()
    )
    placement_rank = torch.tensor(placement_rank, dtype=torch.float32, device=device)

    p = torch.zeros((W, N), dtype=torch.float32, requires_grad=True, device=device)
    a_raw = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    b_raw = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    gamma_raw = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    alpha_final_raw = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)
    beta_final_raw = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device=device)

    opt = torch.optim.Adam([p, a_raw, b_raw, gamma_raw, alpha_final_raw, beta_final_raw], lr=LR)

    for _ in range(N_STEPS):
        opt.zero_grad()
        loss = loss_for_season(
            p,
            J,
            struct["elim_week"],
            active_mask,
            Jz,
            placement_rank,
            a_raw,
            b_raw,
            gamma_raw,
            alpha_final_raw,
            beta_final_raw,
            regime,
        )
        loss.backward()
        opt.step()

    with torch.no_grad():
        s_map = softmax_masked(p, active_mask).cpu().numpy()

    # SGLD samples for uncertainty
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
                a_raw.detach(),
                b_raw.detach(),
                gamma_raw.detach(),
                alpha_final_raw.detach(),
                beta_final_raw.detach(),
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
