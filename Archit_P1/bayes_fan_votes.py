"""
Bayesian fan-vote inference for Dancing With The Stars (percent-combination era).

You observe:
- weekly judges totals per contestant (already aggregated across judges),
- which contestant was eliminated at the end of each (season, week).

You do NOT observe fan votes. We infer latent fan vote shares via a softmax utility:

    eta_{i,t} = beta0
              + beta_age   * age_z
              + beta_fame  * log_wiki_edits_z
              + beta_judge * J_{i,t,z}
              + (optional) beta_prev_judge * prev_judge_total_z
              + (optional) beta_delta      * judge_delta_z
              + industry_effect[industry]
              + partner_effect[partner]
              + b_contestant[celebrity_id]

    F_{i,t} = softmax(eta_{.,t})  over alive contestants in week t
    C_{i,t} = J_{i,t} + F_{i,t}

Elimination is modeled as a "soft" argmin on combined score:

    P(elim=i | t) = softmax(alpha * (-C_{.,t}))

Likelihood: for each week with exactly one elimination, we add:

    log P(eliminated contestant | week t)

Implementation notes
--------------------
- No file I/O: this module expects `df_long` is already loaded by you.
- Restrict to seasons 3–27 for fitting/evaluation (percent-combination era).
- Drop weeks with no elimination or multiple eliminations (per prompt).
- Identifiability:
  - continuous covariates are z-scored using training stats
  - industry/partner effects are sum-to-zero centered (effect coding)

Dependencies: numpy, pandas, pymc, pytensor, arviz
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import arviz as az  # type: ignore
    import pymc as pm  # type: ignore
    import pytensor.tensor as pt  # type: ignore
except ImportError:  # pragma: no cover
    # Allow importing and using data-prep utilities (e.g., wide->long conversion)
    # even when Bayesian dependencies are not installed in the runtime.
    az = None  # type: ignore[assignment]
    pm = None  # type: ignore[assignment]
    pt = None  # type: ignore[assignment]


# -----------------------------
# Wide (problem CSV) -> long (df_long) conversion
# -----------------------------

_WEEK_SCORE_RE = r"^week(?P<week>\d+)_judge(?P<judge>\d+)_score$"


def make_df_long_from_problem_c(
    df_wide: pd.DataFrame,
    *,
    treat_withdraw_as_elimination: bool = False,
    default_log_wiki_edits: float = 0.0,
    log_wiki_edits_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convert the provided problem dataset `Data/2026_MCM_Problem_C_Data.csv` (wide format)
    into the `df_long` format expected by this module.

    Input (wide) expected columns:
    - celebrity_name, ballroom_partner, celebrity_industry, celebrity_age_during_season, season, results
    - week{w}_judge{j}_score columns (strings, numbers, or 'N/A')

    Output (long) columns:
    - season (int), week (int), celebrity_id (str), alive (bool)
    - judge_total (float), eliminated (bool)
    - age (float), industry (str), partner (str), log_wiki_edits (float)
    - prev_judge_total (float), judge_delta (float)

    Notes / assumptions:
    - We infer the set of weeks from the `week*_judge*_score` columns present.
    - A contestant is considered "active" through their last week with a positive judge_total.
      (The raw CSV uses 0s after elimination for many rows.)
    - If `results` is "Eliminated Week k", we mark eliminated=True at week k.
    - If `results` is "Withdrew", we either:
        - mark no elimination (default), or
        - treat it as eliminated at last active week if treat_withdraw_as_elimination=True.
    - `log_wiki_edits` is not in the CSV, so we fill it with `default_log_wiki_edits`
      unless you provide `log_wiki_edits_col` in df_wide.
    """
    df = df_wide.copy()

    # normalize column naming
    if "celebrity_homecountry/region" in df.columns and "celebrity_homecountry" not in df.columns:
        df = df.rename(columns={"celebrity_homecountry/region": "celebrity_homecountry"})

    needed = {"celebrity_name", "ballroom_partner", "celebrity_industry", "celebrity_age_during_season", "season", "results"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"df_wide missing required columns: {sorted(missing)}")

    # Discover week/judge score columns
    score_cols: list[str] = []
    week_numbers: set[int] = set()
    for c in df.columns:
        m = pd.Series([c]).str.match(_WEEK_SCORE_RE).iloc[0]
        if m:
            # This match approach doesn't capture groups; do regex with re for groups:
            pass
    import re

    for c in df.columns:
        m = re.match(_WEEK_SCORE_RE, c)
        if not m:
            continue
        score_cols.append(c)
        week_numbers.add(int(m.group("week")))

    if not score_cols:
        raise ValueError("No week/judge score columns found (expected week{w}_judge{j}_score).")

    max_week = max(week_numbers)

    # Coerce score columns to numeric (treat N/A/blank as NaN)
    df_scores = df[score_cols].replace({"N/A": np.nan, "": np.nan})
    for c in score_cols:
        df_scores[c] = pd.to_numeric(df_scores[c], errors="coerce")

    # Compute judge_total_w and has_any_score_w per row
    judge_total_by_week: Dict[int, np.ndarray] = {}
    has_any_score_by_week: Dict[int, np.ndarray] = {}
    for w in range(1, max_week + 1):
        cols_w = [c for c in score_cols if c.startswith(f"week{w}_")]
        if not cols_w:
            judge_total_by_week[w] = np.full(len(df), np.nan)
            has_any_score_by_week[w] = np.zeros(len(df), dtype=bool)
            continue
        vals = df_scores[cols_w].to_numpy(dtype=float)
        has_any = np.any(~np.isnan(vals), axis=1)
        tot = np.nansum(vals, axis=1)
        # If all judges NaN, treat total as NaN (not 0)
        tot = np.where(has_any, tot, np.nan)
        judge_total_by_week[w] = tot
        has_any_score_by_week[w] = has_any

    # Determine last active week per contestant-season row (last week with judge_total > 0)
    last_active = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        lp = 0
        for w in range(1, max_week + 1):
            jt = judge_total_by_week[w][i]
            if np.isfinite(jt) and jt > 0:
                lp = w
        last_active[i] = lp

    # Parse elimination week from results
    def parse_elim_week(res: Any) -> Optional[int]:
        if res is None or (isinstance(res, float) and np.isnan(res)):
            return None
        s = str(res)
        m = re.search(r"Eliminated\s+Week\s+(\d+)", s)
        if m:
            return int(m.group(1))
        if "Withdrew" in s:
            return None
        return None

    elim_week = np.array([parse_elim_week(r) or 0 for r in df["results"]], dtype=int)
    withdrew = df["results"].astype(str).str.contains("Withdrew", na=False).to_numpy()
    if treat_withdraw_as_elimination:
        # treat withdraw as elimination at last active week (if any)
        elim_week = np.where(withdrew & (last_active > 0), last_active, elim_week)

    # Build df_long rows
    rows: list[dict[str, Any]] = []
    for idx, r in df.iterrows():
        season = int(pd.to_numeric(r["season"], errors="raise"))
        name = str(r["celebrity_name"]).strip()
        partner = str(r["ballroom_partner"]).strip()
        industry = str(r["celebrity_industry"]).strip()
        age = float(pd.to_numeric(r["celebrity_age_during_season"], errors="coerce"))

        # Make a stable per-season contestant id (same celeb can appear multiple seasons)
        celeb_id = f"{season}:{name}"

        la = int(last_active[idx])
        if la <= 0:
            # no usable weekly scores; skip
            continue

        ew = int(elim_week[idx])
        is_withdrew = bool(withdrew[idx])
        for w in range(1, la + 1):
            jt = float(judge_total_by_week[w][idx]) if np.isfinite(judge_total_by_week[w][idx]) else np.nan
            has_any = bool(has_any_score_by_week[w][idx])
            if not has_any:
                # If no judge scores at all for this week, skip the row
                continue

            alive = True  # in the long format: row exists only while alive

            eliminated = False
            if ew > 0 and w == ew and (("Eliminated Week" in str(r["results"])) or (treat_withdraw_as_elimination and is_withdrew)):
                eliminated = True

            # log_wiki_edits (fame proxy) placeholder
            if log_wiki_edits_col is not None and log_wiki_edits_col in df.columns:
                lwe = float(pd.to_numeric(r[log_wiki_edits_col], errors="coerce"))
            else:
                lwe = float(default_log_wiki_edits)

            rows.append(
                {
                    "season": season,
                    "week": int(w),
                    "celebrity_id": celeb_id,
                    "alive": alive,
                    "judge_total": jt,
                    "eliminated": eliminated,
                    "age": age,
                    "industry": industry,
                    "partner": partner,
                    "log_wiki_edits": lwe,
                    # keep raw identifiers for debugging/merging
                    "celebrity_name": name,
                    "ballroom_partner": partner,
                }
            )

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        raise ValueError("Conversion produced empty df_long. Check score columns / parsing assumptions.")

    # Compute prev_judge_total and judge_delta within each (season, celebrity_id)
    df_long = df_long.sort_values(["season", "celebrity_id", "week"], kind="stable").reset_index(drop=True)
    df_long["prev_judge_total"] = df_long.groupby(["season", "celebrity_id"], sort=False)["judge_total"].shift(1)
    df_long["judge_delta"] = df_long["judge_total"] - df_long["prev_judge_total"]

    # Sanity: alive is True for all rows by construction
    df_long["alive"] = True
    return df_long


# -----------------------------
# Pytensor helper: segment softmax
# -----------------------------


def _segment_softmax(logits: pt.TensorVariable, group_idx: pt.TensorVariable) -> pt.TensorVariable:
    """
    Stable softmax computed separately within each group.

    Parameters
    ----------
    logits:
        Shape (N,) vector of logits for each row.
    group_idx:
        Shape (N,) int vector in [0, G-1] assigning each row to a group.
    """
    if pt is None:  # pragma: no cover
        raise ImportError("pytensor is required for building/fitting the Bayesian model. Install `pytensor` + `pymc`.")
    seg_max = pt.extra_ops.segment_max(logits, group_idx)
    centered = logits - seg_max[group_idx]
    expc = pt.exp(centered)
    seg_sum = pt.extra_ops.segment_sum(expc, group_idx)
    return expc / (seg_sum[group_idx] + 1e-12)


# -----------------------------
# Preprocessing
# -----------------------------


@dataclass(frozen=True)
class PreprocessArtifacts:
    """Training-time mappings/stats used to transform test/new data."""

    season_min: int
    season_max: int

    cont_means: Dict[str, float]
    cont_stds: Dict[str, float]

    industry_categories: Tuple[str, ...]
    partner_categories: Tuple[str, ...]
    contestant_categories: Tuple[str, ...]

    use_prev_judge_total: bool
    use_judge_delta: bool


def prepare_week_groups(
    df_long: pd.DataFrame,
    *,
    seasons: Tuple[int, int] = (3, 27),
    drop_multi_elim_weeks: bool = True,
    drop_no_elim_weeks: bool = True,
    use_prev_judge_total: bool = True,
    use_judge_delta: bool = False,
    center_scale_continuous: bool = True,
    train_artifacts: Optional[PreprocessArtifacts] = None,
    handle_unknown: Literal["error", "use_unk"] = "error",
) -> Tuple[pd.DataFrame, PreprocessArtifacts]:
    """
    Prepare `df_long` for modeling.

    Produces:
    - `group_id` for each (season, week)
    - `judge_pct` = judge_total / sum(judge_total over alive in that week)
    - z-scored covariates using training stats (or fit them if train_artifacts=None)
    - integer indices: industry_idx, partner_idx, contestant_idx
    - df.attrs['elim_row_by_group']: for each group, the global row index of eliminated contestant
    """
    required = {
        "season",
        "week",
        "celebrity_id",
        "alive",
        "judge_total",
        "eliminated",
        "age",
        "industry",
        "partner",
        "log_wiki_edits",
    }
    missing = required - set(df_long.columns)
    if missing:
        raise ValueError(f"df_long is missing required columns: {sorted(missing)}")

    df = df_long.copy()
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["week"] = pd.to_numeric(df["week"], errors="raise").astype(int)
    df["alive"] = df["alive"].astype(bool)
    df["eliminated"] = df["eliminated"].astype(bool)

    season_min, season_max = seasons
    df = df[(df["season"] >= season_min) & (df["season"] <= season_max)].copy()
    df = df[df["alive"]].copy()

    # Build group_id
    df["group_key"] = list(zip(df["season"].tolist(), df["week"].tolist()))
    keys = pd.Index(df["group_key"].unique())
    group_map = {k: i for i, k in enumerate(keys)}
    df["group_id"] = df["group_key"].map(group_map).astype(int)

    # Judges percent within week (alive only)
    df["judge_total"] = pd.to_numeric(df["judge_total"], errors="coerce")
    group_sum = df.groupby("group_id")["judge_total"].transform("sum")
    df["judge_pct"] = np.where(group_sum > 0, df["judge_total"] / group_sum, np.nan)

    # Drop weeks by elimination count (per prompt)
    elim_count = df.groupby("group_id")["eliminated"].transform("sum")
    if drop_no_elim_weeks:
        df = df[elim_count >= 1].copy()
        elim_count = df.groupby("group_id")["eliminated"].transform("sum")
    if drop_multi_elim_weeks:
        df = df[elim_count == 1].copy()

    # Recompute compact group_id after dropping
    df["group_key"] = list(zip(df["season"].tolist(), df["week"].tolist()))
    keys = pd.Index(df["group_key"].unique())
    group_map = {k: i for i, k in enumerate(keys)}
    df["group_id"] = df["group_key"].map(group_map).astype(int)

    # If we're in prediction mode (not dropping), we allow non-single-elim weeks.
    if drop_multi_elim_weeks or drop_no_elim_weeks:
        counts = df.groupby("group_id")["eliminated"].sum()
        bad = counts[counts != 1]
        if len(bad):
            raise ValueError(f"After filtering, some groups don't have exactly one eliminated row: {bad.head(10).to_dict()}")

    df = df.reset_index(drop=True)

    # Build elim_row_by_group (only meaningful when we kept single-elim weeks)
    elim_row_by_group = np.full(df["group_id"].nunique(), -1, dtype=int)
    if df["eliminated"].any():
        elim_rows = df.index[df["eliminated"]].to_numpy()
        elim_groups = df.loc[df["eliminated"], "group_id"].to_numpy()
        elim_row_by_group[elim_groups] = elim_rows
    df.attrs["elim_row_by_group"] = elim_row_by_group

    # Continuous covariates
    cont_cols = ["age", "log_wiki_edits", "judge_pct"]
    if use_prev_judge_total:
        if "prev_judge_total" not in df.columns:
            raise ValueError("use_prev_judge_total=True but df_long has no 'prev_judge_total' column")
        cont_cols.append("prev_judge_total")
    if use_judge_delta:
        if "judge_delta" not in df.columns:
            raise ValueError("use_judge_delta=True but df_long has no 'judge_delta' column")
        cont_cols.append("judge_delta")

    for c in cont_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Category mappings
    def _cat(series: pd.Series, cats: Optional[Tuple[str, ...]], name: str) -> Tuple[pd.Categorical, Tuple[str, ...]]:
        ser = series.astype(str)
        if cats is None:
            cat = pd.Categorical(ser)
            return cat, tuple(cat.categories.astype(str).tolist())
        if handle_unknown == "use_unk":
            cats_list = list(cats)
            if "__UNK__" not in cats_list:
                cats_list.append("__UNK__")
            unknown = ~ser.isin(cats)
            ser = ser.where(~unknown, "__UNK__")
            cat = pd.Categorical(ser, categories=cats_list)
            return cat, tuple(cats_list)
        unknown = sorted(set(ser.unique()) - set(cats))
        if unknown:
            raise ValueError(f"Unknown {name} categories (not in training): {unknown[:10]}")
        cat = pd.Categorical(ser, categories=list(cats))
        return cat, cats

    if train_artifacts is None:
        industry_cat, industry_categories = _cat(df["industry"], None, "industry")
        partner_cat, partner_categories = _cat(df["partner"], None, "partner")
        contestant_cat, contestant_categories = _cat(df["celebrity_id"], None, "celebrity_id")
    else:
        industry_cat, industry_categories = _cat(df["industry"], train_artifacts.industry_categories, "industry")
        partner_cat, partner_categories = _cat(df["partner"], train_artifacts.partner_categories, "partner")
        contestant_cat, contestant_categories = _cat(df["celebrity_id"], train_artifacts.contestant_categories, "celebrity_id")

    df["industry_idx"] = industry_cat.codes.astype(int)
    df["partner_idx"] = partner_cat.codes.astype(int)
    df["contestant_idx"] = contestant_cat.codes.astype(int)

    # z-scoring stats (fit on train, apply on test/new)
    if train_artifacts is None:
        cont_means: Dict[str, float] = {}
        cont_stds: Dict[str, float] = {}
        for c in cont_cols:
            mu = float(np.nanmean(df[c].to_numpy()))
            sd = float(np.nanstd(df[c].to_numpy()))
            cont_means[c] = mu
            cont_stds[c] = sd if sd > 1e-8 else 1.0
    else:
        cont_means = dict(train_artifacts.cont_means)
        cont_stds = dict(train_artifacts.cont_stds)

    if center_scale_continuous:
        for c in cont_cols:
            df[f"{c}_z"] = (df[c] - cont_means[c]) / cont_stds[c]
    else:
        for c in cont_cols:
            df[f"{c}_z"] = df[c]

    artifacts = PreprocessArtifacts(
        season_min=season_min,
        season_max=season_max,
        cont_means=cont_means,
        cont_stds=cont_stds,
        industry_categories=industry_categories,
        partner_categories=partner_categories,
        contestant_categories=contestant_categories,
        use_prev_judge_total=use_prev_judge_total,
        use_judge_delta=use_judge_delta,
    )
    df.attrs["artifacts"] = artifacts
    return df, artifacts


# -----------------------------
# Model
# -----------------------------


def build_model(
    df_train: pd.DataFrame,
    artifacts: Optional[PreprocessArtifacts] = None,
    *,
    prior_sd_beta: float = 1.5,
    prior_sd_cat: float = 1.0,
    prior_sd_alpha: float = 2.0,
) -> pm.Model:
    if pm is None or pt is None:  # pragma: no cover
        raise ImportError("PyMC model building requires `pymc` and `pytensor`. Install them in your environment.")
    """
    Build the PyMC model. `df_train` must be the output of `prepare_week_groups()`.

    Likelihood is implemented via a Potential using `elim_row_by_group`.
    """
    if artifacts is None:
        artifacts = df_train.attrs.get("artifacts")
    if artifacts is None:
        raise ValueError("artifacts not provided and not found in df_train.attrs['artifacts']")

    group_idx = df_train["group_id"].to_numpy(np.int32)
    industry_idx = df_train["industry_idx"].to_numpy(np.int32)
    partner_idx = df_train["partner_idx"].to_numpy(np.int32)
    contestant_idx = df_train["contestant_idx"].to_numpy(np.int32)

    age_z = df_train["age_z"].to_numpy(float)
    fame_z = df_train["log_wiki_edits_z"].to_numpy(float)
    judge_z = df_train["judge_pct_z"].to_numpy(float)
    judge_pct = df_train["judge_pct"].to_numpy(float)

    use_prev = artifacts.use_prev_judge_total
    use_delta = artifacts.use_judge_delta
    prev_z = df_train["prev_judge_total_z"].to_numpy(float) if use_prev else None
    delta_z = df_train["judge_delta_z"].to_numpy(float) if use_delta else None

    elim_row_by_group = np.asarray(df_train.attrs["elim_row_by_group"], dtype=np.int32)
    # enforce "single-elim weeks" for training
    if (elim_row_by_group < 0).any():
        raise ValueError(
            "Training data has groups without exactly one eliminated row. "
            "Call prepare_week_groups(..., drop_no_elim_weeks=True, drop_multi_elim_weeks=True)."
        )

    K_industry = len(artifacts.industry_categories)
    K_partner = len(artifacts.partner_categories)
    K_contestant = len(artifacts.contestant_categories)

    with pm.Model() as model:
        # Data containers (so model can be re-used with pm.set_data if you want)
        pm.Data("group_idx", group_idx, mutable=True)
        pm.Data("industry_idx", industry_idx, mutable=True)
        pm.Data("partner_idx", partner_idx, mutable=True)
        pm.Data("contestant_idx", contestant_idx, mutable=True)

        pm.Data("age_z", age_z, mutable=True)
        pm.Data("fame_z", fame_z, mutable=True)
        pm.Data("judge_z", judge_z, mutable=True)
        pm.Data("judge_pct", judge_pct, mutable=True)
        if use_prev:
            pm.Data("prev_z", prev_z, mutable=True)
        if use_delta:
            pm.Data("delta_z", delta_z, mutable=True)

        pm.Data("elim_row_by_group", elim_row_by_group, mutable=True)

        # Priors
        beta0 = pm.Normal("beta0", 0.0, prior_sd_beta)
        beta_age = pm.Normal("beta_age", 0.0, prior_sd_beta)
        beta_fame = pm.Normal("beta_fame", 0.0, prior_sd_beta)
        beta_judge = pm.Normal("beta_judge", 0.0, prior_sd_beta)

        if use_prev:
            beta_prev_judge = pm.Normal("beta_prev_judge", 0.0, prior_sd_beta)
        if use_delta:
            beta_delta = pm.Normal("beta_delta", 0.0, prior_sd_beta)

        # Sum-to-zero category effects
        industry_raw = pm.Normal("industry_raw", 0.0, prior_sd_cat, shape=K_industry)
        partner_raw = pm.Normal("partner_raw", 0.0, prior_sd_cat, shape=K_partner)
        industry_effect = pm.Deterministic("industry_effect", industry_raw - pt.mean(industry_raw))
        partner_effect = pm.Deterministic("partner_effect", partner_raw - pt.mean(partner_raw))

        # Contestant random effects
        sigma_b = pm.HalfNormal("sigma_b", 1.0)
        b_contestant = pm.Normal("b_contestant", 0.0, sigma_b, shape=K_contestant)

        alpha = pm.HalfNormal("alpha", prior_sd_alpha)

        g = pm.Data("group_idx")
        ind = pm.Data("industry_idx")
        par = pm.Data("partner_idx")
        con = pm.Data("contestant_idx")

        eta = (
            beta0
            + beta_age * pm.Data("age_z")
            + beta_fame * pm.Data("fame_z")
            + beta_judge * pm.Data("judge_z")
            + industry_effect[ind]
            + partner_effect[par]
            + b_contestant[con]
        )
        if use_prev:
            eta = eta + beta_prev_judge * pm.Data("prev_z")
        if use_delta:
            eta = eta + beta_delta * pm.Data("delta_z")

        fan_share = pm.Deterministic("fan_share", _segment_softmax(eta, g))
        combined_score = pm.Deterministic("combined_score", pm.Data("judge_pct") + fan_share)
        elim_prob = pm.Deterministic("elim_prob", _segment_softmax(-alpha * combined_score, g))

        elim_rows = pm.Data("elim_row_by_group")
        pm.Potential("elim_loglike", pt.sum(pt.log(elim_prob[elim_rows] + 1e-12)))

    return model


def fit_model(
    model: pm.Model,
    *,
    draws: int = 1500,
    tune: int = 1500,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 123,
) -> az.InferenceData:
    """Fit with NUTS and return ArviZ InferenceData."""
    if pm is None or az is None:  # pragma: no cover
        raise ImportError("Model fitting requires `pymc` and `arviz`. Install them in your environment.")
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
        )
    return idata


# -----------------------------
# Posterior summaries & vote-share prediction
# -----------------------------


def posterior_summaries(
    idata: az.InferenceData,
    *,
    var_names: Optional[Iterable[str]] = None,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    if az is None:  # pragma: no cover
        raise ImportError("Posterior summarization requires `arviz`. Install it in your environment.")
    """
    Posterior summaries for weights and alpha.

    Returns a DataFrame like arviz.summary().
    """
    if var_names is None:
        var_names = ["beta0", "beta_age", "beta_fame", "beta_judge", "alpha", "sigma_b"]
        if "beta_prev_judge" in idata.posterior:
            var_names = list(var_names) + ["beta_prev_judge"]
        if "beta_delta" in idata.posterior:
            var_names = list(var_names) + ["beta_delta"]
    return az.summary(idata, var_names=list(var_names), hdi_prob=hdi_prob)


def _extract_draws(idata: az.InferenceData) -> Dict[str, np.ndarray]:
    """Flatten chain/draw into a single samples dim."""
    post = idata.posterior

    def flat(name: str) -> np.ndarray:
        x = post[name].values
        return x.reshape((-1,) + x.shape[2:])

    out: Dict[str, np.ndarray] = {k: flat(k) for k in ["beta0", "beta_age", "beta_fame", "beta_judge", "alpha", "sigma_b"] if k in post}
    for k in ["beta_prev_judge", "beta_delta", "industry_effect", "partner_effect", "b_contestant"]:
        if k in post:
            out[k] = flat(k)
    return out


def _softmax_by_group_numpy(x: np.ndarray, group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """Stable numpy softmax within groups."""
    out = np.empty_like(x)
    max_g = np.full(n_groups, -np.inf)
    np.maximum.at(max_g, group_idx, x)
    expc = np.exp(x - max_g[group_idx])
    sum_g = np.zeros(n_groups)
    np.add.at(sum_g, group_idx, expc)
    out[:] = expc / (sum_g[group_idx] + 1e-12)
    return out


def posterior_vote_shares(
    idata: az.InferenceData,
    df_prep: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    *,
    hdi_prob: float = 0.95,
    max_posterior_samples: int = 2000,
) -> pd.DataFrame:
    if az is None:  # pragma: no cover
        raise ImportError("Vote-share posterior intervals require `arviz`. Install it in your environment.")
    """
    Posterior distribution of fan vote share F_{i,t} for each row in df_prep.

    Returns row-aligned mean and HDI.
    """
    draws = _extract_draws(idata)
    S = draws["beta0"].shape[0]
    if S > max_posterior_samples:
        sample_idx = np.random.default_rng(0).choice(S, size=max_posterior_samples, replace=False)
    else:
        sample_idx = np.arange(S)

    group_idx = df_prep["group_id"].to_numpy(int)
    n_groups = int(df_prep["group_id"].nunique())
    ind = df_prep["industry_idx"].to_numpy(int)
    par = df_prep["partner_idx"].to_numpy(int)
    con = df_prep["contestant_idx"].to_numpy(int)
    age_z = df_prep["age_z"].to_numpy(float)
    fame_z = df_prep["log_wiki_edits_z"].to_numpy(float)
    judge_z = df_prep["judge_pct_z"].to_numpy(float)

    prev_z = df_prep["prev_judge_total_z"].to_numpy(float) if artifacts.use_prev_judge_total else None
    delta_z = df_prep["judge_delta_z"].to_numpy(float) if artifacts.use_judge_delta else None

    F_samples = np.empty((len(sample_idx), len(df_prep)), dtype=float)

    for si, s in enumerate(sample_idx):
        eta = (
            draws["beta0"][s]
            + draws["beta_age"][s] * age_z
            + draws["beta_fame"][s] * fame_z
            + draws["beta_judge"][s] * judge_z
            + draws["industry_effect"][s][ind]
            + draws["partner_effect"][s][par]
            + draws["b_contestant"][s][con]
        )
        if artifacts.use_prev_judge_total:
            eta = eta + draws["beta_prev_judge"][s] * prev_z
        if artifacts.use_judge_delta:
            eta = eta + draws["beta_delta"][s] * delta_z
        F_samples[si] = _softmax_by_group_numpy(eta, group_idx, n_groups)

    mean = F_samples.mean(axis=0)
    hdi = az.hdi(F_samples, hdi_prob=hdi_prob)

    out = df_prep[["season", "week", "celebrity_id", "judge_total", "judge_pct", "eliminated"]].copy()
    out["fan_share_mean"] = mean
    out["fan_share_hdi_low"] = hdi[:, 0]
    out["fan_share_hdi_high"] = hdi[:, 1]
    return out


def predict_vote_shares(
    idata: az.InferenceData,
    df_new_long: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    *,
    hdi_prob: float = 0.95,
    handle_unknown: Literal["error", "use_unk"] = "error",
) -> pd.DataFrame:
    """
    Posterior vote-share intervals for new week data.

    Input does not need `eliminated`; it will be created as False if missing.
    """
    df_new = df_new_long.copy()
    if "eliminated" not in df_new.columns:
        df_new["eliminated"] = False
    if "alive" not in df_new.columns:
        df_new["alive"] = True

    df_prep, _ = prepare_week_groups(
        df_new,
        seasons=(artifacts.season_min, artifacts.season_max),
        drop_multi_elim_weeks=False,
        drop_no_elim_weeks=False,
        use_prev_judge_total=artifacts.use_prev_judge_total,
        use_judge_delta=artifacts.use_judge_delta,
        train_artifacts=artifacts,
        handle_unknown=handle_unknown,
    )
    return posterior_vote_shares(idata, df_prep, artifacts, hdi_prob=hdi_prob)


# -----------------------------
# Evaluation
# -----------------------------


def evaluate(
    model: Any,
    idata: az.InferenceData,
    df_test_long: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    *,
    handle_unknown: Literal["error", "use_unk"] = "error",
    max_posterior_samples: int = 2000,
) -> Dict[str, Any]:
    if az is None:  # pragma: no cover
        raise ImportError("Evaluation requires `arviz` (and typically PyMC). Install it in your environment.")
    """
    Held-out season evaluation (train/test split outside this function).

    Metrics:
    - top-1 accuracy: argmax P(elim) matches true eliminated
    - average rank of true eliminated under mean posterior risk distribution
    """
    _ = model  # kept for API symmetry per prompt

    df_test_prep, _ = prepare_week_groups(
        df_test_long,
        seasons=(artifacts.season_min, artifacts.season_max),
        drop_multi_elim_weeks=True,
        drop_no_elim_weeks=True,
        use_prev_judge_total=artifacts.use_prev_judge_total,
        use_judge_delta=artifacts.use_judge_delta,
        train_artifacts=artifacts,
        handle_unknown=handle_unknown,
    )

    draws = _extract_draws(idata)
    S = draws["beta0"].shape[0]
    if S > max_posterior_samples:
        sample_idx = np.random.default_rng(1).choice(S, size=max_posterior_samples, replace=False)
    else:
        sample_idx = np.arange(S)

    group_idx = df_test_prep["group_id"].to_numpy(int)
    n_groups = int(df_test_prep["group_id"].nunique())
    ind = df_test_prep["industry_idx"].to_numpy(int)
    par = df_test_prep["partner_idx"].to_numpy(int)
    con = df_test_prep["contestant_idx"].to_numpy(int)
    age_z = df_test_prep["age_z"].to_numpy(float)
    fame_z = df_test_prep["log_wiki_edits_z"].to_numpy(float)
    judge_z = df_test_prep["judge_pct_z"].to_numpy(float)
    judge_pct = df_test_prep["judge_pct"].to_numpy(float)
    prev_z = df_test_prep["prev_judge_total_z"].to_numpy(float) if artifacts.use_prev_judge_total else None
    delta_z = df_test_prep["judge_delta_z"].to_numpy(float) if artifacts.use_judge_delta else None

    true_row = df_test_prep.attrs["elim_row_by_group"].astype(int)

    p_sum = np.zeros(len(df_test_prep), dtype=float)
    for s in sample_idx:
        eta = (
            draws["beta0"][s]
            + draws["beta_age"][s] * age_z
            + draws["beta_fame"][s] * fame_z
            + draws["beta_judge"][s] * judge_z
            + draws["industry_effect"][s][ind]
            + draws["partner_effect"][s][par]
            + draws["b_contestant"][s][con]
        )
        if artifacts.use_prev_judge_total:
            eta = eta + draws["beta_prev_judge"][s] * prev_z
        if artifacts.use_judge_delta:
            eta = eta + draws["beta_delta"][s] * delta_z

        F = _softmax_by_group_numpy(eta, group_idx, n_groups)
        C = judge_pct + F
        logits = -draws["alpha"][s] * C
        p = _softmax_by_group_numpy(logits, group_idx, n_groups)
        p_sum += p

    p_mean = p_sum / len(sample_idx)

    pred_row = np.full(n_groups, -1, dtype=int)
    ranks: list[int] = []
    for g in range(n_groups):
        rows_g = np.where(group_idx == g)[0]
        pg = p_mean[rows_g]
        pred_row[g] = int(rows_g[np.argmax(pg)])
        order = np.argsort(-pg)
        rank = int(np.where(rows_g[order] == true_row[g])[0][0]) + 1
        ranks.append(rank)

    return {
        "n_weeks_test": int(n_groups),
        "top1_accuracy": float(np.mean(pred_row == true_row)),
        "avg_rank_true_elim": float(np.mean(ranks)),
        "ranks": ranks,
    }


# -----------------------------
# Example usage (no file I/O)
# -----------------------------


if __name__ == "__main__":
    # This expects df_long exists in the current Python process.
    try:
        df_long  # type: ignore[name-defined]
    except NameError as e:
        raise SystemExit(
            "`df_long` is not defined. In your notebook/script, construct df_long first, then run:\n"
            "  df_train = df_long[(df_long.season>=3) & (df_long.season<=23)]\n"
            "  df_test  = df_long[(df_long.season>=24) & (df_long.season<=27)]\n"
            "  df_train_prep, artifacts = prepare_week_groups(df_train)\n"
            "  model = build_model(df_train_prep, artifacts)\n"
            "  idata = fit_model(model)\n"
            "  print(posterior_summaries(idata))\n"
            "  vote_pp = posterior_vote_shares(idata, df_train_prep, artifacts)\n"
            "  metrics = evaluate(model, idata, df_test, artifacts)\n"
        ) from e


