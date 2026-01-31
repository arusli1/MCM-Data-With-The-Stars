"""
DWTS: Does preseason Wikipedia "edit count" (fame proxy) relate to final placement?

This script joins:
  - Data/dwts_wiki_edits_preseason.csv (preseason Wikipedia revision counts per contestant-season)
  - Data/2026_MCM_Problem_C_Data.csv (DWTS contestant outcomes + per-week judge scores)

and produces:
  - Archit_P1/outputs/wiki_fame_merged.csv
  - Archit_P1/outputs/plots/*.png
  - Archit_P1/wiki_fame_vs_outcomes_report.md

Notes / modeling choices:
  - Outcome: final placement. Lower is better (1 = winner).
  - Cross-season comparability: we also compute placement percentile within season:
        placement_pct = (placement - 1) / (n_cast - 1)
    so 0 is best and 1 is worst.
  - Judge performance summary: for each week, we sum judge scores across judges.
    We then compute averages over weeks where the weekly total is > 0.
    (In this dataset, contestants typically have 0s after elimination/withdrawal.)
  - Fame: uses edits_total_to_cutoff from preseason file; we analyze log1p(edits).
  - "Third variable" view (judge scores): we report raw correlations and a
    partial-correlation style analysis by residualizing placement and fame on judge score.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Make matplotlib/font caches writable in this environment (prevents crashes in sandboxed runs).
_HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_HERE / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(_HERE / ".cache"))

# Fontconfig sometimes has no writable cache dirs in this environment.
# Force a writable cache directory via a minimal fontconfig config.
_fc_cache = _HERE / ".fontconfig-cache"
_fc_cache.mkdir(parents=True, exist_ok=True)
_fc_conf = _HERE / "fonts.conf"
if not _fc_conf.exists():
    _fc_conf.write_text(
        "\n".join(
            [
                '<?xml version="1.0"?>',
                '<!DOCTYPE fontconfig SYSTEM "fonts.dtd">',
                "<fontconfig>",
                f"  <cachedir>{_fc_cache.as_posix()}</cachedir>",
                "</fontconfig>",
                "",
            ]
        ),
        encoding="utf-8",
    )
os.environ.setdefault("FONTCONFIG_FILE", str(_fc_conf))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clean_name(s: str) -> str:
    # Normalize whitespace + strip.
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ")  # NBSP -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_wiki_preseason(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"celebrityname": "celebrity_name"})
    df["celebrity_name"] = df["celebrity_name"].map(_clean_name)
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["status"] = df["status"].astype(str)
    df["edits_total_to_cutoff"] = pd.to_numeric(df["edits_total_to_cutoff"], errors="coerce")
    df["log_edits"] = np.log1p(df["edits_total_to_cutoff"])
    return df


def _week_judge_columns(df: pd.DataFrame) -> Dict[int, list[str]]:
    """
    Map week -> list of judge score columns for that week.
    Expects columns like week3_judge2_score.
    """
    pat = re.compile(r"^week(\d+)_judge(\d+)_score$")
    week_cols: Dict[int, list[str]] = {}
    for c in df.columns:
        m = pat.match(c)
        if not m:
            continue
        w = int(m.group(1))
        week_cols.setdefault(w, []).append(c)
    for w in week_cols:
        week_cols[w] = sorted(week_cols[w])
    return dict(sorted(week_cols.items()))


def load_dwts_outcomes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "celebrity_name": "celebrity_name",
            "ballroom_partner": "ballroom_partner",
            "celebrity_industry": "celebrity_industry",
            "celebrity_age_during_season": "celebrity_age_during_season",
        }
    )
    df["celebrity_name"] = df["celebrity_name"].map(_clean_name)
    df["ballroom_partner"] = df["ballroom_partner"].map(_clean_name)
    df["celebrity_industry"] = df["celebrity_industry"].map(_clean_name)

    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["placement"] = pd.to_numeric(df["placement"], errors="coerce")

    # Judge score aggregation
    week_cols = _week_judge_columns(df)
    if not week_cols:
        raise ValueError("No week/judge score columns found in DWTS dataset.")

    # Parse N/A -> NaN then sum per week
    for w, cols in week_cols.items():
        block = df[cols].replace("N/A", np.nan)
        block = block.apply(pd.to_numeric, errors="coerce")
        df[f"week{w}_judge_total"] = block.sum(axis=1, min_count=1)

    week_total_cols = [f"week{w}_judge_total" for w in week_cols]
    week_totals = df[week_total_cols]

    # Weeks where contestant appears to have danced / received non-zero points
    danced_mask = week_totals.fillna(0.0) > 0.0
    df["weeks_danced"] = danced_mask.sum(axis=1).astype(int)

    def _mean_over_mask(row: pd.Series, mask_row: pd.Series) -> float:
        vals = row[mask_row.values]
        if len(vals) == 0:
            return np.nan
        return float(np.nanmean(vals))

    df["judge_avg_week_total"] = [
        _mean_over_mask(week_totals.iloc[i], danced_mask.iloc[i]) for i in range(len(df))
    ]
    df["judge_sum_week_total"] = week_totals.where(danced_mask, np.nan).sum(axis=1, min_count=1)
    df["judge_max_week_total"] = week_totals.where(danced_mask, np.nan).max(axis=1)

    # Early-weeks summary (less affected by survival length)
    early_weeks = [w for w in week_cols.keys() if w <= 3]
    early_cols = [f"week{w}_judge_total" for w in early_weeks]
    if early_cols:
        early = df[early_cols]
        early_mask = early.fillna(0.0) > 0.0
        df["judge_avg_first3_total"] = [
            _mean_over_mask(early.iloc[i], early_mask.iloc[i]) for i in range(len(df))
        ]
    else:
        df["judge_avg_first3_total"] = np.nan

    # Season cast size + placement percentile
    cast_size = df.groupby("season")["celebrity_name"].transform("count").astype(int)
    df["season_cast_size"] = cast_size
    df["placement_pct"] = (df["placement"] - 1) / (df["season_cast_size"] - 1)

    return df


def build_merged_dataset(dwts: pd.DataFrame, wiki: pd.DataFrame) -> pd.DataFrame:
    # Keep wiki "ok" rows with numeric edit counts
    wiki_ok = wiki[(wiki["status"] == "ok") & wiki["edits_total_to_cutoff"].notna()].copy()

    merged = dwts.merge(
        wiki_ok[
            [
                "season",
                "celebrity_name",
                "edits_total_to_cutoff",
                "log_edits",
                "wiki_title",
                "pageid",
            ]
        ],
        on=["season", "celebrity_name"],
        how="left",
        validate="one_to_one",
    )

    merged["has_wiki_ok"] = merged["edits_total_to_cutoff"].notna()
    return merged


def spearman_corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return np.nan, np.nan, int(len(df))
    r, p = stats.spearmanr(df["x"].to_numpy(), df["y"].to_numpy())
    return float(r), float(p), int(len(df))


def pearson_corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return np.nan, np.nan, int(len(df))
    r, p = stats.pearsonr(df["x"].to_numpy(), df["y"].to_numpy())
    return float(r), float(p), int(len(df))


def partial_corr_residualize(x: pd.Series, y: pd.Series, z: pd.Series) -> Dict[str, float]:
    """
    Partial correlation via residualization:
      x_res = x - E[x|z], y_res = y - E[y|z], corr(x_res, y_res)
    """
    d = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(d) < 10:
        return {"n": float(len(d)), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}

    # Regress x on z (with intercept), then y on z, via least squares.
    Z = d["z"].to_numpy()
    Xz = np.column_stack([np.ones(len(d)), Z])
    bx, *_ = np.linalg.lstsq(Xz, d["x"].to_numpy(), rcond=None)
    by, *_ = np.linalg.lstsq(Xz, d["y"].to_numpy(), rcond=None)
    x_res = d["x"].to_numpy() - Xz @ bx
    y_res = d["y"].to_numpy() - Xz @ by

    pr, pp = stats.pearsonr(x_res, y_res)
    sr, sp = stats.spearmanr(x_res, y_res)
    return {
        "n": float(len(d)),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }

def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> Dict[str, object]:
    """
    Simple OLS with intercept using NumPy.
    Returns dict containing coef table + fit stats.
    """
    d = df[[y_col] + x_cols].dropna().copy()
    y = d[y_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(dtype=float) for c in x_cols])
    n, k = X.shape

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(resid.T @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = np.nan if tss == 0 else 1.0 - rss / tss

    df_resid = max(n - k, 1)
    sigma2 = rss / df_resid
    XtX_inv = np.linalg.inv(X.T @ X)
    cov_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(cov_beta))
    tvals = beta / se
    pvals = 2.0 * stats.t.sf(np.abs(tvals), df=df_resid)
    tcrit = stats.t.ppf(0.975, df=df_resid)
    ci_low = beta - tcrit * se
    ci_high = beta + tcrit * se

    terms = ["Intercept"] + x_cols
    table = pd.DataFrame(
        {
            "term": terms,
            "coef": beta,
            "se": se,
            "t": tvals,
            "p": pvals,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )

    return {
        "table": table,
        "nobs": int(n),
        "df_resid": int(df_resid),
        "r2": float(r2),
        "y_col": y_col,
        "x_cols": list(x_cols),
    }


def _save_corr_table(corrs: Dict[str, Tuple[float, float, int]], out_path: Path) -> None:
    rows = []
    for k, (r, p, n) in corrs.items():
        rows.append({"metric": k, "corr_r": r, "p_value": p, "n": n})
    pd.DataFrame(rows).sort_values("metric").to_csv(out_path, index=False)


def _plot_scatter(df: pd.DataFrame, x: str, y: str, hue: str, title: str, out_path: Path) -> None:
    d = df[[x, y, hue]].dropna().copy()
    if d.empty:
        return
    plt.figure(figsize=(8.5, 6))
    ax = sns.scatterplot(data=d, x=x, y=y, hue=hue, palette="viridis", alpha=0.8, edgecolor="none")
    ax.set_title(title)
    if y in ("placement", "placement_pct"):
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_binned_means(df: pd.DataFrame, x: str, y: str, q: int, title: str, out_path: Path) -> None:
    d = df[[x, y]].dropna().copy()
    if len(d) < q * 3:
        return
    d["bin"] = pd.qcut(d[x], q=q, duplicates="drop")
    g = d.groupby("bin", observed=False)[y].agg(["mean", "count", "std"]).reset_index()
    g["se"] = g["std"] / np.sqrt(g["count"].clip(lower=1))

    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    ax.errorbar(range(len(g)), g["mean"], yerr=1.96 * g["se"], fmt="o-", capsize=4)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(b) for b in g["bin"]], rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(y)
    if y in ("placement", "placement_pct"):
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_report(
    merged: pd.DataFrame,
    corrs: Dict[str, Tuple[float, float, int]],
    partial: Dict[str, float],
    ols_placement: Dict[str, object],
    ols_pct: Dict[str, object],
    out_md: Path,
    outputs_dir: Path,
) -> None:
    def _fmt_r(x: float) -> str:
        return "NA" if pd.isna(x) else f"{x:.3f}"

    def _fmt_p(x: float) -> str:
        return "NA" if pd.isna(x) else f"{x:.2g}"

    def _df_to_md_table(df: pd.DataFrame) -> str:
        """Minimal markdown table renderer (avoids pandas.to_markdown dependency on tabulate)."""
        df = df.copy()
        for c in df.columns:
            df[c] = df[c].map(lambda v: "NA" if pd.isna(v) else str(v))
        headers = list(df.columns)
        rows = df.values.tolist()

        def esc(s: str) -> str:
            return str(s).replace("|", "\\|")

        out = []
        out.append("| " + " | ".join(esc(h) for h in headers) + " |")
        out.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for r in rows:
            out.append("| " + " | ".join(esc(v) for v in r) + " |")
        return "\n".join(out) + "\n"

    lines = []
    lines.append("# Wikipedia fame (preseason edits) vs outcomes (placement)\n")
    lines.append("## Dataset join quality\n")
    lines.append(f"- Total contestant-season rows (DWTS outcomes): **{len(merged)}**\n")
    lines.append(f"- Rows with wiki preseason `status=ok` + numeric edits: **{int(merged['has_wiki_ok'].sum())}**\n")
    lines.append(f"- Rows missing wiki edits (dropped from fame analyses): **{int((~merged['has_wiki_ok']).sum())}**\n")

    lines.append("\n## Correlations (lower placement is better)\n")
    lines.append("| Metric | r | p | n |\n|---|---:|---:|---:|\n")
    for k in sorted(corrs.keys()):
        r, p, n = corrs[k]
        lines.append(f"| {k} | {_fmt_r(r)} | {_fmt_p(p)} | {n} |\n")

    lines.append("\n## Partial correlation: fame vs placement controlling for judge score\n")
    lines.append(
        f"- Using residualization with control `judge_avg_week_total` (HC0 OLS residuals).\n"
        f"- Pearson r (residuals): **{_fmt_r(partial.get('pearson_r', np.nan))}** (p={_fmt_p(partial.get('pearson_p', np.nan))}, n={int(partial.get('n', 0))})\n"
        f"- Spearman r (residuals): **{_fmt_r(partial.get('spearman_r', np.nan))}** (p={_fmt_p(partial.get('spearman_p', np.nan))}, n={int(partial.get('n', 0))})\n"
    )

    lines.append("\n## Regression (OLS): placement ~ judge + fame\n")
    tab1 = ols_placement["table"]  # type: ignore[index]
    lines.append(_df_to_md_table(tab1.round(6)))
    lines.append(f"- R²: **{float(ols_placement['r2']):.3f}** (n={int(ols_placement['nobs'])})\n")

    lines.append("\n## Regression (OLS): placement_pct ~ judge + fame\n")
    tab2 = ols_pct["table"]  # type: ignore[index]
    lines.append(_df_to_md_table(tab2.round(6)))
    lines.append(f"- R²: **{float(ols_pct['r2']):.3f}** (n={int(ols_pct['nobs'])})\n")

    lines.append("\n## Plots\n")
    plot_dir = outputs_dir / "plots"
    for fname in sorted([p.name for p in plot_dir.glob("*.png")]):
        lines.append(f"- `{(plot_dir / fname).as_posix()}`\n")

    lines.append("\n## Notes / caveats\n")
    lines.append(
        "- This is **observational** and not causal: judges’ scores and fan responses co-evolve through the season.\n"
        "- `judge_avg_week_total` uses only weeks with positive totals (to avoid the post-elimination 0 padding).\n"
        "- Some wiki rows are missing due to disambiguation / missing titles; those contestants are excluded from fame analyses.\n"
    )

    out_md.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    root = _repo_root()
    data_wiki = root / "Data" / "dwts_wiki_edits_preseason.csv"
    data_dwts = root / "Data" / "2026_MCM_Problem_C_Data.csv"

    outputs_dir = _HERE / "outputs"
    (outputs_dir / "plots").mkdir(parents=True, exist_ok=True)

    wiki = load_wiki_preseason(data_wiki)
    dwts = load_dwts_outcomes(data_dwts)
    merged = build_merged_dataset(dwts, wiki)

    # Save merged dataset for inspection
    merged_out = outputs_dir / "wiki_fame_merged.csv"
    merged.to_csv(merged_out, index=False)

    fame_df = merged[merged["has_wiki_ok"]].copy()

    # Correlations: (remember placement is lower=better, so negative means "more fame => better finish")
    corrs: Dict[str, Tuple[float, float, int]] = {}
    corrs["Spearman(placement, log_edits)"] = spearman_corr(fame_df["placement"], fame_df["log_edits"])
    corrs["Spearman(placement_pct, log_edits)"] = spearman_corr(fame_df["placement_pct"], fame_df["log_edits"])
    corrs["Spearman(placement, judge_avg_week_total)"] = spearman_corr(fame_df["placement"], fame_df["judge_avg_week_total"])
    corrs["Spearman(placement_pct, judge_avg_week_total)"] = spearman_corr(
        fame_df["placement_pct"], fame_df["judge_avg_week_total"]
    )
    corrs["Spearman(log_edits, judge_avg_week_total)"] = spearman_corr(fame_df["log_edits"], fame_df["judge_avg_week_total"])

    corrs["Pearson(placement_pct, log_edits)"] = pearson_corr(fame_df["placement_pct"], fame_df["log_edits"])
    corrs["Pearson(placement_pct, judge_avg_week_total)"] = pearson_corr(fame_df["placement_pct"], fame_df["judge_avg_week_total"])
    corrs["Pearson(log_edits, judge_avg_week_total)"] = pearson_corr(fame_df["log_edits"], fame_df["judge_avg_week_total"])

    # Partial correlation: fame vs placement (pct) controlling for judge
    partial = partial_corr_residualize(
        x=fame_df["log_edits"],
        y=fame_df["placement_pct"],
        z=fame_df["judge_avg_week_total"],
    )

    # Simple regressions: placement and placement_pct on judge + fame
    ols_placement = fit_ols(fame_df, "placement", ["judge_avg_week_total", "log_edits"])
    ols_pct = fit_ols(fame_df, "placement_pct", ["judge_avg_week_total", "log_edits"])

    # Output correlation table
    _save_corr_table(corrs, outputs_dir / "correlations.csv")

    # Plots
    _plot_scatter(
        fame_df,
        x="log_edits",
        y="placement_pct",
        hue="judge_avg_week_total",
        title="Placement percentile vs preseason Wikipedia edits (colored by judge avg)",
        out_path=outputs_dir / "plots" / "scatter_placementpct_vs_logedits_hue_judge.png",
    )
    _plot_scatter(
        fame_df,
        x="judge_avg_week_total",
        y="placement_pct",
        hue="log_edits",
        title="Placement percentile vs judge avg (colored by log edits)",
        out_path=outputs_dir / "plots" / "scatter_placementpct_vs_judge_hue_logedits.png",
    )
    _plot_binned_means(
        fame_df,
        x="log_edits",
        y="placement_pct",
        q=5,
        title="Mean placement percentile by fame quintile (log edits)",
        out_path=outputs_dir / "plots" / "binned_mean_placementpct_by_fame_quintile.png",
    )
    _plot_binned_means(
        fame_df,
        x="judge_avg_week_total",
        y="placement_pct",
        q=5,
        title="Mean placement percentile by judge avg quintile",
        out_path=outputs_dir / "plots" / "binned_mean_placementpct_by_judge_quintile.png",
    )

    # Report
    report_path = _HERE / "wiki_fame_vs_outcomes_report.md"
    write_report(
        merged=merged,
        corrs=corrs,
        partial=partial,
        ols_placement=ols_placement,
        ols_pct=ols_pct,
        out_md=report_path,
        outputs_dir=outputs_dir,
    )

    print(f"Wrote merged dataset: {merged_out}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote plots to: {outputs_dir / 'plots'}")


if __name__ == "__main__":
    main()


