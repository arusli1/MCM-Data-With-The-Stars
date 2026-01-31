"""
DWTS: Google Trends (pytrends) fame vs outcomes (placement), controlling for judge scores.

Inputs:
  - Data/2026_MCM_Problem_C_Data.csv
  - Archit_Preliminary/celebrity_season_premiere_table.csv  (for season premiere dates)

Outputs (written under Archit_P1/pytrends_fame/):
  - outputs/pytrends_fame_scores.csv
  - outputs/pytrends_fame_merged.csv
  - outputs/correlations.csv
  - outputs/plots/*.png
  - pytrends_fame_vs_outcomes_report.md

Fame definition (anchored stitching):
  Google Trends scales values within each request.
  For each season, we query groups of <=4 contestants plus an anchor term (default: "Barack Obama").
  For each contestant: fame_ratio = mean(interest(contestant)) / mean(interest(anchor)) over preseason window.
  This makes contestants comparable across groups for that season.

Preseason window:
  [premiere_date - window_days, premiere_date - 1 day] (inclusive).

Run:
  python3 -m pip install pytrends
  python3 Archit_P1/pytrends_fame/pytrends_fame_vs_outcomes.py
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure matplotlib caches are writable; use a non-interactive backend.
_HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(_HERE / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(_HERE / ".cache"))

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clean_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _detect_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _week_judge_columns(df: pd.DataFrame) -> Dict[int, list[str]]:
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
    df["celebrity_name"] = df["celebrity_name"].map(_clean_name)
    df["season"] = pd.to_numeric(df["season"], errors="raise").astype(int)
    df["placement"] = pd.to_numeric(df["placement"], errors="coerce")

    week_cols = _week_judge_columns(df)
    if not week_cols:
        raise ValueError("No week/judge score columns found in DWTS dataset.")

    # Aggregate judge totals by week.
    for w, cols in week_cols.items():
        block = df[cols].replace("N/A", np.nan)
        block = block.apply(pd.to_numeric, errors="coerce")
        df[f"week{w}_judge_total"] = block.sum(axis=1, min_count=1)

    week_total_cols = [f"week{w}_judge_total" for w in week_cols]
    week_totals = df[week_total_cols]
    danced_mask = week_totals.fillna(0.0) > 0.0
    df["weeks_danced"] = danced_mask.sum(axis=1).astype(int)

    def _mean_over_mask(vals_row: pd.Series, mask_row: pd.Series) -> float:
        vals = vals_row[mask_row.values]
        if len(vals) == 0:
            return np.nan
        return float(np.nanmean(vals))

    df["judge_avg_week_total"] = [
        _mean_over_mask(week_totals.iloc[i], danced_mask.iloc[i]) for i in range(len(df))
    ]

    # Season cast size & placement percentile
    df["season_cast_size"] = df.groupby("season")["celebrity_name"].transform("count").astype(int)
    df["placement_pct"] = (df["placement"] - 1) / (df["season_cast_size"] - 1)
    return df


def load_season_premieres(path: Path) -> Dict[int, str]:
    """
    Returns season -> premiere_date_iso (YYYY-MM-DD)
    """
    df = pd.read_csv(path)
    season_col = _detect_col(df, ["season"])
    prem_col = _detect_col(df, ["season_premiere_date"])
    if not season_col or not prem_col:
        raise ValueError("Season table must include columns: season, season_premiere_date")
    tmp = df[[season_col, prem_col]].drop_duplicates()
    tmp = tmp.rename(columns={season_col: "season", prem_col: "season_premiere_date"})
    tmp["season"] = pd.to_numeric(tmp["season"], errors="raise").astype(int)
    tmp["season_premiere_date"] = tmp["season_premiere_date"].astype(str)
    out: Dict[int, str] = {}
    for r in tmp.itertuples(index=False):
        d = str(r.season_premiere_date).strip()
        if not d or d.lower() == "nan":
            continue
        out[int(r.season)] = d
    return out


def preseason_timeframe(premiere_date_iso: str, window_days: int) -> str:
    """
    Build pytrends timeframe string: YYYY-MM-DD YYYY-MM-DD
    """
    d0 = pd.to_datetime(premiere_date_iso)
    start = d0 - pd.Timedelta(days=window_days)
    end = d0 - pd.Timedelta(days=1)
    return f"{start.date().isoformat()} {end.date().isoformat()}"


@dataclass
class FameRow:
    season: int
    celebrity_name: str
    timeframe: str
    anchor_term: str
    mean_interest: float
    mean_anchor: float
    fame_ratio: float
    status: str
    notes: str


def _chunks(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _safe_mean(s: pd.Series) -> float:
    if s is None or len(s) == 0:
        return float("nan")
    return float(np.nanmean(pd.to_numeric(s, errors="coerce").to_numpy()))


def compute_pytrends_fame_scores(
    *,
    dwts: pd.DataFrame,
    premieres: Dict[int, str],
    window_days: int,
    anchor_term: str,
    sleep_seconds: float,
    hl: str,
    tz: int,
) -> pd.DataFrame:
    """
    For each season, compute fame_ratio for each contestant using pytrends.
    """
    try:
        from pytrends.request import TrendReq  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "pytrends is not installed. Install it with:\n"
            "  python3 -m pip install pytrends\n"
            f"Import error: {exc}"
        )

    rows: List[FameRow] = []
    tr = TrendReq(hl=hl, tz=tz)

    # Work season-by-season because timeframe depends on premiere date.
    for season, sdf in dwts.groupby("season"):
        if int(season) not in premieres:
            # Skip seasons without premiere date.
            for name in sorted(set(sdf["celebrity_name"])):
                rows.append(
                    FameRow(
                        season=int(season),
                        celebrity_name=name,
                        timeframe="",
                        anchor_term=anchor_term,
                        mean_interest=float("nan"),
                        mean_anchor=float("nan"),
                        fame_ratio=float("nan"),
                        status="missing_premiere_date",
                        notes="No season_premiere_date available for this season.",
                    )
                )
            continue

        timeframe = preseason_timeframe(premieres[int(season)], window_days)
        names = sorted({_clean_name(n) for n in sdf["celebrity_name"] if _clean_name(n)})

        # Query in groups of 4 + anchor (5 terms max).
        for group in _chunks(names, 4):
            kw_list = group + [anchor_term]
            try:
                tr.build_payload(kw_list=kw_list, timeframe=timeframe)
                data = tr.interest_over_time()
            except Exception as exc:
                # Mark all in this group as failed.
                for n in group:
                    rows.append(
                        FameRow(
                            season=int(season),
                            celebrity_name=n,
                            timeframe=timeframe,
                            anchor_term=anchor_term,
                            mean_interest=float("nan"),
                            mean_anchor=float("nan"),
                            fame_ratio=float("nan"),
                            status="pytrends_error",
                            notes=f"build_payload/interest_over_time failed: {exc}",
                        )
                    )
                time.sleep(sleep_seconds)
                continue

            if data is None or data.empty:
                for n in group:
                    rows.append(
                        FameRow(
                            season=int(season),
                            celebrity_name=n,
                            timeframe=timeframe,
                            anchor_term=anchor_term,
                            mean_interest=float("nan"),
                            mean_anchor=float("nan"),
                            fame_ratio=float("nan"),
                            status="empty_response",
                            notes="interest_over_time returned empty.",
                        )
                    )
                time.sleep(sleep_seconds)
                continue

            mean_anchor = _safe_mean(data.get(anchor_term, pd.Series(dtype=float)))
            for n in group:
                mean_interest = _safe_mean(data.get(n, pd.Series(dtype=float)))
                if not np.isfinite(mean_anchor) or mean_anchor <= 0:
                    rows.append(
                        FameRow(
                            season=int(season),
                            celebrity_name=n,
                            timeframe=timeframe,
                            anchor_term=anchor_term,
                            mean_interest=mean_interest,
                            mean_anchor=mean_anchor,
                            fame_ratio=float("nan"),
                            status="anchor_zero_or_nan",
                            notes="Anchor mean interest was 0/NaN; cannot scale.",
                        )
                    )
                else:
                    rows.append(
                        FameRow(
                            season=int(season),
                            celebrity_name=n,
                            timeframe=timeframe,
                            anchor_term=anchor_term,
                            mean_interest=mean_interest,
                            mean_anchor=mean_anchor,
                            fame_ratio=float(mean_interest / mean_anchor) if np.isfinite(mean_interest) else float("nan"),
                            status="ok",
                            notes="",
                        )
                    )

            time.sleep(sleep_seconds)

    out = pd.DataFrame([r.__dict__ for r in rows])
    out["log_fame_ratio"] = np.log1p(pd.to_numeric(out["fame_ratio"], errors="coerce"))
    return out


def spearman_corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3:
        return np.nan, np.nan, int(len(d))
    r, p = stats.spearmanr(d["x"].to_numpy(), d["y"].to_numpy())
    return float(r), float(p), int(len(d))


def pearson_corr(x: pd.Series, y: pd.Series) -> Tuple[float, float, int]:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 3:
        return np.nan, np.nan, int(len(d))
    r, p = stats.pearsonr(d["x"].to_numpy(), d["y"].to_numpy())
    return float(r), float(p), int(len(d))


def partial_corr_residualize(x: pd.Series, y: pd.Series, z: pd.Series) -> Dict[str, float]:
    d = pd.DataFrame({"x": x, "y": y, "z": z}).dropna()
    if len(d) < 10:
        return {"n": float(len(d)), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    Z = d["z"].to_numpy()
    Xz = np.column_stack([np.ones(len(d)), Z])
    bx, *_ = np.linalg.lstsq(Xz, d["x"].to_numpy(), rcond=None)
    by, *_ = np.linalg.lstsq(Xz, d["y"].to_numpy(), rcond=None)
    x_res = d["x"].to_numpy() - Xz @ bx
    y_res = d["y"].to_numpy() - Xz @ by
    pr, pp = stats.pearsonr(x_res, y_res)
    sr, sp = stats.spearmanr(x_res, y_res)
    return {"n": float(len(d)), "pearson_r": float(pr), "pearson_p": float(pp), "spearman_r": float(sr), "spearman_p": float(sp)}


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    d = df[[y_col] + x_cols].dropna().copy()
    y = d[y_col].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(dtype=float) for c in x_cols])
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - (X @ beta)
    df_resid = max(n - k, 1)
    sigma2 = float((resid.T @ resid) / df_resid)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov_beta))
    tvals = beta / se
    pvals = 2.0 * stats.t.sf(np.abs(tvals), df=df_resid)
    tcrit = stats.t.ppf(0.975, df=df_resid)
    ci_low = beta - tcrit * se
    ci_high = beta + tcrit * se
    terms = ["Intercept"] + x_cols
    return pd.DataFrame({"term": terms, "coef": beta, "se": se, "t": tvals, "p": pvals, "ci_low": ci_low, "ci_high": ci_high})


def _df_to_md_table(df: pd.DataFrame) -> str:
    df = df.copy()
    for c in df.columns:
        df[c] = df[c].map(lambda v: "NA" if pd.isna(v) else str(v))
    headers = list(df.columns)
    rows = df.values.tolist()
    out = []
    out.append("| " + " | ".join(h.replace("|", "\\|") for h in headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(str(v).replace("|", "\\|") for v in r) + " |")
    return "\n".join(out) + "\n"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-days", type=int, default=30, help="Days before premiere to compute fame (default: 30)")
    ap.add_argument("--anchor-term", type=str, default="Barack Obama", help="Anchor term used for stitching batches")
    ap.add_argument("--sleep-seconds", type=float, default=1.0, help="Sleep between pytrends calls (default: 1.0)")
    ap.add_argument("--hl", type=str, default="en-US", help="pytrends host language (default: en-US)")
    ap.add_argument("--tz", type=int, default=360, help="pytrends timezone offset minutes (default: 360)")
    args = ap.parse_args()

    root = _repo_root()
    dwts_path = root / "Data" / "2026_MCM_Problem_C_Data.csv"
    season_table = root / "Archit_Preliminary" / "celebrity_season_premiere_table.csv"

    out_dir = _HERE / "outputs"
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    dwts = load_dwts_outcomes(dwts_path)
    premieres = load_season_premieres(season_table)

    fame = compute_pytrends_fame_scores(
        dwts=dwts,
        premieres=premieres,
        window_days=args.window_days,
        anchor_term=args.anchor_term,
        sleep_seconds=args.sleep_seconds,
        hl=args.hl,
        tz=args.tz,
    )
    fame_out = out_dir / "pytrends_fame_scores.csv"
    fame.to_csv(fame_out, index=False)

    fame_ok = fame[fame["status"] == "ok"].copy()
    merged = dwts.merge(
        fame_ok[["season", "celebrity_name", "fame_ratio", "log_fame_ratio"]],
        on=["season", "celebrity_name"],
        how="left",
        validate="one_to_one",
    )
    merged["has_fame"] = merged["fame_ratio"].notna()
    merged_out = out_dir / "pytrends_fame_merged.csv"
    merged.to_csv(merged_out, index=False)

    df = merged[merged["has_fame"]].copy()
    corrs: Dict[str, Tuple[float, float, int]] = {}
    corrs["Spearman(placement_pct, log_fame_ratio)"] = spearman_corr(df["placement_pct"], df["log_fame_ratio"])
    corrs["Spearman(placement_pct, judge_avg_week_total)"] = spearman_corr(df["placement_pct"], df["judge_avg_week_total"])
    corrs["Spearman(log_fame_ratio, judge_avg_week_total)"] = spearman_corr(df["log_fame_ratio"], df["judge_avg_week_total"])
    corrs["Pearson(placement_pct, log_fame_ratio)"] = pearson_corr(df["placement_pct"], df["log_fame_ratio"])

    partial = partial_corr_residualize(df["log_fame_ratio"], df["placement_pct"], df["judge_avg_week_total"])
    reg = fit_ols(df, "placement_pct", ["judge_avg_week_total", "log_fame_ratio"])

    corr_rows = [{"metric": k, "corr_r": v[0], "p_value": v[1], "n": v[2]} for k, v in sorted(corrs.items())]
    pd.DataFrame(corr_rows).to_csv(out_dir / "correlations.csv", index=False)

    _plot_scatter(
        df,
        x="log_fame_ratio",
        y="placement_pct",
        hue="judge_avg_week_total",
        title="Placement percentile vs pytrends fame (colored by judge avg)",
        out_path=plot_dir / "scatter_placementpct_vs_pytrends_fame_hue_judge.png",
    )
    _plot_scatter(
        df,
        x="judge_avg_week_total",
        y="placement_pct",
        hue="log_fame_ratio",
        title="Placement percentile vs judge avg (colored by pytrends fame)",
        out_path=plot_dir / "scatter_placementpct_vs_judge_hue_pytrends_fame.png",
    )
    _plot_binned_means(
        df,
        x="log_fame_ratio",
        y="placement_pct",
        q=5,
        title="Mean placement percentile by pytrends fame quintile",
        out_path=plot_dir / "binned_mean_placementpct_by_pytrends_fame_quintile.png",
    )

    # Paper-style report
    rep = []
    rep.append("# Google Trends (pytrends) fame vs outcomes (placement)\n\n")
    rep.append("## Data\n")
    rep.append(f"- Total contestant-season rows (DWTS outcomes): **{len(dwts)}**\n")
    rep.append(f"- Rows with pytrends fame (status=ok): **{int(df.shape[0])}**\n")
    rep.append(f"- Fame window: **{args.window_days}** days before season premiere (per season)\n")
    rep.append(f"- Anchor term: **{args.anchor_term}** (used to stitch batches)\n\n")

    rep.append("## Correlations (lower placement_pct is better)\n")
    rep.append("| Metric | r | p | n |\n|---|---:|---:|---:|\n")
    for k in sorted(corrs.keys()):
        r, p, n = corrs[k]
        rep.append(f"| {k} | {r:.3f} | {p:.2g} | {n} |\n")

    rep.append("\n## Partial correlation controlling for judge score\n")
    rep.append(
        f"- Spearman residual r: **{partial['spearman_r']:.3f}** (p={partial['spearman_p']:.2g}, n={int(partial['n'])})\n"
        f"- Pearson residual r: **{partial['pearson_r']:.3f}** (p={partial['pearson_p']:.2g}, n={int(partial['n'])})\n"
    )

    rep.append("\n## Regression (OLS): placement_pct ~ judge + fame\n")
    rep.append(_df_to_md_table(reg.round(6)))

    rep.append("\n## Plots\n")
    for p in sorted(plot_dir.glob("*.png")):
        rep.append(f"- `{p.as_posix()}`\n")

    rep.append("\n## Notes / caveats\n")
    rep.append(
        "- Google Trends values are *relative within each request*. The anchor-ratio helps comparability, but it is still noisy.\n"
        "- Names that are ambiguous (e.g., common names) may yield polluted Trends signals. A future improvement is to query with disambiguating context.\n"
        "- This is observational and not causal; judge scores and public attention co-evolve.\n"
    )

    report_path = _HERE / "pytrends_fame_vs_outcomes_report.md"
    report_path.write_text("".join(rep), encoding="utf-8")

    print(f"Wrote fame scores: {fame_out}")
    print(f"Wrote merged dataset: {merged_out}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote plots to: {plot_dir}")


if __name__ == "__main__":
    main()


