import os
import re

os.environ.setdefault("MPLCONFIGDIR", "AR-EDA/.mplconfig")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
FIG_PATH = "AR-EDA/eda_summary.png"
PARTNER_PATH = "AR-EDA/partner_effect.png"
PARTNER_TABLE = "AR-EDA/partner_effect.csv"
PARTNER_PERMUTATIONS = 1000


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, na_values=["N/A"])


def get_week_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]


def to_long(df: pd.DataFrame, week_cols: list[str]) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=[
            "celebrity_name",
            "ballroom_partner",
            "celebrity_industry",
            "celebrity_homestate",
            "celebrity_homecountry/region",
            "celebrity_age_during_season",
            "season",
            "results",
            "placement",
        ],
        value_vars=week_cols,
        var_name="week_judge",
        value_name="score",
    )
    long_df["week"] = long_df["week_judge"].str.extract(r"week(\d+)_").astype(int)
    long_df["judge"] = long_df["week_judge"].str.extract(r"_judge(\d+)_").astype(int)
    return long_df


def parse_elimination_week(results: pd.Series) -> pd.Series:
    elim_week = results.str.extract(r"Eliminated Week (\d+)", expand=False)
    return pd.to_numeric(elim_week, errors="coerce")


def summarize_data(df: pd.DataFrame, long_df: pd.DataFrame) -> dict:
    week_cols = get_week_cols(df)
    elim_week = parse_elimination_week(df["results"])

    week_scores = long_df.copy()
    week_scores["score"] = pd.to_numeric(week_scores["score"], errors="coerce")

    non_null_scores = week_scores["score"].dropna()
    active_week_max = (
        week_scores.loc[week_scores["score"].notna()]
        .groupby(["celebrity_name", "season"])["week"]
        .max()
    )

    elim_week_idx = (
        df.assign(elim_week=elim_week)
        .set_index(["celebrity_name", "season"])["elim_week"]
    )
    last_active = active_week_max.reindex(elim_week_idx.index)

    zeros_before_elim = 0
    if not elim_week_idx.isna().all():
        merged = week_scores.merge(
            elim_week_idx.rename("elim_week"),
            on=["celebrity_name", "season"],
            how="left",
        )
        zeros_before_elim = merged[
            (merged["elim_week"].notna())
            & (merged["week"] <= merged["elim_week"])
            & (merged["score"] == 0)
        ].shape[0]

    high_scores = non_null_scores[non_null_scores > 10]
    negative_scores = non_null_scores[non_null_scores < 0]

    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "seasons": df["season"].nunique(),
        "week_cols": len(week_cols),
        "missing_scores": week_scores["score"].isna().sum(),
        "zero_scores": (week_scores["score"] == 0).sum(),
        "zeros_before_elim": zeros_before_elim,
        "score_min": non_null_scores.min(),
        "score_max": non_null_scores.max(),
        "high_scores_count": high_scores.shape[0],
        "negative_scores_count": negative_scores.shape[0],
        "elim_week_mismatch": (elim_week_idx.notna() & (last_active < elim_week_idx)).sum(),
    }


def make_plots(df: pd.DataFrame, long_df: pd.DataFrame) -> None:
    long_df["score"] = pd.to_numeric(long_df["score"], errors="coerce")

    # Treat zero after elimination as missing for score distribution summaries.
    last_active = (
        long_df.loc[long_df["score"].notna()]
        .groupby(["celebrity_name", "season"])["week"]
        .max()
        .rename("last_week")
    )
    long_df = long_df.merge(
        last_active, on=["celebrity_name", "season"], how="left"
    )
    long_df["score_active"] = long_df["score"].where(
        long_df["week"] <= long_df["last_week"]
    )
    long_df["score_active"] = long_df["score_active"].replace(0, pd.NA)

    season_counts = df.groupby("season")["celebrity_name"].nunique()
    season_weeks = (
        long_df.loc[long_df["score"].notna()]
        .groupby("season")["week"]
        .max()
    )

    avg_by_week = (
        long_df.groupby("week")["score_active"].mean().reset_index()
    )

    age_place = df[
        ["celebrity_age_during_season", "placement", "celebrity_industry"]
    ].copy()
    age_place["celebrity_age_during_season"] = pd.to_numeric(
        age_place["celebrity_age_during_season"], errors="coerce"
    )
    age_place["placement"] = pd.to_numeric(age_place["placement"], errors="coerce")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ax = axes[0, 0]
    ax.bar(season_counts.index, season_counts.values, color="#4C72B0")
    ax.set_title("Contestants per Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Contestants")

    ax = axes[0, 1]
    ax.plot(season_weeks.index, season_weeks.values, marker="o", color="#55A868")
    ax.set_title("Season Length (max week with scores)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Weeks")

    ax = axes[1, 0]
    ax.hist(
        long_df["score_active"].dropna(),
        bins=20,
        color="#C44E52",
        alpha=0.8,
    )
    ax.set_title("Judge Score Distribution (active weeks)")
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")

    ax = axes[1, 1]
    contestant_avg = (
        long_df.groupby(["season", "celebrity_name"])["score_active"]
        .mean()
        .reset_index()
    )
    contestant_avg = contestant_avg.merge(
        df[["season", "celebrity_name", "placement"]], on=["season", "celebrity_name"]
    )
    contestant_avg["placement"] = pd.to_numeric(
        contestant_avg["placement"], errors="coerce"
    )
    ax.scatter(
        contestant_avg["score_active"],
        contestant_avg["placement"],
        s=18,
        alpha=0.5,
        color="#8172B3",
    )
    ax.invert_yaxis()
    ax.set_title("Avg Judge Score vs Final Placement")
    ax.set_xlabel("Avg Judge Score (active weeks)")
    ax.set_ylabel("Placement")

    ax = axes[0, 2]
    elim_week = parse_elimination_week(df["results"])
    elim_map = (
        df.assign(elim_week=elim_week)
        .dropna(subset=["elim_week"])
        .set_index(["season", "celebrity_name"])["elim_week"]
    )
    weekly_scores = (
        long_df[long_df["score_active"].notna()]
        .groupby(["season", "week", "celebrity_name"])["score_active"]
        .mean()
        .reset_index()
    )
    weekly_min = (
        weekly_scores.groupby(["season", "week"])["score_active"]
        .min()
        .reset_index()
        .rename(columns={"score_active": "week_min"})
    )
    elim_scores = weekly_scores.merge(
        elim_map.rename("elim_week").reset_index(),
        on=["season", "celebrity_name"],
        how="inner",
    )
    elim_scores = elim_scores[elim_scores["week"] == elim_scores["elim_week"]]
    elim_scores = elim_scores.merge(weekly_min, on=["season", "week"], how="left")
    elim_scores["risk_gap"] = elim_scores["score_active"] - elim_scores["week_min"]
    risk_gap = (
        elim_scores.groupby(["season", "celebrity_name"])["risk_gap"]
        .mean()
        .reset_index()
    )
    risk_gap = risk_gap.merge(
        df[["season", "celebrity_name", "placement"]], on=["season", "celebrity_name"]
    )
    risk_gap["placement"] = pd.to_numeric(risk_gap["placement"], errors="coerce")
    risk_gap["placement_rank"] = (
        risk_gap.groupby("season")["placement"]
        .rank(ascending=True, method="average")
    )
    ax.scatter(
        risk_gap["risk_gap"],
        risk_gap["placement_rank"],
        s=18,
        alpha=0.5,
        color="#4C72B0",
    )
    ax.invert_yaxis()
    ax.axvline(0, color="#999999", linewidth=1, linestyle="--")
    ax.set_title("Elimination Risk Gap vs Final Placement")
    ax.set_xlabel("Eliminated Score − Week Min (avg)")
    ax.set_ylabel("Final Placement Rank")

    ax = axes[1, 2]
    weekly_means = (
        long_df.groupby(["season", "week", "celebrity_name"])["score_active"]
        .mean()
        .reset_index()
        .dropna(subset=["score_active"])
    )
    avg_judge = (
        weekly_means.groupby(["season", "celebrity_name"])["score_active"]
        .mean()
        .reset_index()
    )
    avg_judge["judge_rank"] = (
        avg_judge.groupby("season")["score_active"]
        .rank(ascending=False, method="average")
    )
    placement_rank = (
        df[["season", "celebrity_name", "placement"]]
        .assign(placement=lambda d: pd.to_numeric(d["placement"], errors="coerce"))
        .dropna(subset=["placement"])
        .assign(
            placement_rank=lambda d: d.groupby("season")["placement"]
            .rank(ascending=True, method="average")
        )
    )
    fan_gap = avg_judge.merge(
        placement_rank, on=["season", "celebrity_name"], how="inner"
    )
    fan_gap["fan_boost"] = fan_gap["judge_rank"] - fan_gap["placement_rank"]
    sc = ax.scatter(
        fan_gap["judge_rank"],
        fan_gap["placement_rank"],
        c=fan_gap["fan_boost"],
        cmap="coolwarm",
        s=18,
        alpha=0.6,
    )
    max_rank = max(fan_gap["judge_rank"].max(), fan_gap["placement_rank"].max())
    ax.plot([1, max_rank], [1, max_rank], color="#999999", linestyle="--")
    ax.invert_yaxis()
    ax.set_title("Placement vs Judge Rank (Fan Boost Colored)")
    ax.set_xlabel("Avg Judge Rank (lower = better judges)")
    ax.set_ylabel("Final Placement Rank (lower = better)")
    fig.colorbar(sc, ax=ax, label="Judge Rank − Placement Rank")

    fig.suptitle("DWTS Data – Initial EDA", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)


def analyze_partner_effect(df: pd.DataFrame) -> pd.DataFrame:
    placements = df[
        ["season", "celebrity_name", "ballroom_partner", "placement"]
    ].copy()
    placements["placement"] = pd.to_numeric(placements["placement"], errors="coerce")
    placements = placements.dropna(subset=["placement"])
    placements["placement_rank"] = placements.groupby("season")["placement"].rank(
        ascending=True, method="average"
    )
    placements["rank_z"] = placements.groupby("season")["placement_rank"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-8)
    )

    partner_stats = (
        placements.groupby("ballroom_partner")
        .agg(
            n_contestants=("celebrity_name", "count"),
            mean_rank=("placement_rank", "mean"),
            mean_rank_z=("rank_z", "mean"),
            std_rank_z=("rank_z", "std"),
        )
        .reset_index()
    )
    partner_stats["se_rank_z"] = partner_stats["std_rank_z"] / np.sqrt(
        partner_stats["n_contestants"]
    )
    partner_stats["t_stat"] = partner_stats["mean_rank_z"] / (
        partner_stats["se_rank_z"] + 1e-8
    )

    min_n = 5
    filtered = partner_stats[partner_stats["n_contestants"] >= min_n].copy()

    # Permutation test: shuffle partner labels within season.
    rng = np.random.default_rng(42)
    partner_list = filtered["ballroom_partner"].tolist()
    perm_counts = {p: 0 for p in partner_list}
    if len(partner_list) > 0:
        observed = filtered.set_index("ballroom_partner")["mean_rank_z"].to_dict()
        for _ in range(PARTNER_PERMUTATIONS):
            shuffled = placements.copy()
            shuffled["ballroom_partner"] = shuffled.groupby("season")[
                "ballroom_partner"
            ].transform(lambda s: rng.permutation(s.values))
            perm_means = (
                shuffled.groupby("ballroom_partner")["rank_z"].mean().to_dict()
            )
            for p in partner_list:
                if p not in perm_means:
                    continue
                if abs(perm_means[p]) >= abs(observed[p]):
                    perm_counts[p] += 1
        filtered["p_perm"] = filtered["ballroom_partner"].map(
            lambda p: (perm_counts.get(p, 0) + 1) / (PARTNER_PERMUTATIONS + 1)
        )
    else:
        filtered["p_perm"] = np.nan

    filtered = filtered.sort_values("mean_rank_z")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        filtered["ballroom_partner"],
        filtered["mean_rank_z"],
        color="#4C78A8",
        alpha=0.8,
    )
    ax.errorbar(
        filtered["mean_rank_z"],
        filtered["ballroom_partner"],
        xerr=1.96 * filtered["se_rank_z"],
        fmt="none",
        ecolor="#444444",
        elinewidth=1,
        alpha=0.7,
    )
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean Placement Rank (z-score within season)")
    ax.set_ylabel("Ballroom Partner (>=5 contestants)")
    ax.set_title("Ballroom Partner Effect on Placement (season-normalized)")
    fig.tight_layout()
    fig.savefig(PARTNER_PATH, dpi=200)

    partner_stats = partner_stats.merge(
        filtered[["ballroom_partner", "p_perm"]],
        on="ballroom_partner",
        how="left",
    ).sort_values("mean_rank_z")
    partner_stats.to_csv(PARTNER_TABLE, index=False)
    return partner_stats


def main() -> None:
    df = load_data(DATA_PATH)
    week_cols = get_week_cols(df)
    long_df = to_long(df, week_cols)
    summary = summarize_data(df, long_df)

    print("=== Data Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nNotes:")
    print("- Scores >10 occur (bonus/averaged multi-dance weeks).")
    print("- Zeros after elimination are present by design.")
    print("- N/A indicates no judge or no week in a season.")

    make_plots(df, long_df)
    print(f"\nSaved visualization to {FIG_PATH}")
    analyze_partner_effect(df)
    print(f"Saved partner effect plot to {PARTNER_PATH}")
    print(f"Saved partner effect table to {PARTNER_TABLE}")


if __name__ == "__main__":
    main()
