import os
import re

os.environ.setdefault("MPLCONFIGDIR", "0_data_exploration/.mplconfig")

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = "data.csv"
FIG_PATH = "0_data_exploration/eda_summary.png"


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

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
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
    ax.scatter(
        age_place["celebrity_age_during_season"],
        age_place["placement"],
        s=18,
        alpha=0.5,
        color="#8172B3",
    )
    ax.invert_yaxis()
    ax.set_title("Age vs Final Placement (lower is better)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Placement")

    fig.suptitle("DWTS Data – Initial EDA", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=200)


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


if __name__ == "__main__":
    main()
