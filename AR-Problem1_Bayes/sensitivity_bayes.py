import pandas as pd

import bayesian as bm


DATA_PATH = "Data/2026_MCM_Problem_C_Data.csv"
OUT_PATH = "AR-Problem1_Bayes/sensitivity_bayes.csv"

# Keep sensitivity small and fast.
SEASONS = [1, 10, 34]
TEMP_GRID = [0.5, 0.7, 1.0]
LAM_HARD_GRID = [50.0, 150.0]
LAM_FINAL_GRID = [0.0, 5.0, 10.0]


def run() -> None:
    df = pd.read_csv(DATA_PATH, na_values=["N/A"])
    week_cols = bm.parse_week_cols(df)
    rows = []

    # Speed-focused settings
    base_steps = bm.N_STEPS
    bm.N_STEPS = 80
    bm.RUN_SGLD = False

    for temp in TEMP_GRID:
        for lam_hard in LAM_HARD_GRID:
            for lam_final in LAM_FINAL_GRID:
                bm.TEMP = temp
                bm.TEMP_PLACEMENT = temp
                bm.LAM_HARD = lam_hard
                bm.LAM_FINAL = lam_final

                for season in SEASONS:
                    df_season = df[df["season"] == season].reset_index(drop=True)
                    regime = bm.regime_for_season(season)
                    struct, shares, _ = bm.fit_season(
                        df_season, week_cols, regime, bm.torch.device("cpu")
                    )
                    hits, total = bm.elimination_match_rate(shares, struct, regime)
                    rows.append(
                        {
                            "season": season,
                            "temp": temp,
                            "lam_hard": lam_hard,
                            "lam_final": lam_final,
                            "match_rate": hits / total if total else 1.0,
                        }
                    )

    bm.N_STEPS = base_steps
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    run()
