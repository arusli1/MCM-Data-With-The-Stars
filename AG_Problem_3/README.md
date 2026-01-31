# DWTS Machine Learning Analysis Documentation

This folder contains the results of the machine learning analysis on "Dancing with the Stars" (DWTS) contestant outcomes.

## Input Files Used
- `Data/2026_MCM_Problem_C_Data.csv`: Main dataset with contestant characteristics and judge scores.
- `Data/estimate_votes.csv`: Inferred fan vote shares (`s_share`).

## Column Assumptions & Preprocessing
- **Success Score**: Derived from `placement` column. `score = 1 - (placement - 1) / (max_placement - 1)`.
- **Early Performance**: Computed from Weeks 1–3 only. Missing judge scores (zeros) were ignored in mean/slope calculations.
- **Categorical Bucketing**:
    - `celebrity_industry`: Top 10 categories + "Other".
    - `celebrity_homestate`: Top 10 categories + "Other".
    - `celebrity_homecountry/region`: Top 5 categories + "Other".
    - `ballroom_partner`: Top 15 partners + "Other_Partner".
- **Name Cleaning**: Celebrity names were lowercased, stripped of whitespace, and had punctuation removed to ensure clean merging between files.

## LOGO CV Setup
- **Strategy**: Leave-One-Season-Out (LOGO) cross-validation was used. In each fold, one full season was held out for testing, while the models were trained on the remaining seasons.
- **Models**: XGBRegressor with fixed hyperparameters (`max_depth=5`, `n_estimators=400`, `learning_rate=0.05`).
- **Explainability**: SHAP (SHapley Additive exPlanations) values were used to determine feature importance (mean absolute SHAP) and direction of impact.

## Pro-Boost residuals
Residualized success was calculated by fitting a Success Model *without* partner features and taking `observed - predicted`. Positive residuals indicate that the contestant performed better than expected given their background and early performance, suggesting a "boost" from the pro partner.
