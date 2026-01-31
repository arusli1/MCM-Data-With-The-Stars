# ML Analysis Plan: Impact of Characteristics in DWTS

## Goal
Quantify the impact of celebrity characteristics (age, industry) and professional dancers on competition outcomes, judge scores, and fan votes using provided fan vote estimates.

## Data Sources
1. **Raw Data**: `2026_MCM_Problem_C_Data.csv`
2. **Fan Estimates**: `AR-Problem1-Base/final_results/base_inferred_shares.csv`

## Processing Pipeline

### 1. Data Preparation (`data_prep.py`)
- **Merge**: Join Raw Data and Fan Estimates on `(Season, Celebrity)`. assuming the fan vote file has weekly or summary data.
- **Cleaning**:
    - Handle missing values (N/As in scores).
    - Standardize 'Industry' (group Rare categories).
    - Encode 'Pro Partner' (High cardinality: maybe use target encoding or frequency encoding, or keep top N + 'Other').
- **Feature Aggregation**:
    - Since we want to analyze the impact *overall*, we might want season-level aggregates (Mean Fan Share, Mean Judge Score, Final Placement) OR week-level analysis.
    - **Decision**: Season-level aggregation (One row per Couple) is better for predicting "Placement". Week-level is better for "Judge Score" dynamics, but the prompt asks "How well a celebrity will do", implying final result.
    - **Features**:
        - `Age`
        - `Industry` (One-hot)
        - `Partner` (One-hot)
        - `Home Country` (Is_US vs International?)
        - `Season` (Control variable)

### 2. Modeling (`train_analysis_models.py`)
We will train three separate models to check if factors impact them differently:
1.  **Likelihood of Success**: Target = `Placement` (or `Percentile Rank` to normalize for season size).
2.  **Judge Appeal**: Target = `Average Judge Score` (Standardized per season).
3.  **Fan Appeal**: Target = `Average Fan Share` (from the provided CSV).

**Models**:
- Random Forest Regressor (for robustness and non-linearity).
- XGBoost (for performance).

### 3. Interpretation (`explain_models.py`)
- **SHAP Analysis**:
    - Calculate SHAP values for each model.
    - **Compare**:
        - Does `Age` hurt Fan Votes more than Judge Scores?
        - Do certain `Pro Partners` boost Fan Votes independent of Judge Scores?
    - **Pro Partner Effect**: Extract the mean SHAP value for each Pro Dancer to rank them by "Value Added".

## Outputs
- `analysis_summary.md`: Text report of findings.
- SHAP Summary Plots (saved as images).
