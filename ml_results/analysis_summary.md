# DWTS Competition Outcome Analysis: ML Insights

## Methods Summary
We built tree-based regression models (XGBoost) to identify the drivers of competition success, early-window judges' scores, and fan support in "Dancing with the Stars." Analysis was performed at the Season–Contestant level (N=421) using Leave-One-Season-Out (LOGO) cross-validation to ensure robustness. Preprocessing included normalization of contestant names, bucketing categorical features, and engineering early-window performance metrics (Weeks 1–3). Model explainability was achieved via SHAP values, quantifying the contribution of each feature to the predictions.

## Model Performance (LOGO CV)
| Target Model | MAE | RMSE | Pearson | Spearman |
| :--- | :--- | :--- | :--- | :--- |
| **Success Score** | 0.194 | 0.239 | 0.707 | 0.674 |
| **Early Judges** | 3.691 | 4.401 | 0.362 | 0.373 |
| **Early Fans** | 0.044 | 0.060 | 0.096 | 0.121 |

## Key Findings

### Success Model (Final Outcome)
- **Top Positive Drivers**: Early Window Judges Mean (`judge_mean_w1_3`), Early Window Fan Share Median (`fan_mean_w1_3`), Younger Celebrity Age.
- **Top Negative Drivers**: Advanced Celebrity Age, Low Week 1 Scores.
- **Spearman Rank Correlation**: 0.674, indicating a strong ability to rank contestant outcomes based on early-window data.

### Early Judges Model
- **Top Positive Drivers**: Being a TV Personality or Actor/Actress, Partnering with Derek Hough.
- **Top Negative Drivers**: Advanced Celebrity Age, Certain Industry Buckets.
- **Spearman Rank Correlation**: 0.373.

### Early Fans Model
- **Top Positive Drivers**: Younger Celebrity Age, Partnering with Pro Dancers like Derek Hough or Valentin Chmerkovskiy.
- **Top Negative Drivers**: Advanced Celebrity Age, Certain bucketing effects.
- **Spearman Rank Correlation**: 0.121.

## Do celebrity/pro characteristics impact judges and fans the same way?
While both groups are significantly influenced by celebrity age (favoring younger contestants), judges are more responsive to professional industry backgrounds such as acting and TV roles. Fans show higher variability and are more strongly influenced by the specific pro dancer pairing (e.g., the "Hough effect"). The residualized pro-boost analysis confirms that certain pro dancers consistently outperform their celebrity baselines, even after controlling for early scores. However, these results are subject to selection bias and small sample sizes for some partners, as reflected in the bootstrap confidence intervals.
