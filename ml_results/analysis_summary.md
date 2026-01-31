# DWTS Machine Learning Analysis Summary

## Methods
We trained XGBoost and Random Forest regression models to predict competition success, early judge scores, and early fan support using Leave-One-Season-Out (LOGO) cross-validation (N=421 contestants). Feature engineering focused on celebrity characteristics and early-window (Weeks 1-3) performance metrics, with categorical bucketing for industries and professional partners. Model explainability was quantified using SHAP values to identify global influence and specific category effects.

## Results
*   **Success Score**: Achieving a Spearman correlation of 0.674, the model identified **Early Judge Mean**, **Early Fan Share**, and **Week 1 Fan Score** as the top positive drivers. Negative drivers included **Advanced Celebrity Age** and low initial performance metrics.
*   **Early Judges**: With a Spearman correlation of 0.373, judges were primarily influenced by **Television/Acting backgrounds** and specific pro-partners like **Derek Hough**. Older celebrity age was a significant negative driver of initial judge scores.
*   **Early Fans**: This model (Spearman 0.121) showed that fans are strongly driven by **Celebrity Age** (favoring younger) and the "halo effect" of top-tier pro dancers like **Derek Hough** and **Valentin Chmerkovskiy**.

## Comparison: Judges vs. Fans
While both judges and fans favor younger contestants and specific pro dancers, judges are significantly more responsive to professional entertainment backgrounds (Actors/TV Stars), whereas fans show higher idiosyncratic variation. Pro dancers like **Derek Hough** consistently provide a "boost" to their celebrities' success residuals, even after controlling for early window performance. Residualized analysis confirms that certain veterans significantly outperform their expected baseline, though these effects carry wider uncertainty for newer pros.
