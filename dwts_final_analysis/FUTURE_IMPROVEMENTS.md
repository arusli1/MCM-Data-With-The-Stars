# Future Improvements: Increasing Prediction Accuracy

The current model provides a baseline accuracy of ~23% for picking winners and ~50% for picking the Top 3. To reach "Expert Level" accuracy (>75% Top 3), the following roadmap is recommended.

## 1. Feature Engineering: Beyond Static Traits
The primary limitation is the "Talent Gap." The following features would act as proxies for a celebrity's innate ability:
*   **Performance Background**: A "Previous Experience" score (0-5) based on musical theater, cheerleading, or music video history.
*   **Fame Tiering**: Instead of just "Industry," use log-scaled **Instagram/TikTok follower counts** to capture true fan reach.
*   **Fan Growth**: Measure the *rate of growth* in social media mentions during the first 48 hours after the premiere.

## 2. Dynamic Forecasting (The "Week 3" Model)
Transition from a Pre-Season model to a Live-Season model:
*   **Talent Benchmarking**: Include the **Average Judge Score from Week 1 & 2**. Research shows this explains over 50% of the final variance.
*   **Fan Trajectory**: Include the "Fan Share" from the opening week. Fans tend to be loyal; a high share in Week 1 is the best predictor of a high share in Week 10.

## 3. Algorithmic Upgrades
*   **Learning-to-Rank (LTR)**: Move from standard regression to ranking algorithms like `XGBRanker` or `LightGBM` (Lambdarank). These prioritize predicting the correct order over the correct value.
*   **Ordinal Regression**: Use models specifically designed for ordinal data (where 1 < 2 < 3) to better handle the "Placement" target.

## 4. Interaction Effects
*   **Stylistic Synergy**: Create an interaction feature between "Pro Dancer Style" and "Celebrity Industry." 
    *   *Example: Do Latin-specialist Pros perform better with Athletes than with Singers?*
*   **Age/Partner Interaction**: Older celebrities benefit significantly more from "Veteran" partners than from "Rookie" pros.

## Summary of Impact
| Improvement | Expected MSE Reduction | Complexity |
| :--- | :--- | :--- |
| Early Scores (Talent) | 50% - 60% | Low (Data available) |
| Social Media (Fame) | 15% - 20% | Medium (API required) |
| Experience Flags | 10% - 15% | High (Manual research) |
| LTR Algorithm | 5% - 10% | Medium (Coding) |
