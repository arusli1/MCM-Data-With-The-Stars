# MCM 2026 Problem C: Fan Vote Estimation – Model Answers

## 1. Estimated fan votes consistent with eliminations?
- **Elimination match rate**: 90.02% of weeks where predicted eliminations match actual.
- **Placement MSE**: 0.5590 (mean squared error of predicted vs actual final placement).

## 2. Certainty in fan vote totals – same for each contestant/week?
- **No.** Certainty varies. Bootstrap 80% interval width (p90−p10): mean=0.0007, std=0.0040, min=0.0000, max=0.0918.
- Contestants with tighter constraints (e.g. clear elim) have narrower intervals; marginal cases wider.

## 3. Model logic
- Objective: minimize sum((s − q_iw)²) to stay closest to prior.
- Constraints: s ≥ 0, sum(s)=1, and regime elimination rules.
