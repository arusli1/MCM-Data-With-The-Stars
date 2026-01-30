# MCM-Data-With-The-Stars

Brainstorming:
Assumptions
Judges evaluate only on technical execution (Factors Drive Outcome Model)
S28-34 is ordinal
Data Organization
S1-S2: Ordinal, S3-S27 Proportion, S28-S34 Ordinal + Bottom 2
Estimate Fan Votes (Consistency & Uncertainty)
Total Viewership Dataset (Wikipedia viewership all seasons)
Hierarchical turnout model & baseline scaling ratio model (ABC votes S33+34)
uncertainty here
Inverse optimization & regularization
Infer vote proportions from mathematical constraints
Pick the most reasonable using a regularizer (max entropy or smoothness since fans have similar feelings across weeks)
Testing regularization will allow for good uncertainty quantification
Bayesian (Latent fan support + ranking likelihoods)
latent popularity p which changes over weeks 
previous popularity + tiny performance + noise?
shares (votes) = softmax(p)
3 likelihoods depending on elimination regime
MAP estimation (or MCMC) (find best parameters to max posterior)
Posterior predictive check if match outcomes!
Uncertainty from bayesian distribution interval (also how much can it vary and still match outcome accurately)
Voting System Comparison
Forward season simulator (S28+ just pick lower judge total)
Sample s from bayesian and run analyses multiple times (uncertainty)
Apply rank + percentage to all → difference metrics:
Kendall tau distance (for 2 final ranking lists) 
Means displacement or final placements per contestant
Fraction of weeks where eliminations differ 
% difference in finalist set
difference in winner
Fan influence:
Replace fan votes by uniform support and recompute forward simulator 
Apply difference metrics again, more difference = more influence
Replace judge votes by uniform support and recompute forward simulator
Apply difference metrics again, more difference = less influence
Controversy C = quantified gap in ranks/shares, threshold/clustering to classify
Confirm listed people (Jerry Rice, Billy Ray, etc.) are in this list
Forward simulate 2x2 scenarios (share vs. ordinal, judge-save vs no)
Output elimination week, final placement, winner changing, # weeks being in bottom-two
Rule sensitivity analysis: delta placement, delta elim week, # weeks where judge-save actually save someone
Recommendation of rank vs shares and judge-save vs none:
2x2 simulations: compute metrics:
Fan influence
Less highly controversial outcomes (maybe good for revenue?)
Consulting approach (Revenue, Fan excitement)
Episode-level viewership/ratings vs. controversy/other
Google trends/social media volume/wiki pageviews 
What Factors Drives Outcome?
Success:
Survival model (probability surviving) → expected weeks → condition
(Extra) XGBoost/RandomForest to predict elimination week & compare placement. maybe sensitivity on inputs = factor impact, or SHAP!
Judge scores & Fan votes:
Two-channel hierarchical (why?) regression: use (mostly) same variables and compare weights
Proposed Improved System
??
