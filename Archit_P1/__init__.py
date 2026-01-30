"""Archit-P1 package (analysis code for MCM Problem C).

Primary module:
- `bayes_fan_votes.py`: Bayesian softmax regression model that infers fan vote share
  from judges' scores + elimination outcomes.
"""

from .bayes_fan_votes import (  # noqa: F401
    make_df_long_from_problem_c,
    prepare_week_groups,
    build_model,
    fit_model,
    posterior_summaries,
    posterior_vote_shares,
    predict_vote_shares,
    evaluate,
)


