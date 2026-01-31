Constraint-based feasible set (by scoring regime)
====================================================

Let A_w be active contestants in week w. Let s_{i,w} be fan-share (sum to 1).
Let J_{i,w} be judge totals and j_pct_{i,w} = J_{i,w} / sum_{k in A_w} J_{k,w}.

Simplex and bounds (all regimes):
- s_{i,w} >= 0 for i in A_w
- sum_{i in A_w} s_{i,w} = 1

Percent regime (Seasons 3-27):
- Combined score: C_{i,w} = j_pct_{i,w} + s_{i,w}
- For each eliminated e and survivor j:
  C_{e,w} <= C_{j,w} - epsilon

Rank regime (Seasons 1-2):
- Judge rank rJ_{i,w} is descending rank of J_{i,w}
- Fan rank rF_{i,w} is descending rank of s_{i,w}
- Combined rank R_{i,w} = rJ_{i,w} + rF_{i,w}
- Eliminated set has the largest R_{i,w}

Bottom-two regime (Seasons 28+):
- Same R_{i,w}
- If k=1: eliminated must be in bottom-2 of R_{i,w} and have the lower J_{i,w}
- If k>1: eliminated are the bottom-k by R_{i,w}
