\section{Problem 1: Estimating Fan Votes}

\subsection{Problem Framing, Constraints, and Assumptions}

Our goal is to estimate weekly fan votes for each contestant across all
seasons and scoring regimes. The key difficulty is identifiability: even with
complete judge scores and elimination outcomes, the set of vote distributions
that could have produced those eliminations is large. This is especially true
in rank-based regimes, where only ordering matters. As a result, a model can
match eliminations perfectly while vote estimates remain uncertain.

Assumptions: (i) judge scores are exogenous performance signals, (ii) fan-share
evolves multiplicatively with relative performance, (iii) eliminations follow
the weekly count inferred from score availability, (iv) tied placements are
interchangeable for weekly matching, and (v) one judge influence parameter
applies per season.

Let $A_w$ be active contestants in week $w$ and $s_{i,w}$ be fan-share with
$s_{i,w} \ge 0$ and $\sum_{i \in A_w} s_{i,w} = 1$. The regime constraints are:

\textit{Percent regime (Seasons 3--27):}
$C_{i,w} = j\_pct_{i,w} + s_{i,w}$ and for each eliminated $e$ and survivor $j$,
$C_{e,w} \le C_{j,w} - \epsilon$.

\textit{Rank regime (Seasons 1--2):}
$R_{i,w} = rJ_{i,w} + rF_{i,w}$ where $rJ$ is judge rank and $rF$ is fan rank;
eliminated contestants have the largest $R_{i,w}$.

\textit{Bottom-two regime (Seasons 28+):}
the eliminated contestant must be in the bottom two of $R_{i,w}$ and have the
lower judge score (or bottom-$k$ for multi-elimination weeks).

Formal constraints and regime summaries are provided in
`AR-Problem1-Base/final_results/constraint_equations.md` and
`constraints_regime_summary.csv`.

\subsection{Base Model Design and Motivation}

We estimate vote shares with a simple weekly update model. Each week has a
fan-share vector $s$ (one value per contestant) that is nonnegative and sums
to 1. The idea is: if a contestant scores above average with the judges that
week, their share should increase; if they score below average, it should
decrease. We implement this by multiplying each share by
$\\exp(\\alpha \\cdot Jz \\cdot \\text{JUDGE\\_SCALE})$, then renormalizing so
all shares again sum to 1 and clipping to $[\\text{MIN\\_SHARE},
\\text{MAX\\_SHARE}]$ for realism.

At the start of each season we draw candidate initial shares $s_0$ from a
Dirichlet prior. This is a natural choice because fan shares are nonnegative
and must sum to 1, and the Dirichlet distribution is the standard way to place
a prior over probability vectors. Because we lack a robust external popularity
signal, we use final placement as a weak proxy for baseline popularity (there
is a loose correlation between overall fan support and finishing position).
The prior is kept intentionally weak so weekly performance can override it.
Concretely, we set
\[
  \mathbf{s}_0 \sim \text{Dirichlet}(\boldsymbol{\eta}), \quad
  \eta_i = \left(1 + \kappa \cdot \tilde{p}_i\right)\tau, \quad
  \tilde{p}_i = \frac{1/\text{placement}_i}{\sum_j (1/\text{placement}_j)}.
\]
Here $\kappa$ and $\tau$ are hyperparameters: $\kappa$ controls how strongly
placement shapes the prior (scale), and $\tau$ is the concentration. Larger
$\tau$ makes the prior more tightly clustered around $\tilde{p}$, while
smaller $\tau$ makes it flatter and closer to uniform.
This yields a reasonable starting point when no external popularity data are
available, while keeping the prior weak enough for the dynamics to adjust.

This design is motivated by: (i) multiplicative dynamics that preserve the
simplex and encode proportional changes, (ii) a clean separation of judge
influence via $\alpha$ and JUDGE\_SCALE, and (iii) regime-specific elimination
rules that match historical formats. The per-season objective balances
placement accuracy, elimination consistency, entropy regularization, and a
quadratic penalty on $\alpha$:
\[
  \text{loss} =
  w_r \cdot \text{MSE}_\text{rank} +
  w_e \cdot (1-\text{match}) -
  w_h \cdot H(s_0) +
  \lambda \alpha^2.
\]
We include entropy regularization to discourage degenerate $s_0$ vectors where
a single contestant dominates the initial fan share. The quadratic penalty on
$\alpha$ is standard $L_2$ shrinkage to prevent unrealistically strong judge
influence and to stabilize the non-convex search.
We set these weights as fixed hyperparameters: $w_r=1$, $w_e=3$,
$w_h=0.02$, and $\lambda=0$ to emphasize elimination consistency while
preventing degenerate $s_0$. The optimization is over $s_0$ and $\alpha$
(the fan-share vector and judge influence), not over these weights.

\textbf{How the loss is used.} For each candidate $(\alpha, s_0)$ we
simulate the season forward: the weekly update rule and regime-specific
elimination rules yield predicted elimination weeks and, after the final
week, a predicted placement order. We then compare these to the
\textit{observed} outcomes: (i) \textit{weekly elimination match}---for
each week, we check whether the set of contestants the model predicts as
eliminated that week matches the set implied by the true placement order
(and elimination schedule); (ii) \textit{placement accuracy}---we compute
the mean squared error between predicted and true placement ranks. The
loss is a weighted sum of that rank MSE, the fraction of weeks where
elimination predictions are wrong, minus entropy of $s_0$, plus the
$\alpha$ penalty. Minimizing this loss selects the $(\alpha, s_0)$ whose
simulated eliminations and final order best match the actual data; the
associated trajectory of weekly fan shares is our estimate.

Concretely, for each season we do a grid search over $\alpha$ and sample
many candidate $s_0$ vectors from a Dirichlet prior. Each $(\alpha, s_0)$
pair is simulated forward to produce weekly shares and predicted eliminations,
then scored by the loss. We keep the best candidate and refine $s_0$ locally
by small perturbations, repeating to reduce the objective. This is the
optimization that yields the final fan-share trajectories.
The Dirichlet hyperparameters $\kappa$ and $\tau$ are fixed; we do not learn
them. They define the prior used to generate candidate $s_0$ vectors, and the
optimization selects the best $s_0$ among those candidates.

\subsection{Results: Consistency and Uncertainty}

\textbf{Consistency with eliminations.} Figure~\ref{fig:consistency} summarizes
model fit across seasons. The top panel shows the \textit{weekly elimination
match rate}: for each week, we check whether the set of contestants the model
predicts as eliminated that week matches the set implied by the true placement
and schedule; the curve is the point estimate and the band is the p10--p90
range across bootstrap runs. The bottom panel shows \textit{rank error (MSE)}:
the mean squared error between predicted and true final placement ranks over
contestants in that season (again with a p10--p90 bootstrap band). When weekly
elim match is perfect, placement is determined by elimination order except for
finalists; so any nonzero rank MSE in those seasons reflects only errors in
finalist ranking (and within-week ordering when multiple are eliminated the
same week). Overall, the model achieves a 100\% weekly elimination match rate
across seasons (weighted by season size) and a small rank MSE (about 0.11),
indicating that simulated eliminations align well with the data and that most
remaining error is in the ordering of the top finishers.

\textbf{How we calculate uncertainty.} We quantify uncertainty in estimated
fan shares using two methods. (i) \textit{Bootstrap uncertainty}: we run the
LPSSM fitting procedure multiple times per season; each run uses different
random draws for the initial fan-share candidates (Dirichlet) and for the
refinement steps, so the fitted $(\alpha, s_0)$ and thus the trajectories
differ across runs. For each contestant and week we take the distribution of
shares across these runs and report the 10th, 50th, and 90th percentiles
(p10, p50, p90). This captures variability due to the stochastic search over
$s_0$ and $\alpha$. (ii) \textit{Constraint-based uncertainty}: we
do not use the LPSSM here; instead, for each week we sample fan-share vectors
on the simplex that satisfy the regime-specific elimination constraints (the
eliminated contestant must have the worst combined score, within a small
slack). We then compute p10, p50, and p90 of each contestant's share from
these feasible samples. This reflects the breadth of vote distributions that
are \textit{consistent with observed eliminations} alone, without assuming
the LPSSM dynamics. Full numeric bands are in
`base_inferred_shares_uncertainty.csv` (bootstrap) and
`constraints_shares_uncertainty.csv` (constraint-based).

\textbf{Uncertainty plots (Seasons 1, 27, 34).} We visualize the 90th
percentile (p90) of fan share per contestant and week as heatmaps: contestants
on the vertical axis, weeks on the horizontal axis, and color intensity
proportional to p90 (darker means a higher upper bound and thus broader
uncertainty). Seasons 1, 27, and 34 span the three voting regimes (rank,
percent, and bottom-two). The bootstrap p90 heatmaps for these seasons show
how uncertainty varies by contestant and week under the LPSSM; the
constraint-based p90 heatmap for Season~27 (percent regime) shows the range of
fan shares consistent with the observed eliminations alone. The figures
indicate that even when elimination consistency is high, fan-share estimates
remain uncertain---particularly in rank-based regimes, where many vote
distributions can produce the same elimination order. Bootstrap heatmaps:
`uncertainty_p90_season_1.png`, `uncertainty_p90_season_27.png`,
`uncertainty_p90_season_34.png`; constraint-based example:
`constraints_uncertainty_p90_season_27.png`.

The point estimates of weekly fan shares are provided in
`base_inferred_shares.csv` and the required submission file
`Data/estimate_votes.csv`. A visualization of estimated vote shares
over the season for a single season (e.g. Season 34) is given in
`vote_shares_season_34_stacked.png` (stacked area) and
`vote_shares_season_34_lines.png` (top contestants with uncertainty bands).

\subsection{Discussion}

The LPSSM achieves high elimination consistency while producing smooth,
interpretable vote trajectories. The constraint-based analysis shows why
accurate elimination matching does not imply uniquely determined vote shares,
especially in rank regimes where many feasible vote vectors satisfy the same
outcomes.

\section{Problem 2: Comparing Voting Systems and Controversy}

We use the same inputs each season (judge scores from the data and estimated
fan-share trajectories from Problem~1) and compare two \textit{combination
rules}: (i) \textbf{rank combination}---combined score = judge rank + fan
rank (lower is better); eliminate the contestant(s) with the largest combined
rank; (ii) \textbf{percent combination}---combined score = judge share of
total points + fan share; eliminate the contestant(s) with the lowest combined
score. We do not change the show’s actual rules by season; we ask: ``If we had
used rank (or percent) every week for this season, what would the elimination
order and final placement have been?'' For bottom-two seasons (28+), we also
compare \textbf{judge-save} (in bottom-two weeks, the contestant with the lower
judge score is eliminated) vs.\ fan-decide (the one with the lower fan share
is eliminated).

\subsection{Rank vs.\ Percent: How Different Are the Outcomes?}

Figure~\ref{fig:p2-rank-percent-enhanced} shows three metrics of disagreement
between rank and percent methods across all seasons: (a) Kendall $\tau$ between
final rankings (lower means more disagreement), (b) mean placement displacement
(average change in final placement), and (c) fraction of weeks where different
contestants would be eliminated. Bars are colored by the show's natural regime
for that season (rank s1--2, percent s3--27, bottom-two s28+).

Key observations: In most seasons the two methods produce similar outcomes
(low displacement, high $\tau$), but in several seasons they diverge
substantially. For example, Season~3 shows high disagreement (Kendall
$\tau = 0.15$, mean displacement $= 1.27$) and the winner differs (Mario Lopez
under rank vs.\ Emmitt Smith under percent). Season~27 also shows a winner
change (Evanna Lynch under rank vs.\ Bobby Bones under percent). Across all
seasons, the average Kendall $\tau$ is 0.92 (high agreement) but the average
displacement is 0.67 placements, indicating that while most contestants end
up in similar positions, some experience significant shifts.

\subsection{Which Method Favors Fans vs.\ Judges?}

To determine which combination method gives more weight to fan votes vs.\ judge
scores, we run two counterfactual experiments per season: (i) replace fan
shares with uniform support (all contestants get equal fan votes) and measure
how much final placements change; (ii) replace judge scores with uniform scores
(all contestants get equal judge points) and measure placement change. Larger
displacement when removing an input indicates that input has more influence on
outcomes.

Figure~\ref{fig:p2-which-favors-fans} shows the \textit{fan favoritism} metric
for each season and method: fan influence minus judge influence (positive means
fans matter more, negative means judges matter more). Under the \textbf{rank
method} (panel a), most seasons show positive values, indicating fans have
slightly more influence than judges. Under the \textbf{percent method}
(panel b), most seasons show negative values, indicating judges have
substantially more influence than fans.

Figure~\ref{fig:p2-aggregate} quantifies this across all seasons: averaging
over all 34 seasons, the rank method gives fans an influence score of 2.85 vs.\
judges 2.64 (fans matter slightly more), while the percent method gives fans
0.22 vs.\ judges 1.14 (judges matter much more). This is a key finding:
\textbf{the rank combination method favors fan votes more than the percent
method does}.

Why does this happen? In the rank method, fan ranks and judge ranks are added
with equal weight (both contribute 1 rank unit). In the percent method, judge
scores are converted to a share of total points, which tends to be more
concentrated (high variance) than fan shares, giving judges disproportionate
influence. Additionally, in our estimated fan-share trajectories, fan support
often spreads across multiple contestants, while judge scores can vary more
dramatically week-to-week, amplifying their impact in the percent rule.

Figure~\ref{fig:p2-scatter} shows this relationship as a scatter plot: for each
season, we plot judge influence vs.\ fan influence under each method. Points
above the diagonal indicate fan dominance; points below indicate judge
dominance. The rank method (panel a) shows most points clustered near or above
the diagonal, while the percent method (panel b) shows most points well below
the diagonal. Figure~\ref{fig:p2-heatmap} displays fan weight (normalized
influence) as a heatmap: green indicates fans matter more, red indicates judges
matter more. The percent row is predominantly red (judge-dominated), while the
rank row shows more green (more balanced or fan-leaning).

\subsection{Controversy Cases and Judge-Save}

We define ``controversy'' as disagreement between judge scores and final
placement: \textit{controversy score} = $|$judge percentile $-$ placement
percentile$|$ (judge percentile 1 = best with judges, 0 = worst; placement
percentile 1 = winner, 0 = last). Contestants with controversy score $\ge 0.36$
are classified as controversial; this set includes the four often-cited examples
(Jerry Rice s2, Billy Ray Cyrus s4, Bristol Palin s11, Bobby Bones s27) and 16
others, for 20 total. We run the 2$\times$2 comparison (rank vs.\ percent
$\times$ judge-save vs.\ no) for each of these contestants.

\textbf{Would the choice of method (rank vs.\ percent) have led to the same
result for each?} No. For only 7 of 20 controversial contestants would final
placement be the same under rank vs.\ percent; for 10 of 20, elimination week
would be the same. In 4 cases (e.g., Season~27), the \textit{winner} would
differ: under rank (with judge-save) the winner would be Evanna Lynch, while
under percent the winner would be Bobby Bones. So the choice of combination
method would not have led to the same result for these contestants.

\textbf{How would including the judge-save rule impact the results?} Judge-save
(judges choose which of the bottom two to eliminate) applies only in
bottom-two seasons (28+). Among the 20 classified controversial contestants,
only one is in a bottom-two season (Whitney Leavitt s34). None of the classified
contestants would have been ``saved'' by judge-save (i.e., would have been
eliminated by fan vote but kept by judges). Across all bottom-two weeks (s28+),
judge-save changed who was eliminated in 6 of 44 such weeks; Figure~\ref{fig:p2-judge-save}
shows the count by season. Including judge-save can therefore change who goes
home each week when the bottom two disagree between judges and fans; for
controversial contestants in s28+, it can extend or shorten their run depending
on whether judges repeatedly save them or send them home.

Table~\ref{tab:p2-controversy} gives the four named examples; the full 2$\times$2
results for all 20 classified contestants are in \texttt{problem2b\_2x2\_scenarios.csv}.

\begin{table}[h]
\centering
\caption{Controversy cases: elimination week, final placement, and winner under rank (with/without judge-save) and percent.}
\label{tab:p2-controversy}
\begin{tabular}{llccccccccc}
\toprule
Contestant & Season & $e_w$ (rank+save) & $e_w$ (rank) & $e_w$ (pct) & Pl.\ (rank+save) & Pl.\ (rank) & Pl.\ (pct) & Winner (rank+save) & Winner (rank) & Winner (pct) \\
\midrule
Jerry Rice & 2 & 9 & 9 & 9 & 2 & 2 & 2 & Drew Lachey & Drew Lachey & Drew Lachey \\
Billy Ray Cyrus & 4 & 7 & 7 & 9 & 7 & 7 & 5 & Apolo Anton Ohno & Apolo Anton Ohno & Apolo Anton Ohno \\
Bristol Palin & 11 & 7 & 7 & 11 & 7 & 7 & 3 & Jennifer Grey & Jennifer Grey & Jennifer Grey \\
Bobby Bones & 27 & 10 & 10 & 10 & 3 & 3 & 1 & Evanna Lynch & Evanna Lynch & Bobby Bones \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Implementation Details}

Scripts in \texttt{AR-Problem2/} load judge scores and Problem~1 fan-share
estimates, then run forward simulation under each combination rule and (for
s28+) under judge-save vs.\ fan-decide. We compute Kendall $\tau$, mean
displacement, fraction of weeks where eliminations differ, and winner/finalist
agreement. For ``fan influence'' we set fan share to $1/n$ each week and
compare placements to baseline; for ``judge influence'' we set judge scores
equal each week.

Enhanced figures are generated by \texttt{AR-Problem2/plot\_problem2\_enhanced.py}:
\begin{itemize}
\item \texttt{problem2a\_rank\_vs\_percent\_enhanced.png|.pdf} --- 3-panel comparison
  of rank vs.\ percent outcomes (use \verb|\label{fig:p2-rank-percent-enhanced}|)
\item \texttt{problem2a\_which\_method\_favors\_fans.png|.pdf} --- fan favoritism
  by method and season (use \verb|\label{fig:p2-which-favors-fans}|)
\item \texttt{problem2a\_aggregate\_comparison.png|.pdf} --- average fan vs.\
  judge influence by method (use \verb|\label{fig:p2-aggregate}|)
\item \texttt{problem2a\_fan\_vs\_judge\_scatter.png|.pdf} --- scatter plot of
  fan vs.\ judge influence (use \verb|\label{fig:p2-scatter}|)
\item \texttt{problem2a\_method\_comparison\_heatmap.png|.pdf} --- heatmap of
  fan weight by method and season (use \verb|\label{fig:p2-heatmap}|)
\item \texttt{problem2b\_judge\_save\_impact.png|.pdf} --- judge-save impact
  by season (use \verb|\label{fig:p2-judge-save}|)
\end{itemize}

The controversy 2$\times$2 scenarios and judge-save-by-week details are in
\texttt{problem2b\_2x2\_scenarios.csv} and
\texttt{problem2b\_judge\_save\_impact\_by\_week.csv}.
