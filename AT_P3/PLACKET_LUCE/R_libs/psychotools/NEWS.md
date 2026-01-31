# psychotools 0.7-5

* Add packages utilized in `demo(package = "psychotools")` to `DESCRIPTION/Suggests`.
  Also use `hcl.colors()` instead of package `colorspace`.

* Don't test visualization of `likert` objects that can be coerced from `itemresp`
  objects because its `plot()` method currently throws a warning.


# psychotools 0.7-4

* Return the matched function call as `$call` in IRT functions `raschmodel()`,
  `rsmodel()`, `pcmodel()`, `nplmodel()`, `gpcmodel()` (suggested by Rainer W.
  Alexandrowicz).


# psychotools 0.7-3

* Fixed calculation of person parameter covariance matrix estimate (and thus
  confidence intervals) in `personpar()` methods for Rasch models (reported by
  Rainer W. Alexandrowicz).
  
* Fixed bug in `threshpar()` method of `pcmodel` objects fitted to binary items
  reported by Rainer W. Alexandrowicz).

* Fixed bug in `pcmodel()` which failed when the reference category was a
  null category (without observations). Now the first category with observations
  is chosen as the reference category (reported by Rainer W. Alexandrowicz).

* Improved `raschmodel()`, `rsmodel()`, and `pcmodel()` so that an `NA` variance-covariance
  matrix is returned (with a warning) if the Hessian cannot be inverted instead
  of stopping with a technical error message (suggested by Rainer W. Alexandrowicz).

* Fixed names of `"mscale"` attribute in `itemresp` variable in `MathExam14W` data,
  now consistent with column names.


# psychotools 0.7-2

* Bug fix in `discrpar()` method for `nplmodel(..., type = "1PL")` so that the
  variance covariance matrix is computed correctly.

* Documentation enhancements in the mathematical notation (suggested by Kurt
  Hornik) and in the description of the 1PL/Rasch model in `nplmodel()`
  (suggested by Feng Lin).


# psychotools 0.7-1

* The function (and resulting class) for fitting n-PL type
  parametric logistic IRT models has been renamed from `plmodel()` to
  `nplmodel()`. For now, a copy called `plmodel()` is also preserved (but
  this also returns an object of class `"nplmodel"`).

* Fix logicals of length > 1 in some `if()` constructs.

* Accept `.replicates` for user-provided model specifications in `mptmodel()`.


# psychotools 0.7-0

* Changed default constant anchor selection method in `anchor()` and `anchortest()`
  from MPT to Gini based on Strobl et al. (2021, Applied Psychological
  Measurement). The Gini-based anchor selection is simpler because it is based
  on single anchors while at the same time performing very well under DIF.

* The `print()` method for `anchortest()` now just displays the anchor item(s) and
  the final DIF tests while the `summary()` method displays the full information
  including anchored item parameters etc. (rather than vice versa).
  
* The `print()` and `summary()` methods for `anchor()` and `anchortest()` now display
  item labels (rather than indexes) for the selected anchor item(s) and for the
  full vector of criterion values.

* Added the `"proCNI"` multinomial processing tree specification in `mptspec()` for
  `mptmodel()`. This provides the CNI model of moral dilemma judgment for
  proscriptive norms.


# psychotools 0.6-1

* New `vignette("toolbox-simulation", package = "psychotools")` on how to conduct
  simulation studies investigating the performance of score-based tests of
  measurement invariance of IRT models. Accompanies the PsyArXiv preprint mentioned
  below.


# psychotools 0.6-0

* New `anchor()` selection strategy using inequality-based alignment, either
  based on the Gini index or the component loss function (CLF). Thus, also directly
  available in `anchortest()`. Improved `print()` and `plot()` methods.

* Changed method to invert Hessian for `raschmodel()`, `rsmodel()`, and `pcmodel()`
  from `qr.solve()` to `chol2inv(chol())`.

* Added two demos on how to conduct simulation studies for score-based test
  of measurement invariance.

* All IRT models now have a function to simulate IRT data, see `rrm()`, `rrsm()`,
  `rpcm()`, `rpl()`, and `rgpcm()`.

* An accompanying new PsyArXiv Preprint "An R Toolbox for Score-Based
  Measurement Invariance Tests in IRT Models" by Lennart Schneider, Carolin
  Strobl, Achim Zeileis, and Rudolf Debelak is available at
  <https://doi.org/10.31234/osf.io/r9w34>


# psychotools 0.5-1

* Added "IGNORE_RDIFF" flags in some examples in order to avoid showing
  diffs due to small numeric deviations in some checks (especially on CRAN).


# psychotools 0.5-0

* Infrastructure for IRT modeling in the unified `psychotools` framework is
  extended by marginal maximum likelihood (MML) estimation of generalized partial
  credit models and parametric logistic models, respectively. The corresponding
  fitting functions (see below for details) call `mirt()` or `multipleGroup()` from
  the `mirt` package but return objects for which all standard extractor methods
  (item parameters, person parameters, etc.) and visualization methods (item
  response curves, parameter profiles, person-item maps, etc.) are available.
  
* The new `gpcmodel()` function interfaces `mirt` (see above) and fits (generalized)
  partial credit models (GPCMs) by MML.
  
* The new `plmodel()` function interfaces `mirt` (see above) and fits various
  parametric IRT logistic models using MML: 1PL (Rasch), 2PL, 3PL, 3PLu, and
  4PL.

* New functions and eponymous classes `guesspar()`, and `upperpar()` to
  extract/represent so-called guessing parameters and upper asymptote parameters
  of IRT models.

* `personpar()` now distinguishes between parameters of the assumed person ability
  distribution (`personwise = FALSE`) and the individual person parameters for
  each person/subject in the underlying data set (`personwise = TRUE`). In the CML
  case, the latter simply computes the raw score for each person and then extracts
  the corresponding person parameter. In the MML case, this necessitates
  (numerically) integrating out the individual person parameters (also known as
  factor scores or latent trait estimates) based on the underlying normal
  distribution.

* Added new data set `ConspiracistBeliefs2016` from the Open Source Psychometrics
  Project (2016).

* Added new simulated data set `Sim3PL` for fitting dichotomous IRT models,
  especially the 3PL and 3PLu.


# psychotools 0.4-3

* Conditionally register all `estfun()` and `bread()` S3 methods for model
  objects, provided that the `sandwich` package is attached.

* Added native routine registration for `esf.c`.

* Use R version of `elementary_symmetric_functions()` by default on Win/i386
  due to small numeric differences on that platform.

* The `estfun()` method for `btmodel` objects always computed the scores
  with the last object for the reference category - even if a different
  `ref=` was specified in the model. (Thanks to Heather Turner for pointing
  out the problem.)

* The `itempar()` method for `btmodel` objects miscomputed the variance
  covariance matrix (unless the first object was used as the ref when
  estimating the model). (Thanks to Heather Turner for pointing out the
  problem.)


# psychotools 0.4-2

* Added new data set `PairClustering` from Klauer (2006).

* Fixed replication code in example of `StereotypeThreat` (reported by
  Ed Merkle).

* Basil Abou El-Komboz changed his name to Basil Komboz.


# psychotools 0.4-1

* Properly imported `grDevices` and `utils` in `NAMESPACE`.

* Added new item response data set `MathExam14W` with esponses of 729
  students to 13 items in a written exam of introductory mathematics
  along with several covariates. 


# psychotools 0.4-0

* New function `mptmodel()` and corresponding extractor functions for fitting
  multinomial processing tree (MPT) models. These functions are somewhat
  experimental, and their user interface might change in future releases.

* Bug fix in `itempar()` method for `raschmodel` objects if `alias = FALSE`.
  In the previous version the methods had an erroneous trailing `NA`.

* Improved item names labeling in `plot()` method for `itemresp` objects
  to conform with `regionplot()` function for IRT models.

* `mscale<-()` method for `itemresp` has been improved so that categories
  can be easily collapsed (e.g., dichotomized).


# psychotools 0.3-0

* Infrastructure for IRT modeling in `psychotools` is greatly enhanced.
  Therefore the main modeling functions are now called `raschmodel()`
  for Rasch models, `rsmodel()` for rating scale models, `pcmodel()`
  for partial credit models, and `btmodel()` for Bradley-Terry models.
  The old `*.fit()` functions from previous versions of the package still
  exist but now internally call the new `*model()` functions. Also, the
  classes returned have the same names as the `*model` functions.

* A unified visualization framework for fitted IRT models has been added:
  For all types of models (Rasch, RSM, PCM) one can visualize profiles
  of the item parameters, regions for the most likely response, item
  or category characteristic curves, item information, and person-item
  plots. All of these rely on the unified framework for extracting
  parameters and predictions (see below).

* New functions and eponymous classes `itempar()`, `threshpar()`, and
  `discrpar()` to extract/represent item, threshold, and discrimination
  parameters of item response models. Methods for the IRT models (Rasch,
  RSM, PCM) are provided. In addition, several methods for standard generic
  functions (`print()`, `coef()`, `vcov()`) are available.

* The `worth()` generic now internally calls the methods for `itempar()`.

* Estimation of person parameters for a given item response model is
  now available via the generic function `personpar()`. Specific methods for
  Rasch, rating scale and partial credit models allow the estimatation of
  person parameters via joint maximum likelihood estimation. Methods for
  standard generic functions (`print()`, `coef()`, `vcov()`) are
  available for the resulting objects of class `personpar`.

* `predict()` methods for Rasch, rating scale and partial credit models
  have been added. For a given fitted model object, these can be 
  used to predict various types of response probabilities or actual 
  reponses.

* New functions `anchor()` and `anchortest()` provide a variety of anchor 
  methods for the detection of uniform differential item functioning 
  (DIF) between two pre-specified groups in the Rasch model. To test 
  for DIF, the itemwise Wald test is implemented.

* `itemresp()` is the class constructor for responses of n subjects
  to k items which can be polytomous and have different measurement
  scales. A wide range of methods to standard generics is provided
  as well as to generics created for the `paircomp` class. Thus,
  features can be easily extracted/replaced, summaries/visualizations
  can be produced, subsetting/merging/etc. is facilitated.

* The handling of argument `ref` when producing a region plot (previously
  called effect plot) was changed. Whereas in the previous implementation,
  the restriction specified in this argument was applied to the cumulative
  absolute item threshold parameters, it now is applied to the absolute
  item threshold parameters.    

* A bug occuring in `pcmodel()` when null categories are present and 
  `nullcats = "keep"` was fixed. (Thanks to Oliver Prosperi for reporting
  this.)

* The processing of the minimal category zero in the function `rsmodel()` 
  was changed. Only if for all items, the minimal category is above zero,
  downcoding takes place. Otherwise, the missing minimal categories are 
  treated as not observed, i.e., with a frequency of zero.


# psychotools 0.2-0

* Major update with new model fitting functions (partial credit and
  rating scale model) and improved infrastructure for conditional
  maximum likelihood estimation (C implementation of elementary
  symmetric functions).
  
* Partial credit models (PCMs) can be fitted with the function
  `PCModel.fit()`. The interface and return value is similar to that
  of `RaschModel.fit()`.

* Rating scale models (RSMs) can be fitted with the function
  `RSModel.fit()`. The interface and return value is similar to that
  of `RaschModel.fit()` and `PCModel.fit()`.

* The function `elementary_symmetric_functions()` for computing ESFs
  is extended and now part of the exported user interface. The
  R implementation for binary items up to order 2 is complemented
  by a C implementation for both binary and polytomous items
  up to order 1.

* Due to numerical instabilities in the coefficients and standard 
  errors between different architectures, the optimization method
  for `Rasch`/`RSModel`/`PCModel.fit()` was changed from `nlm(...)` to 
  `optim(..., method = "BFGS")`. Consequently, the arguments `reltol`
  and `maxit` are used now instead of `gradtol` and `iterlim`. For
  backward compatibility `RaschModel.fit()` still supports the old
  arguments but might cease to do so in future releases.


# psychotools 0.1-4

* Added `YouthGratitude` data from Froh, Fan, Emmons, Bono, Huebner, Watkins
  (2011, Psychological Assessment), provided by Jeff Froh and Jinyan Fan. Some approximate
  replication code is provided in the examples (the parts depending on
  `lavaan` are in `\dontrun`).


# psychotools 0.1-3

* Fully exported `elementary_symmetric_functions()`. (An extended C
  implementation is under development and will be included in
  future releases.)


# psychotools 0.1-2

* Support of non-integer weights in `btReg.fit()`. To facilitate this,
  `summary.paircomp()` gained a weights argument so that optionally
  the weights are aggregated instead of observations counted.  

* Actually pass on `nlm()` arguments from `RaschModel.fit()`. Also
  support `iterlim = 0`, i.e., set up model at pre-specified parameters.

* Added `StereotypeThreat` data from Wicherts, Conor, Hessen (2005,
  Journal of Personality and Social Psychology), provided by Jelte M. Wicherts.
  Replication code is provided in the examples (the parts depending on
  `lavaan` are in `\dontrun`).


# psychotools 0.1-1

* New `psychotools` package containing all "base" infrastructure
  previously contained in `psychotree`. This is in order to provide
  both methods and data that can be reused by `psychotree` and
  the new package `psychomix` (as well as potentially further packages).
  
* Classes: `paircomp` and associated methods.

* Models: `btReg.fit()` and `RaschModel.fit()` and associated methods.

* Data: `Firstnames`, `GermanParties2009`, `Soundquality` (previoulsy in
  `psychotree`) and `VerbalAggression` (new data, contained in other
  formatting in `difR` as `verbal` and `lme4` as `VerbAgg`).
