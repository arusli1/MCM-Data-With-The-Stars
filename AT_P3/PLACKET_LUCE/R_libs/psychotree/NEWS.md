## psychotree 0.16-2

* Improve non-anchored links in manual pages (prompted by CRAN).


## psychotree 0.16-1

* Make sure that `mob()` from `partykit` (rather than `party`) is called by all
  tree functions.

* Improved version of `node_profileplot(..., what = ...)` which now allows to
  visualize not only `itempar()` but also `coef()`, `threshpar` and so on by
  setting `what = "item"` or `"coef"` or `"threshold"` etc.

* In previous versions the `gpcmtree(..., minsize = ...)` argument was ignored.
  Now it is passed on correctly to `mob_control()`.


## psychotree 0.16-0

* New functions `npltree()` and `gpcmtree()` with corresponding visualization functions
  for model-based recursive partitioning of parametric logistic IRT models
  (2PL, 3PL, 3PLu, 4PL, Rasch/1PL) and the generalized partial credit model, 
  respectively. The output of these functions might still change slightly in
  future versions.

* Make sure that the "raschtree" vignette can also be compiled even when the
  `stablelearner` package is not available.


## psychotree 0.15-4

* Updated the "raschtree" vignette (_Using the raschtree function for detecting
  differential item functioning in the Rasch model_). This vignette now also
  includes a section regarding stability assessment via the `stablelearner`
  package (>= 0.1-3).


## psychotree 0.15-3

* Added `IGNORE_RDIFF` flags in some examples in order to avoid showing
  diffs due to small numeric deviations in some checks (especially on CRAN).


## psychotree 0.15-2

* The method behind `mpttree()` has now been published by 
  Behavior Research Methods, see
  [doi:10.3758/s13428-017-0937-z](https://doi.org/10.3758/s13428-017-0937-z).

* The `predict()` method for `bttree` objects works again correctly for
  `type = "parameter"` (previously called `type = "worth"`) and `type = "rank"`.
  Thanks to Heather Turner for pointing out the problem.

* `node_btplot()` used to fail for `worth = FALSE` or `ref = <character>` but works
  now correctly again. Thanks to Heather Turner for pointing out the problem.
  Also `ref = <character>` now either matches to `<object>$labels` or (if not available)
  to `colnames(<coefficients>)` to facilitate usage with the `PlackettLuce`
  package.

* `node_regionplot()` now also works correctly on devices with a default white
  (rather than transparent) background such as `png()` or `jpeg()`.

* `node_*` panel functions gained `bg = "white"` argument to enable different
  background filling color.


## psychotree 0.15-1

* The recently added `mpttree()` function is now accompanied by a working
  paper, see `citation("psychotree")` or `?mpttree` for details.

* Properly imported `grDevices` in `NAMESPACE`.

* Basil Abou El-Komboz changed his name to Basil Komboz.


## psychotree 0.15-0

* New function `mpttree()` and visualization function for model-based
  recursive partitioning of multinomial processing tree (MPT) models. These
  functions are somewhat experimental, and their user interface might change
  in future releases.


## psychotree 0.14-0

* Internals of package `psychotree` completely reorganized to employ
  the recent additions and changes in `psychotools` (>= 0.3-0) and
  to use the new `partykit` (>= 0.2-0) implementation of the `mob()`
  function (rather than the old `party` implementation).

* The handling of argument `ref` when producing a region/effect
  plot was changed. Whereas in the previous implementation,
  the restriction specified in this argument was applied to
  the cumulative absolute item threshold parameters, it now is
  applied to the absolute item threshold parameters.

* New panel-generating visualization functions `node_profileplot()`
  and `node_regionplot()` have been added which replace the old 
  functions `node_raschplot()` and `node_effects()`. 


## psychotree 0.13-0

* New functionality for recursive partitioning of partial credit
  and rating scale models via functions `pctree()` and `rstree()`,
  respectively. These require package `psychotools` >= 0.2-0
  which provides the corresponding basic model fitting tools.


## psychotree 0.12-3

* Improved Depends/Imports/Suggests declarations in
  `DESCRIPTION`.


## psychotree 0.12-2

* Adjusted vignette "raschtree" according to the slightly 
  different results due to the new approximation 
  formulae for p values in the supLM test used in 
  `party::mob()` (>= 1.0.3).

* Improved the package's tests by adding reference output for
  the examples and vignette for comparison checks.


## psychotree 0.12-1

* Moved all base functionality to the new `psychotools` package
  in order to provide common infrastructure for both
  `psychotree` and `psychomix` as well as other packages.
  This affects the `paircomp` class and associated methods as well
  as the model fitters `btReg.fit()` and `RaschModel.fit()` and
  associated methods.

* The data sets `Firstnames`, `GermanParties2009`, and `Soundquality`
  have also been moved to `psychotools`.


## psychotree 0.11-1

* Enhanced `raschtree()` functionality, introduced in detail
  in the new `vignette("raschtree", package = "psychotree")`.

* Added subsample from SPIEGEL Studentenpisa (`SPISA`) quiz
  for students from Bavaria that answered the same set of
  questions.
  
* Added artificial data set `DIFSim` exhibiting differential
  item functioning, employed for illustrating Rasch trees.
  

## psychotree 0.9-1

* Added `raschtree()` function, based on simple `RaschModel()`
  (for binary 0/1 items).

* Some bug fixes (subsets with missing `drop = FALSE`) for `paircomp()`
  and `bttree()` with only two objects
  

## psychotree 0.9-0

* First CRAN release of the package accompanying the forthcoming
  paper _Accounting for Individual Differences in Bradley-Terry
  Models by Means of Recursive Partitioning_ by Strobl, Wickelmaier,
  and Zeileis, accepted for publication in _Journal of Educational
  and Behavioral Statistics_.
  
* Compared to earlier package versions available from R-Forge,
  this package does not depend on data/code from the `prefmod2`
  package. All required functionality (basic Bradley-Terry
  regression, paircomp class, data sets) are now contained in the
  `psychotree` package.
