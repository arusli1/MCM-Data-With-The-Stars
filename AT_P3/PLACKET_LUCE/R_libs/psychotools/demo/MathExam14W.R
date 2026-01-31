## packages --------------------------------------------------------------------

stopifnot(require("psychotools"), require("multcomp"), require("strucchange"),
  require("psychotree"), require("psychomix"))


## data ------------------------------------------------------------------------

library("psychotools")
data("MathExam14W", package = "psychotools")

## exclude extreme scorers
mex <- subset(MathExam14W, nsolved > 0 & nsolved < 13)

## two subgroups (first and second batch) with slightly different exercise templates
mex1 <- subset(mex, group == 1)
mex2 <- subset(mex, group == 2)

## solved: dichotomous incorrect (0) vs. correct (1)
## credits: polytomous not attempted (0) vs. attempted but not solved (1) vs. solved (2)
par(mfrow = c(2, 2))
plot(mex1$solved)
plot(mex1$credits)
plot(mex2$solved)
plot(mex2$credits)
par(mfrow = c(1, 1))

## dichotomous vs. polytomous --------------------------------------------------

## Rasch models
ram <- raschmodel(mex1$solved)
rsm <- rsmodel(mex1$credits)
pcm <- pcmodel(mex1$credits)

## person item maps
plot(ram, type = "piplot")
plot(rsm, type = "piplot")
plot(pcm, type = "piplot")

## (mean) item profiles similar
cols3 <- hcl.colors(3, "Dark 2")
plot(ram, type = "profile", what = "items", parg = list(ref = 8), col = cols3[1])
plot(rsm, type = "profile", what = "items", parg = list(ref = 8), col = cols3[2], add = TRUE)
plot(pcm, type = "profile", what = "items", parg = list(ref = 8), col = cols3[3], add = TRUE)
legend("topleft", legend = c("RM", "RSM", "PCM"), bty = "n", col = cols3, pch = 16)

## but handling of thresholds for attempting and solving different
plot(rsm, type = "profile", what = "threshold", parg = list(ref = 1), col = cols3[2], ylim = c(-1.5, 2.5))
plot(pcm, type = "profile", what = "threshold", parg = list(ref = 1), col = cols3[3], add = TRUE)
legend("topleft", legend = c("RSM", "PCM"), bty = "n", col = cols3[2:3], pch = 16)

rtprsm <- threshpar(rsm, ref = 1, relative = TRUE)
rtppcm <- threshpar(pcm, ref = 1, relative = TRUE)
print(cbind(coef(rtprsm, type = "matrix"), 
            coef(rtppcm, type = "matrix")), digits = 5)

## -> do not attempt to model distinction between "not attempting" and "attempting but not solving"
## -> focus on "solving" vs. "not solving" only


## DIF: group 1 vs. 2 ----------------------------------------------------------

## Rasch models
mr  <- raschmodel(mex $solved)
mr1 <- raschmodel(mex1$solved)
mr2 <- raschmodel(mex2$solved)

## global test statistics
library("multcomp")
library("strucchange")
c(
  "LR" = -2 * as.numeric(logLik(mr) - (logLik(mr1) + logLik(mr2))),
  "Wald" = summary(anchortest(solved ~ group, data = mex)$final_tests, test = Chisqtest())$test$SSH,
  "Score/LM" = unname(sctest(mr, order.by = mex$group, vcov = "info", functional = "LMuo")$statistic)
)

## itemwise visualization with different fixed anchorings
## anchor: item 1
rb <- hcl.colors(3, "Dark3")[c(3, 1)]
plot(mr1, parg = list(ref = 1), ref = FALSE, ylim = c(-2.6, 2.6), col = rb[1])
plot(mr2, parg = list(ref = 1), ref = FALSE, add = TRUE, col = rb[2])
legend("topleft", paste("Group", 1:2), pch = 19, col = rb, bty = "n")
## anchor: item 8
plot(mr1, parg = list(ref = 8), ref = FALSE, ylim = c(-1.6, 3.6), col = rb[1])
plot(mr2, parg = list(ref = 8), ref = FALSE, add = TRUE, col = rb[2])
legend("topleft", paste("Group", 1:2), pch = 19, col = rb, bty = "n")

## automatic anchor selection
set.seed(1)
mra <- anchortest(solved ~ group, data = mex, adjust = "single-step", class = "forward")
par(mar = c(5, 8, 4, 1))
plot(mra$final_tests, main = paste("Anchor items:", paste(mra$anchor_items, collapse = ", ")))
par(mar = c(5, 4, 4, 2))


## DIF: Ability (tests or nsolved) with group 1 only ---------------------------

## maxLM test (jittered)
set.seed(0)
sctest(mr1, order.by = mex1$tests + runif(nrow(mex1), -0.15, 0),
  vcov = "info", functional = "maxLM",
  plot = TRUE, xlab = "tests (jittered)", ylim = c(0, 40))

## maxLM test (ordinal)
set.seed(1)
mex1$otests <- cut(mex1$tests, breaks = c(0, 14:24, 26),
  ordered = TRUE, labels = c("<= 14", 15:24, ">= 25"))
sctest(mr1, order.by = mex1$otests,
  vcov = "info", functional = "maxLMo",
  plot = TRUE, xlab = "tests (ordinal)", ylim = c(0, 40))

## maxLM vs. maxLR test
set.seed(1)
sctest(mr1, order.by = mex1$otests,
  vcov = "info", functional = "maxLMo",
  plot = TRUE, xlab = "tests (ordinal)", ylim = c(0, 40))
lines(sapply(1:(nlevels(mex1$otests) - 1), function(i)
  2 * (logLik(raschmodel(subset(mex1, as.numeric(otests) <= i)$solved)) +
  logLik(raschmodel(subset(mex1, as.numeric(otests) > i)$solved)) - logLik(mr1))),
  type = "b", lty = 2)
legend("topleft", c("LR", "LM"), lty = 2:1, bty = "n")


## DIF: tree-based detection ---------------------------------------------------

mex$tests <- ordered(mex$tests)
mex$nsolved <- ordered(mex$nsolved)
mex$attempt <- ordered(mex$attempt)

library("psychotree")
set.seed(1)
mrt <- raschtree(solved ~ group + tests + nsolved + gender + attempt + study + semester,
  data = mex, vcov = "info", minsize = 50, ordinal = "l2", nrep = 1e5)
plot(mrt)


## DIF: finite mixture model with group 1 only ---------------------------------

library("psychomix")
set.seed(1)
mrm <- raschmix(mex1$solved, k = 2, scores = "meanvar")
mrm

plot(mrm)

