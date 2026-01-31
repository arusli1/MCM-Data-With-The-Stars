## packages --------------------------------------------------------------------

stopifnot(require("psychotools"), require("likert"), require("lmtest"),
  require("car"), require("psychotree"))


## data ------------------------------------------------------------------------

## load data
data("VerbalAggression", package = "psychotools")

## new data set with covariates
d <- VerbalAggression[, c("gender", "anger")]

## and polytomous item responses
d$poly <- itemresp(VerbalAggression$resp[, 13:18], mscale = 0:2,
  labels = c("Want-Curse", "Do-Curse", "Want-Scold", "Do-Scold", "Want-Shout", "Do-Shout"))

## printing
head(d$poly)

## summarizing
summary(d$poly)
prop.table(summary(d$poly), 1)

## visualization
plot(d$poly)

## connection to likert package
library("likert")
lik <- likert(as.data.frame(as.list(d$poly)))
lik
plot(lik)

## dichotomized version of item responses
d$dich <- d$poly
mscale(d$dich) <- c(0, 1, 1)

## compare
par(mfrow = c(1, 2))
plot(d$poly)
plot(d$dich)
par(mfrow = c(1, 1))

## mention is.na()


## Rasch model -----------------------------------------------------------------

## model
ram <- raschmodel(d$dich)
ram
summary(ram)

## visualization
plot(ram, type = "curves")

par(mfrow = c(1, 2))
plot(ram, type = "profile")
plot(ram, type = "regions")
par(mfrow = c(1, 1))

plot(ram, type = "piplot")

plot(ram, type = "information")


## item parameters
raip <- itempar(ram)
raip
confint(raip)

## inference
confint(itempar(ram, ref = 1, alias = FALSE))
library("lmtest")
coeftest(itempar(ram, ref = 1, alias = FALSE))

## person parameters
rapp <- personpar(ram)
rapp
confint(rapp)


## Polytomous Rasch models -----------------------------------------------------

## rating scale and partial credit model
rsm <- rsmodel(d$poly)
pcm <- pcmodel(d$poly)


## person parameters
pp <- personpar(pcm)
vcpp <- vcov(pp)


## comparison of (mean) item parameters
iprm <- itempar(ram, alias = FALSE, ref = 1)
iprsm <- itempar(rsm, alias = FALSE, ref = 1)
ippcm <- itempar(pcm, alias = FALSE, ref = 1)
print(cbind(RM = iprm, RSM = iprsm, PCM = ippcm), digits = 5)

## comparison of absolute threshold parameters
atprsm <- threshpar(rsm, relative = FALSE, ref = 1)
atppcm <- threshpar(pcm, relative = FALSE, ref = 1)
print(cbind(coef(atprsm, type = "matrix"), 
            coef(atppcm, type = "matrix")), digits = 5)

## comparison of relative threshold parameters
rtprsm <- threshpar(rsm, ref = 1, relative = TRUE)
rtppcm <- threshpar(pcm, ref = 1, relative = TRUE)
print(cbind(coef(rtprsm, type = "matrix"), 
            coef(rtppcm, type = "matrix")), digits = 5)


## PCM visualizations
lbs <- labels(d$poly)
tlm <- c(-2, 6)
cols <- hcl.colors(4, "Set 2")
cols2 <- hcl.colors(6, "ag_Sunset", rev = TRUE)
cols3 <- hcl.colors(3, "Dark 2")

plot(pcm, type = "curves", ref = 1, items = 1:6, layout = matrix(1:6, ncol = 3, nrow = 2, byrow = FALSE), 
     names = lbs, ylim = c(0, 1), xlim = tlm, col = cols[1:3], lwd = 1.5,
     xlab = expression(paste("Latent trait ", theta)), ylab = "Probability")
plot(pcm, type = "regions", names = lbs, parg = list(ref = 1),
     ylab = expression(paste("Latent trait ", theta)), ylim = tlm, col = cols[1:3])
plot(pcm, type = "profile", what = "items", names = lbs, 
     ylab = expression(paste("Latent trait ", theta)), parg = list(ref = 1), ylim = tlm)


## PCM vs. RSM (vs. RAM) visualizations
## characteristic curves
plot(rsm, type = "curves", items = 1, ref = 1, xlim = tlm, col = c("#E495A5", "#ABB065", "#39BEB1"), lwd = 2, main = "", xlab = expression(paste("Latent trait ", theta)))
plot(pcm, type = "curves", items = 1, ref = 1, lty = 2, xlim = tlm, col = c("#E495A5", "#ABB065", "#39BEB1"), lwd = 2, add = TRUE)
legend(x = 0.85, y = 6.5, legend = c("RSM", "PCM"), bty = "n", lty = 1:2, lwd = 2)

## profile (items)
plot(ram, type = "profile", what = "items", parg = list(ref = 1), col = cbind(cols3[1], "lightgray"), ref = FALSE, names = lbs, ylim = tlm, ylab = "Item location parameters")
plot(rsm, type = "profile", what = "items", parg = list(ref = 1), col = cbind(cols3[2], "lightgray"), ylim = tlm, add = TRUE)
plot(pcm, type = "profile", what = "items", parg = list(ref = 1), col = cbind(cols3[3], "lightgray"), ylim = tlm, add = TRUE)
legend(x = 0.85, y = 6.5, legend = c("RM", "RSM", "PCM"), bty = "n", col = cols3, pch = 16)

## profile (thresholds)
plot(rsm, type = "profile", what = "threshold", parg = list(ref = 1), col = cbind(cols3[2], cols3[2]), names = lbs, ylim = tlm)
plot(pcm, type = "profile", what = "threshold", parg = list(ref = 1), col = cbind(cols3[3], cols3[3]), add = TRUE)
legend(x = 0.85, y = 6.5, legend = c("RSM", "PCM"), bty = "n", col = cols3[2:3], pch = 16)

## information curves
plot(ram, type = "information", what = "items", ref = 1, xlab = expression(paste("Latent trait ", theta)),
     names = lbs, lty = 1, xlim = c(-4, 8), ylim = c(0, 0.6), main = "", lwd = 2, col = cols2, ylab = "Item information")
plot(pcm, type = "information", what = "items", ref = 1, lty = 2, xlim = c(-4, 8), lwd = 2, col = cols2, add = TRUE)
legend(x = 6, y = 0.74, legend = c("RM", "PCM"), bty = "n", lwd = 2, lty = 1:2)

## category information
plot(pcm, type = "information", what = "categories", ref = 1, items = 1:6, xlab = expression(paste("Latent trait ", theta)), ylab = "Category information",
     lwd = 2, names = lbs, col = cols, xlim = c(-4, 8), ylim = c(0, 0.3), layout = matrix(1:6, ncol = 3, nrow = 2, byrow = FALSE))


## model comparisons
## information criteria
AIC(rsm, pcm)
BIC(rsm, pcm)
## likelihood ratio test
library("lmtest")
lrtest(rsm, pcm)
## Wald test
tp <- threshpar(pcm, type = "mode", 
                relative = TRUE, ref = 1, alias = FALSE)
C <- cbind(1, diag(-1, nrow = 5, ncol = 5))
library("car")
linearHypothesis(tp, hypothesis.matrix = C, rhs = rep.int(0, 5))


## DIF detection by gender -----------------------------------------------------

## models
rmall <- raschmodel(d$dich)
rmmale <- raschmodel(subset(d, gender == "male")$dich)
rmfemale <- raschmodel(subset(d, gender == "female")$dich)

## likelihood ratio test (global)
(LRT <- as.vector(- 2 * (logLik(rmall) - 
                        (logLik(rmmale) + logLik(rmfemale)))))
pchisq(LRT, df = 5, lower.tail = FALSE)

## Wald test (itemwise with given anchoring)
ipmale <- itempar(rmmale, ref = c(2, 3))
ipfemale <- itempar(rmfemale, ref = c(2, 3))
waldtests <- (coef(ipmale) - coef(ipfemale)) / 
    sqrt(diag(vcov(ipmale)) + diag(vcov(ipfemale)))
pvalues <- 2*pnorm(abs(waldtests), lower.tail = FALSE)
cbind("Test Statistic" = waldtests, "P-Value" = pvalues)

## Wald test (automatic anchor selection
set.seed(1)
anchortest(rmfemale, rmmale, class = "forward", 
           method = "MPT", adjust = "holm")

## tree
library("psychotree")
rt <- raschtree(dich ~ gender + anger, data = d)
plot(rt, tp_args = list(names = lbs, abbreviate = FALSE))
