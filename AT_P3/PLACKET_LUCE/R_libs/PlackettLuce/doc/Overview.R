## ----setup, include = FALSE---------------------------------------------------
library(knitr)
knitr::knit_hooks$set(crop = function(before, options, envir) {
    # disable cropping even if magick package installed
    if (!before) return()
})
eval_all <- FALSE # evaluate all timings and pltree?
extra <- requireNamespace("kableExtra")
# if render using rmarkdown, use output format to decide table format
table.format <- opts_knit$get("rmarkdown.pandoc.to")
if (!identical(table.format, "latex")) table.format <- "html"
opts_knit$set(knitr.table.format = table.format)
opts_chunk$set(message = FALSE)

## ----echo = FALSE-------------------------------------------------------------
as.matrix(data.frame(choice = c(1, 1, 1, 2, 2),
                     `item 1` = c(1, 0, 0, 1, 0),
                     `item 2` = c(0, 1, 0, 0, 1),
                     `item 3` = c(0, 0, 1, 0, 0),
                     count = c(0, 0, 1, 1, 0), check.names = FALSE))


## ----soc, eval = eval_all, echo = FALSE---------------------------------------
# library(PlackettLuce)
# # read in example data sets
# # - originally from https://PrefLib.org, also distributed in PlackettLuce package
# # netflix: 00004-00000138.soc
# netflix <- read.soc(system.file("extdata", "netflix.soc",
#                                 package = "PlackettLuce"))
# # tshirt: 00012-00000001.soc
# tshirt <- read.soc(system.file("extdata", "shirt.soc",
#                                package = "PlackettLuce"))
# # sushi: 00014-00000001.soc
# sushi <- read.soc(system.file("extdata", "sushi.soc",
#                               package = "PlackettLuce"))

## ----wrappers, eval = eval_all, echo = FALSE----------------------------------
# pl <- function(dat, ...){
#     # convert ordered items to ranking
#     R <- as.rankings(dat[,-1], "ordering")
#     # fit without adding pseudo-rankings, weight rankings by count
#     PlackettLuce(R, npseudo = 0, weights = dat$Freq)
# }
# hyper2 <- function(dat, ...){
#     requireNamespace("hyper2")
#     # create likelihood object based on ordered items and counts
#     pnames <- paste0("p", seq_len(ncol(dat) - 1))
#     H <- hyper2::hyper2(pnames = pnames)
#     for (i in seq_len(nrow(dat))){
#         x <-  pnames[dat[i, -1][dat[i, -1] > 0]]
#         H <- H + hyper2::race(x)*dat[i, 1]
#     }
#     # find parameters to maximise likelihood
#     p <- hyper2::maxp(H)
#     structure(p, loglik = hyper2::loglik(p = p[-length(p)], H = H))
# }
# plmix <- function(dat, ...){
#     requireNamespace("PLMIX")
#     # disaggregate data (no functionality for weights or counts)
#     r <- rep(seq_len(nrow(dat)), dat$Freq)
#     # maximum a posteriori estimate, with non-informative prior
#     # K items in each ranking, single component distribution
#     # default starting values do not always work so specify as uniform
#     K <- ncol(dat) - 1
#     PLMIX::mapPLMIX(as.matrix(dat[r, -1]), K = K, G = 1,
#                     init = list(p = rep.int(1/K, K)), plot_objective = FALSE)
# }
# pmr <- function(dat, ...){
#     requireNamespace("pmr")
#     # convert ordered items to ranking
#     R <- as.rankings(dat[,-1], "ordering")
#     # create data frame with counts as required by pl
#     X <- as.data.frame(unclass(R))
#     X$Freq <- dat$Freq
#     capture.output(res <- pmr::pl(X))
#     res
# }
# statrank <- function(dat, iter){
#     requireNamespace("StatRank")
#     # disaggregate data (no functionality for weights or counts)
#     r <- rep(seq_len(nrow(dat)), dat$Freq)
#     capture.output(res <- StatRank::Estimation.PL.MLE(as.matrix(dat[r, -1]),
#                                                       iter = iter))
#     res
# }

## ----timings, eval = eval_all, echo = FALSE, message = FALSE, warning = FALSE----
# timings <- function(dat, iter = NULL,
#                     fun = c("pl", "hyper2", "plmix", "pmr", "statrank")){
#     res <- list()
#     for (nm in c("pl", "hyper2", "plmix", "pmr", "statrank")){
#         if (nm %in% fun){
#             res[[nm]] <- suppressWarnings(
#                 system.time(do.call(nm, list(dat, iter)))[["elapsed"]])
#         } else res[[nm]] <- NA
#     }
#     res
# }
# netflix_timings <- timings(netflix, 6)
# tshirt_timings <- timings(tshirt, 341,
#                           fun = c("pl", "hyper2", "plmix", "statrank"))
# sushi_timings <- timings(sushi, 5,
#                          fun = c("pl", "hyper2", "plmix", "statrank"))

## ----save-timings, echo = FALSE-----------------------------------------------
if (eval_all){
    saveRDS(netflix_timings, "netflix_timings.rds", version = 2)
    saveRDS(tshirt_timings, "tshirt_timings.rds", version = 2)
    saveRDS(sushi_timings, "sushi_timings.rds", version = 2)
} else {
    netflix_timings <- readRDS("netflix_timings.rds")
    tshirt_timings <- readRDS("tshirt_timings.rds")
    sushi_timings <- readRDS("sushi_timings.rds")
}

## ----data-features, echo = FALSE----------------------------------------------
features <- cbind(c(1256, 30, 5000),
                  c(24, 30, 4926),
                  c(4, 11, 10))
dimnames(features) <- list(c("Netflix", "T-shirt", "Sushi"),
                           c("Rankings", "Unique rankings", "Items"))
kab <- kable(features,
             caption = "(ref:features)",
             booktabs = TRUE) 
if (extra){
    kableExtra::kable_styling(kab)
} else kab

## ----timings-kable, echo = FALSE, results = "asis"----------------------------
res <- data.frame(netflix = round(unlist(netflix_timings), 3),
                  tshirt = format(unlist(tshirt_timings), 3),
                  sushi = round(unlist(sushi_timings), 3),
                  stringsAsFactors = FALSE)
res["pmr", "tshirt"] <- "a"
res["pmr", "sushi"] <- "a"
res <- t(as.matrix(res))
dimnames(res) <- list(c("Netflix", "T-shirt", "Sushi"),
                      c("PlackettLuce",
                        "hyper2", "PLMIX", "pmr", "StatRank"))

kab <- kable(res, booktabs = TRUE, align = rep("r", 6), caption = "(ref:timings)",
             escape = FALSE)
if (extra){
    kab <- kableExtra::kable_styling(kab)
    kab <- kableExtra::add_header_above(kab, c(" " = 1, "Time elapsed (s)" = 5))
    kableExtra::add_footnote(kab, "Function fails to complete.", 
                             notation = "alphabet")
} else {
    cat("**Time elapsed (s)**\n", kab, "\n^a Function fails to complete.", 
        sep = "\n")
}

## ----timings-sub, eval = eval_all, echo = FALSE, message = FALSE, warning = FALSE----
# data(Data.Nascar, package = "StatRank")
# # add column of frequencies so the format the same as before
# nascar <- cbind(n = 1, Data.Nascar)
# nascar_timings <- timings(nascar, fun = c("pl", "hyper2"))

## ----save-timings-sub, echo = FALSE-------------------------------------------
if (eval_all){
    saveRDS(nascar_timings, "nascar_timings.rds", version = 2)
} else {
    nascar_timings <- readRDS("nascar_timings.rds")
}

## ----nascar, echo = FALSE-----------------------------------------------------
res <- round(unlist(nascar_timings)[c("pl", "hyper2")], 3)
res <- data.frame(Rankings = 36,
                  Items = 83,
                  `Items per ranking` = "42-43",
                  PlackettLuce = res[1],
                  hyper2 = res[2],
                  check.names = FALSE, row.names = NULL)
kab <- kable(res, booktabs = TRUE, align = rep("r", 5),
      caption = "(ref:nascar)") 
if (requireNamespace("kableExtra")){
    kab <- kableExtra::kable_styling(kab)
    kableExtra::add_header_above(kab, c("Features of NASCAR data" = 3, 
                                        "Time elapsed (s)" = 2))
} else kab

## ----package-summary, echo = FALSE--------------------------------------------
tab <- data.frame(Feature = c("Inference", "Disconnected networks",
                              "Ties", "Teams", "Heterogenous case"),
                  PlackettLuce = c("Frequentist", "Yes", "Yes", "No", "Trees"),
                  hyper2 = c("No", "No", "No", "Yes", "No"),
                  pmr = c("No", "No", "No", "No", "No"),
                  StatRank = c("No", "No", "No", "No", "No"),
                  PLMIX = c("Bayesian", "Yes", "No", "No", "Mixtures"))
kab <- kable(tab, booktabs = TRUE,
      caption = "Features of packages for fitting the Plackett-Luce model.")
if (requireNamespace("kableExtra")){
    kableExtra::kable_styling(kab)
} else kab

## -----------------------------------------------------------------------------
library(PlackettLuce)
head(pudding)

## -----------------------------------------------------------------------------
i_wins <- data.frame(Winner = pudding$i, Loser = pudding$j)
j_wins <- data.frame(Winner = pudding$j, Loser = pudding$i)
if (getRversion() < "3.6.0"){
  n <- nrow(pudding)
  ties <- data.frame(Winner = array(split(pudding[c("i", "j")], 1:n), n),
                     Loser = rep(NA, 15))
} else {
  ties <- data.frame(Winner = asplit(pudding[c("i", "j")], 1),
                     Loser = rep(NA, 15))
}
head(ties, 2)

## -----------------------------------------------------------------------------
R <- as.rankings(rbind(i_wins, j_wins, ties),
                 input = "orderings")
head(R, 2)
tail(R, 2)

## -----------------------------------------------------------------------------
head(unclass(R), 2)

## -----------------------------------------------------------------------------
w <- unlist(pudding[c("w_ij", "w_ji", "t_ij")])

## -----------------------------------------------------------------------------
mod <- PlackettLuce(R, weights = w, npseudo = 0, maxit = 7)
coef(mod, log = FALSE)

## -----------------------------------------------------------------------------
summary(mod)

## -----------------------------------------------------------------------------
summary(mod, ref = NULL)

## -----------------------------------------------------------------------------
qv <- qvcalc(mod)
summary(qv)

## ----pudding-qv, fig.cap = "Worth of brands of chocolate pudding. Intervals based on quasi-standard errors."----
plot(qv, xlab = "Brand of pudding", ylab = "Worth (log)", main = NULL)

## -----------------------------------------------------------------------------
R <- matrix(c(1, 2, 0, 0,
              2, 0, 1, 0,
              1, 0, 0, 2,
              2, 1, 0, 0,
              0, 1, 2, 0), byrow = TRUE, ncol = 4,
            dimnames = list(NULL, LETTERS[1:4]))
R <- as.rankings(R)

## -----------------------------------------------------------------------------
A <- adjacency(R)
A

## ----always-loses, fig.cap = "Network representation of toy rankings.", fig.small = TRUE, fig.width = 3.5, fig.height = 3.5----
library(igraph)
net <- graph_from_adjacency_matrix(A)
plot(net, edge.arrow.size = 0.5, vertex.size = 30)

## -----------------------------------------------------------------------------
connectivity(A)

## -----------------------------------------------------------------------------
R2 <- R[, -4]
R2
mod <- PlackettLuce(R2, npseudo = 0)
summary(mod)

## -----------------------------------------------------------------------------
mod2 <- PlackettLuce(R)
coef(mod2)

## -----------------------------------------------------------------------------
data(nascar)
nascar[1:2, ]

## -----------------------------------------------------------------------------
R <- as.rankings(nascar, input = "orderings", items = attr(nascar, "drivers"))
R[1:2]

## -----------------------------------------------------------------------------
keep <- seq_len(83)
R2 <- R[, keep]
mod <- PlackettLuce(R2, npseudo = 0)

## -----------------------------------------------------------------------------
avRank <- apply(R, 2, function(x) mean(x[x > 0]))
coefs <- round(coef(mod)[order(avRank[keep])], 2)
head(coefs, 3)
tail(coefs, 3)

## -----------------------------------------------------------------------------
mod2 <- PlackettLuce(R)

## -----------------------------------------------------------------------------
coefs2 <- round(coef(mod2), 2)
coefs2[names(coefs)[1:3]]
coefs2[names(coefs)[81:83]]

## -----------------------------------------------------------------------------
coefs2[84:87]

## -----------------------------------------------------------------------------
coef(summary(mod2))[84:87,]

## ----nascar-qv, fig.cap = "Ability of drivers based on NASCAR 2002 season. Intervals based on quasi-standard errors.", fig.wide = TRUE----
qv <- qvcalc(mod2)
qv$qvframe <- qv$qvframe[order(coef(mod2)),]
plot(qv, xlab = NULL, ylab = "Ability (log)", main = NULL,
     xaxt="n", xlim = c(3, 85))
axis(1, at = seq_len(87), labels = rownames(qv$qvframe), las = 2, cex.axis = 0.6)

## ----beans-preparation, results = "hide"--------------------------------------
example("beans", package = "PlackettLuce")

## -----------------------------------------------------------------------------
n <- nrow(beans)
G <- group(R, index = rep(seq_len(n), 4))
format(head(G, 2), width = 50)

## ----pltree, eval = eval_all--------------------------------------------------
# beans$year <- factor(beans$year)
# tree <- pltree(G ~ ., data = beans[c("season", "year", "maxTN")],
#                minsize = 0.05*n, maxdepth = 3)
# tree

## ----save-tree, echo = FALSE--------------------------------------------------
if (eval_all){
    png("pltree.png", width = 11, height = 5, units = "in", res = 72)
    plot(tree, names = FALSE, abbreviate = 2, ylines = 2)
    dev.off()
}

## ----plot-pltree-code, eval = FALSE-------------------------------------------
# plot(tree, names = FALSE, abbreviate = 2, ylines = 2)

## ----plot-pltree, fig.wide = TRUE, fig.cap = "Worth parameters for the ten trial varieties and the local variety for each node in the Plackett-Luce tree. Varieties are 1: ALS 0532-6, 2: BRT 103-182, 3: INTA Centro Sur, 4: INTA Ferroso, 5: INTA Matagalpa, 6: INTA Precoz, 7: INTA Rojo, 8: INTA Sequia, 9: Local, 10: PM2 Don Rey, 11: SJC 730-79.", fig.width = 11, fig.height = 5, out.width = "\\textwidth", echo = FALSE----
include_graphics("pltree.png")

## ----show-soc-----------------------------------------------------------------

## ----soc, eval = FALSE--------------------------------------------------------
# library(PlackettLuce)
# # read in example data sets
# # - originally from https://PrefLib.org, also distributed in PlackettLuce package
# # netflix: 00004-00000138.soc
# netflix <- read.soc(system.file("extdata", "netflix.soc",
#                                 package = "PlackettLuce"))
# # tshirt: 00012-00000001.soc
# tshirt <- read.soc(system.file("extdata", "shirt.soc",
#                                package = "PlackettLuce"))
# # sushi: 00014-00000001.soc
# sushi <- read.soc(system.file("extdata", "sushi.soc",
#                               package = "PlackettLuce"))

## ----chunk2-------------------------------------------------------------------

## ----wrappers, eval = FALSE---------------------------------------------------
# pl <- function(dat, ...){
#     # convert ordered items to ranking
#     R <- as.rankings(dat[,-1], "ordering")
#     # fit without adding pseudo-rankings, weight rankings by count
#     PlackettLuce(R, npseudo = 0, weights = dat$Freq)
# }
# hyper2 <- function(dat, ...){
#     requireNamespace("hyper2")
#     # create likelihood object based on ordered items and counts
#     pnames <- paste0("p", seq_len(ncol(dat) - 1))
#     H <- hyper2::hyper2(pnames = pnames)
#     for (i in seq_len(nrow(dat))){
#         x <-  pnames[dat[i, -1][dat[i, -1] > 0]]
#         H <- H + hyper2::race(x)*dat[i, 1]
#     }
#     # find parameters to maximise likelihood
#     p <- hyper2::maxp(H)
#     structure(p, loglik = hyper2::loglik(p = p[-length(p)], H = H))
# }
# plmix <- function(dat, ...){
#     requireNamespace("PLMIX")
#     # disaggregate data (no functionality for weights or counts)
#     r <- rep(seq_len(nrow(dat)), dat$Freq)
#     # maximum a posteriori estimate, with non-informative prior
#     # K items in each ranking, single component distribution
#     # default starting values do not always work so specify as uniform
#     K <- ncol(dat) - 1
#     PLMIX::mapPLMIX(as.matrix(dat[r, -1]), K = K, G = 1,
#                     init = list(p = rep.int(1/K, K)), plot_objective = FALSE)
# }
# pmr <- function(dat, ...){
#     requireNamespace("pmr")
#     # convert ordered items to ranking
#     R <- as.rankings(dat[,-1], "ordering")
#     # create data frame with counts as required by pl
#     X <- as.data.frame(unclass(R))
#     X$Freq <- dat$Freq
#     capture.output(res <- pmr::pl(X))
#     res
# }
# statrank <- function(dat, iter){
#     requireNamespace("StatRank")
#     # disaggregate data (no functionality for weights or counts)
#     r <- rep(seq_len(nrow(dat)), dat$Freq)
#     capture.output(res <- StatRank::Estimation.PL.MLE(as.matrix(dat[r, -1]),
#                                                       iter = iter))
#     res
# }

## ----chunk2-------------------------------------------------------------------

## ----timings, eval = FALSE----------------------------------------------------
# timings <- function(dat, iter = NULL,
#                     fun = c("pl", "hyper2", "plmix", "pmr", "statrank")){
#     res <- list()
#     for (nm in c("pl", "hyper2", "plmix", "pmr", "statrank")){
#         if (nm %in% fun){
#             res[[nm]] <- suppressWarnings(
#                 system.time(do.call(nm, list(dat, iter)))[["elapsed"]])
#         } else res[[nm]] <- NA
#     }
#     res
# }
# netflix_timings <- timings(netflix, 6)
# tshirt_timings <- timings(tshirt, 341,
#                           fun = c("pl", "hyper2", "plmix", "statrank"))
# sushi_timings <- timings(sushi, 5,
#                          fun = c("pl", "hyper2", "plmix", "statrank"))

## ----beans-1------------------------------------------------------------------
data(beans)
head(beans[c("best", "worst")], 2)

## ----beans-2------------------------------------------------------------------
beans$middle <- complete(beans[c("best", "worst")],
                         items = c("A", "B", "C"))
head(beans[c("best", "middle", "worst")], 2)

## ----beans-3------------------------------------------------------------------
head(beans[c("variety_a", "variety_b", "variety_c")], 2)

## ----beans-4------------------------------------------------------------------
order3 <- decode(beans[c("best", "middle", "worst")],
                 items = beans[c("variety_a", "variety_b", "variety_c")],
                 code = c("A", "B", "C"))

## ----beans-6------------------------------------------------------------------
head(beans[c("var_a", "var_b", "var_c")], 2)

## ----beans-7------------------------------------------------------------------
trial_variety <- unlist(beans[c("variety_a", "variety_b", "variety_c")])
outcome <- unlist(beans[c("var_a", "var_b", "var_c")])

## -----------------------------------------------------------------------------
order2 <- data.frame(Winner = ifelse(outcome == "Worse",
                                     "Local", trial_variety),
                     Loser = ifelse(outcome == "Worse",
                                    trial_variety, "Local"),
                     stringsAsFactors = FALSE, row.names = NULL)
head(order2, 2)

## ----beans-8------------------------------------------------------------------
R <- rbind(as.rankings(order3, input = "ordering"),
           as.rankings(order2, input = "ordering"))
head(R)
tail(R)

