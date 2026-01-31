stopifnot(require("strucchange"))

set.seed(2409)

data("VerbalAggression", package = "psychotools")

refmodel <- nplmodel(VerbalAggression$resp2[, 1:6])

dgp <- function(model, N = 1000, G = 1, impact = FALSE, cotype = "random") {

  mu <- if(impact) rep_len(c(-0.5, 0.5), N) else rep_len(0, N)
  theta <- rnorm(N, mu, 1)

  covariate <- switch(as.character(cotype),
    "random" = rnorm(N, 0, 1),
    "linear" = theta + rnorm(N, 0, 1),
    "quadratic" = theta ^ 2 + rnorm(N, 0, 1)
  )

  d <- data.frame(theta = theta, covariate = covariate)
  if(G > 1) {
    d$impact <- cut(covariate, labels = 1:G, include.lowest = TRUE,
      breaks = quantile(covariate, probs = 0:G / G))
  }

  d$resp <- rpl(theta,
    a = discrpar(model),
    b = itempar(model),
    g = guesspar(model),
    u = upperpar(model),
    return_setting = FALSE)

  return(d)
}

hitrate <- function(model, M = 1000, alpha = 0.05, parm = NULL, N = 1000,
  G = 1, impact = FALSE, cotype = "random") {

  pval <- replicate(M, {
    d <- dgp(model, N = N, G = G, impact = impact, cotype = cotype)
    m <- nplmodel(d$resp, type = "2PL", impact = d$impact,
      maxit = 5000, reltol = 1e-4, vcov = FALSE)
    sctest(m, order.by = d$covariate, functional = "DM", parm = parm)$p.value
  })
  mean(pval < alpha)
}

sim <- function(model, M = 1000, alpha = 0.05, parm = NULL, N = 1000,
  G = c(1, 2, 5, 25), impact = c(FALSE, TRUE),
  cotype = c("random", "linear", "quadratic")) {

  d <- expand.grid(G = G, impact = impact, cotype = cotype)
  d$hitrate <- NA
  for(i in seq_len(NROW(d))) {
    d$hitrate[i] <- hitrate(model, M = M, alpha = alpha, parm = parm, N = N,
    G = d$G[i], impact = d$impact[i], cotype = d$cotype[i])
  }
  return(d)
}

simulation1 <- sim(refmodel)
