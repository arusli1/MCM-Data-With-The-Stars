## 5-fold cross-validation by season (no hyperparameter changes).
##
## Uses the same fixed configuration as the final model:
##   npseudo, min_partner, min_state, min_country, lambda
## and refits within each fold on the training seasons only.

ART <- file.path(ROOT, "artifacts")
df_cs <- readRDS(file.path(ART, "contestant_season.rds"))
rankings_by_season <- readRDS(file.path(ART, "rankings_by_season.rds"))

cfg <- readr::read_csv(file.path(ART, "final_config.csv"), show_col_types = FALSE)
npseudo <- cfg$npseudo[[1]]
min_partner <- cfg$min_partner[[1]]
min_state <- cfg$min_state[[1]]
min_country <- cfg$min_country[[1]]
lambda <- cfg$lambda[[1]]

seasons_all <- sort(unique(df_cs$season))

set.seed(123)
K <- 5
seasons_shuf <- sample(seasons_all)
folds <- split(seasons_shuf, rep(1:K, length.out = length(seasons_shuf)))

get_rare_levels <- function(x, min_n) {
  x <- as.character(x)
  x[is.na(x) | x == ""] <- "Unknown"
  tab <- table(x)
  names(tab)[tab < min_n]
}

apply_collapse <- function(x, rare, other = "Other") {
  x <- as.character(x)
  x[is.na(x) | x == ""] <- "Unknown"
  ifelse(x %in% rare, other, x)
}

build_covariates <- function(df, seasons_use, rare_partner, rare_state, rare_country, age_center, age_scale) {
  d <- df |> dplyr::filter(season %in% seasons_use)
  d <- d |>
    dplyr::mutate(
      celebrity_industry = ifelse(is.na(celebrity_industry) | celebrity_industry == "", "Unknown", celebrity_industry),
      ballroom_partner_c = apply_collapse(ballroom_partner, rare_partner),
      celebrity_homestate_c = apply_collapse(celebrity_homestate, rare_state),
      celebrity_homecountry_region_c = apply_collapse(celebrity_homecountry_region, rare_country),
      age_z = (celebrity_age_during_season - age_center) / age_scale
    )
  d |>
    dplyr::select(item_id, age_z, celebrity_industry, ballroom_partner_c, celebrity_homestate_c, celebrity_homecountry_region_c) |>
    dplyr::distinct(item_id, .keep_all = TRUE)
}

fit_pl_worth <- function(rankings_list, npseudo) {
  R <- do.call(rbind, rankings_list)
  PlackettLuce::PlackettLuce(R, npseudo = npseudo, verbose = FALSE)
}

ridge_fit <- function(X, y, pen) {
  XtX <- crossprod(X)
  A <- XtX + diag(pen, nrow = ncol(X))
  b <- crossprod(X, y)
  as.numeric(solve(A, b))
}

eval_season <- function(df_season, scores_named) {
  truth <- df_season |> dplyr::arrange(placement_final, celebrity_name)
  sc <- scores_named[match(truth$item_id, names(scores_named))]
  pred_rank <- rank(-sc, ties.method = "average")
  truth_rank <- truth$placement_final
  topk <- min(3, length(truth_rank))
  idx_top3 <- order(truth_rank)[1:topk]
  idx_top1 <- which.min(truth_rank)

  # Plackett–Luce log-likelihood of observed ranking under predicted worths.
  logsumexp <- function(x) {
    m <- max(x)
    m + log(sum(exp(x - m)))
  }
  ll <- 0
  for (j in seq_along(sc)) {
    ll <- ll + sc[j] - logsumexp(sc[j:length(sc)])
  }
  tibble::tibble(
    spearman = suppressWarnings(cor(truth_rank, pred_rank, method = "spearman")),
    kendall_tau = suppressWarnings(cor(truth_rank, pred_rank, method = "kendall")),
    kendall_tau_top3 = suppressWarnings(cor(truth_rank[idx_top3], pred_rank[idx_top3], method = "kendall")),
    mse_top3 = mean((pred_rank[idx_top3] - truth_rank[idx_top3])^2),
    se_top1 = (pred_rank[idx_top1] - 1)^2,
    top1 = as.integer(which.min(truth_rank) == which.min(pred_rank)),
    top3 = as.integer(any(order(pred_rank)[1:3] %in% order(truth_rank)[1:3])),
    ndcg = ndcg_score(truth_rank, pred_rank),
    loglik = ll,
    loglik_per_item = ll / length(sc)
  )
}

cv_metrics_rows <- list()

for (f in seq_along(folds)) {
  test_seasons <- folds[[f]]
  train_seasons <- setdiff(seasons_all, test_seasons)

  tr_df <- df_cs |> dplyr::filter(season %in% train_seasons)
  rare_partner <- get_rare_levels(tr_df$ballroom_partner, min_partner)
  rare_state <- get_rare_levels(tr_df$celebrity_homestate, min_state)
  rare_country <- get_rare_levels(tr_df$celebrity_homecountry_region, min_country)

  age_center <- mean(tr_df$celebrity_age_during_season, na.rm = TRUE)
  age_scale <- sd(tr_df$celebrity_age_during_season, na.rm = TRUE)
  if (!is.finite(age_scale) || age_scale == 0) age_scale <- 1

  # Stage A: PL worths on training seasons
  pl_fit <- fit_pl_worth(rankings_by_season[as.character(train_seasons)], npseudo)
  logworth <- coef(pl_fit)

  # Stage B: ridge regression on log-worths
  cov_tr <- build_covariates(df_cs, train_seasons, rare_partner, rare_state, rare_country, age_center, age_scale)
  y <- logworth[cov_tr$item_id]
  keep <- which(!is.na(y))
  cov_tr2 <- cov_tr[keep, , drop = FALSE]
  y2 <- as.numeric(y[keep])

  mf <- model.frame(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c,
                    data = cov_tr2, drop.unused.levels = TRUE)
  X <- model.matrix(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c,
                    data = mf)

  pen <- rep(lambda, ncol(X))
  pen[1] <- 0
  pen[grep("^age_z$", colnames(X))] <- 0
  pen[grep("^celebrity_industry", colnames(X))] <- 0
  beta <- ridge_fit(X, y2, pen)

  # Predict test seasons
  cov_te <- build_covariates(df_cs, test_seasons, rare_partner, rare_state, rare_country, age_center, age_scale)
  mf_te <- model.frame(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c,
                       data = cov_te, drop.unused.levels = FALSE)
  Xte <- model.matrix(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c,
                      data = mf_te)
  missing_cols <- setdiff(colnames(X), colnames(Xte))
  if (length(missing_cols) > 0) {
    Xte <- cbind(Xte, matrix(0, nrow = nrow(Xte), ncol = length(missing_cols), dimnames = list(NULL, missing_cols)))
  }
  Xte <- Xte[, colnames(X), drop = FALSE]
  pred_logw <- drop(Xte %*% beta)
  names(pred_logw) <- cov_te$item_id

  for (s in test_seasons) {
    d_season <- df_cs |> dplyr::filter(season == s) |> dplyr::filter(!is.na(placement_final))
    met <- eval_season(d_season, pred_logw)
    met$fold <- f
    met$season <- s
    cv_metrics_rows[[length(cv_metrics_rows) + 1]] <- met
  }
}

cv_metrics <- dplyr::bind_rows(cv_metrics_rows) |>
  dplyr::select(fold, season, dplyr::everything())

readr::write_csv(cv_metrics, file.path(ART, "cv5_season_metrics_by_season.csv"))

cv_overall <- cv_metrics |>
  dplyr::summarise(
    n_seasons = dplyr::n(),
    spearman_mean = mean(spearman, na.rm = TRUE),
    spearman_sd = sd(spearman, na.rm = TRUE),
    kendall_mean = mean(kendall_tau, na.rm = TRUE),
    kendall_sd = sd(kendall_tau, na.rm = TRUE),
    kendall_top3_mean = mean(kendall_tau_top3, na.rm = TRUE),
    kendall_top3_sd = sd(kendall_tau_top3, na.rm = TRUE),
    mse_top3_mean = mean(mse_top3, na.rm = TRUE),
    mse_top3_sd = sd(mse_top3, na.rm = TRUE),
    se_top1_mean = mean(se_top1, na.rm = TRUE),
    se_top1_sd = sd(se_top1, na.rm = TRUE),
    top1 = mean(top1, na.rm = TRUE),
    top3 = mean(top3, na.rm = TRUE),
    ndcg_mean = mean(ndcg, na.rm = TRUE),
    ndcg_sd = sd(ndcg, na.rm = TRUE)
    ,loglik_mean = mean(loglik, na.rm = TRUE),
    loglik_sd = sd(loglik, na.rm = TRUE),
    loglik_per_item_mean = mean(loglik_per_item, na.rm = TRUE),
    loglik_per_item_sd = sd(loglik_per_item, na.rm = TRUE)
  )

readr::write_csv(cv_overall, file.path(ART, "cv5_season_metrics_overall.csv"))


