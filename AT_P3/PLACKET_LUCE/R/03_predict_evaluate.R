## Predict full rankings for held-out seasons and compute ranking metrics.

ART <- file.path(ROOT, "artifacts")
df_cs <- readRDS(file.path(ART, "contestant_season.rds"))
rankings_by_season <- readRDS(file.path(ART, "rankings_by_season.rds"))
ridge <- readRDS(file.path(ART, "ridge_model.rds"))
cfg <- readr::read_csv(file.path(ART, "final_config.csv"), show_col_types = FALSE)
best <- readr::read_csv(file.path(ART, "best_hyperparams.csv"), show_col_types = FALSE)

train_seasons <- readr::read_csv(file.path(ART, "train_seasons.csv"), show_col_types = FALSE)$train_season
test_seasons <- readr::read_csv(file.path(ART, "test_seasons.csv"), show_col_types = FALSE)$test_season

rare_partner <- readr::read_csv(file.path(ART, "rare_partner_levels.csv"), show_col_types = FALSE)$level
rare_state <- readr::read_csv(file.path(ART, "rare_state_levels.csv"), show_col_types = FALSE)$level
rare_country <- readr::read_csv(file.path(ART, "rare_country_levels.csv"), show_col_types = FALSE)$level

apply_collapse <- function(x, rare, other = "Other") {
  x <- as.character(x)
  x[is.na(x) | x == ""] <- "Unknown"
  ifelse(x %in% rare, other, x)
}

build_cov_for_seasons <- function(seasons_use) {
  d <- df_cs |> dplyr::filter(season %in% seasons_use)
  d <- d |>
    dplyr::mutate(
      ballroom_partner_c = apply_collapse(ballroom_partner, rare_partner),
      celebrity_homestate_c = apply_collapse(celebrity_homestate, rare_state),
      celebrity_homecountry_region_c = apply_collapse(celebrity_homecountry_region, rare_country),
      celebrity_industry = ifelse(is.na(celebrity_industry) | celebrity_industry == "", "Unknown", celebrity_industry),
      age_z = (celebrity_age_during_season - cfg$age_center[[1]]) / cfg$age_scale[[1]]
    )
  cov <- d |>
    dplyr::select(item_id, age_z, celebrity_industry, ballroom_partner_c, celebrity_homestate_c, celebrity_homecountry_region_c) |>
    dplyr::distinct(item_id, .keep_all = TRUE)
  cov
}

predict_scores <- function(model, covariates_df) {
  mf <- model.frame(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c,
                    data = covariates_df, drop.unused.levels = FALSE)
  X <- model.matrix(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c,
                    data = mf)
  # align columns to training
  missing_cols <- setdiff(model$colnames, colnames(X))
  if (length(missing_cols) > 0) {
    X <- cbind(X, matrix(0, nrow = nrow(X), ncol = length(missing_cols), dimnames = list(NULL, missing_cols)))
  }
  X <- X[, model$colnames, drop = FALSE]
  drop(X %*% model$beta)
}

eval_one_season <- function(df_season, scores_vec) {
  truth <- df_season |> dplyr::arrange(placement_final, celebrity_name)
  s <- scores_vec[match(truth$item_id, names(scores_vec))]
  pred_rank <- rank(-s, ties.method = "average")
  truth_rank <- truth$placement_final

  # Plackett–Luce log-likelihood of the observed (true) ranking under predicted worths.
  # Using log-worth scores s: loglik = sum_{j} (s_j - logsumexp(s_j..s_n)).
  logsumexp <- function(x) {
    m <- max(x)
    m + log(sum(exp(x - m)))
  }
  ll <- 0
  for (j in seq_along(s)) {
    ll <- ll + s[j] - logsumexp(s[j:length(s)])
  }

  tibble::tibble(
    spearman = suppressWarnings(cor(truth_rank, pred_rank, method = "spearman")),
    kendall_tau = suppressWarnings(cor(truth_rank, pred_rank, method = "kendall")),
    top1 = as.integer(which.min(truth_rank) == which.min(pred_rank)),
    top3 = as.integer(any(order(pred_rank)[1:3] %in% order(truth_rank)[1:3])),
    ndcg = ndcg_score(truth_rank, pred_rank),
    loglik = ll,
    loglik_per_item = ll / length(s)
  )
}

# Predict + store per-season rankings
cov_test <- build_cov_for_seasons(test_seasons)
scores_test <- predict_scores(ridge, cov_test)
names(scores_test) <- cov_test$item_id

pred_rows <- list()
metric_rows <- list()

for (s in test_seasons) {
  d <- df_cs |> dplyr::filter(season == s) |> dplyr::filter(!is.na(placement_final))
  sc <- scores_test[match(d$item_id, names(scores_test))]
  d$score <- sc
  d <- d |>
    dplyr::mutate(
      predicted_rank = rank(-score, ties.method = "average"),
      true_rank = placement_final
    ) |>
    dplyr::arrange(predicted_rank, celebrity_name)

  pred_rows[[length(pred_rows) + 1]] <- d |>
    dplyr::select(season, celebrity_name, true_rank, predicted_rank, score, ballroom_partner, celebrity_homestate, celebrity_homecountry_region, celebrity_industry, celebrity_age_during_season)

  met <- eval_one_season(d, setNames(scores_test, names(scores_test)))
  met$season <- s
  metric_rows[[length(metric_rows) + 1]] <- met
}

pred_df <- dplyr::bind_rows(pred_rows)
metrics_df <- dplyr::bind_rows(metric_rows) |>
  dplyr::select(season, dplyr::everything())

readr::write_csv(pred_df, file.path(ART, "predictions_test_seasons.csv"))
readr::write_csv(metrics_df, file.path(ART, "metrics_by_season.csv"))

overall <- metrics_df |>
  dplyr::summarise(
    n_seasons = dplyr::n(),
    spearman_mean = mean(spearman, na.rm = TRUE),
    spearman_sd = sd(spearman, na.rm = TRUE),
    kendall_mean = mean(kendall_tau, na.rm = TRUE),
    kendall_sd = sd(kendall_tau, na.rm = TRUE),
    top1 = mean(top1, na.rm = TRUE),
    top3 = mean(top3, na.rm = TRUE),
    ndcg_mean = mean(ndcg, na.rm = TRUE),
    ndcg_sd = sd(ndcg, na.rm = TRUE)
  )
readr::write_csv(overall, file.path(ART, "metrics_overall.csv"))


