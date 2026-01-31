## Factor ablation tests for Plackett–Luce ranking model with covariates.
##
## Primary metric: held-out negative log-likelihood (NLL) of the observed season rankings
## under predicted worths. Secondary: Spearman, Kendall, Top-1, Top-3.
##
## This module reuses the existing PLACKET_LUCE pipeline approach:
##   Stage A: fit Plackett–Luce worths on training seasons with fixed npseudo (shrinkage/connectivity)
##   Stage B: ridge regression mapping covariates -> log-worth, with differential penalties:
##            - no shrink: age, industry
##            - shrink: partner/state/country (penalized + optional rare-level collapsing)
##
## NOTE: PlackettLuce::pladmm (ranking regression) was numerically unstable in this repo,
## so ablations are done using this stable two-stage approach while keeping the core PL likelihood.

ART <- file.path(ROOT, "artifacts")
ABL <- file.path(ART, "ablation")
dir.create(ABL, recursive = TRUE, showWarnings = FALSE)

df_cs <- readRDS(file.path(ART, "contestant_season.rds"))

# --- folds ---
make_season_folds <- function(df, k = 5, seed = 1) {
  set.seed(seed)
  seasons <- sort(unique(df$season))
  shuf <- sample(seasons)
  folds <- split(shuf, rep(1:k, length.out = length(shuf)))
  lapply(seq_along(folds), function(i) {
    list(
      fold = i,
      test_seasons = folds[[i]],
      train_seasons = setdiff(seasons, folds[[i]])
    )
  })
}

# --- utilities ---
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

prep_rankings <- function(df_subset) {
  # Build season rankings objects from a contestant-season subset.
  seasons <- sort(unique(df_subset$season))
  ranks <- lapply(seasons, function(s) {
    d <- df_subset[df_subset$season == s, , drop = FALSE]
    d <- d[order(d$placement_final, d$celebrity_name), , drop = FALSE]
    long <- tibble::tibble(
      ranking = 1L,
      item = d$item_id,
      rank = as.integer(d$placement_final)
    )
    PlackettLuce::rankings(long, id = "ranking", item = "item", rank = "rank", aggregate = FALSE)
  })
  names(ranks) <- as.character(seasons)
  ranks
}

fit_pl_model <- function(rankings_list, npseudo) {
  R <- do.call(rbind, rankings_list)
  PlackettLuce::PlackettLuce(R, npseudo = npseudo, verbose = FALSE)
}

build_features <- function(df, seasons_use, rare_partner, rare_state, rare_country, age_center, age_scale) {
  d <- df[df$season %in% seasons_use, , drop = FALSE]
  d$celebrity_industry <- ifelse(is.na(d$celebrity_industry) | d$celebrity_industry == "", "Unknown", d$celebrity_industry)
  d$ballroom_partner_c <- apply_collapse(d$ballroom_partner, rare_partner)
  d$celebrity_homestate_c <- apply_collapse(d$celebrity_homestate, rare_state)
  d$celebrity_homecountry_region_c <- apply_collapse(d$celebrity_homecountry_region, rare_country)
  d$age_z <- (d$celebrity_age_during_season - age_center) / age_scale
  dplyr::distinct(
    dplyr::select(
      dplyr::as_tibble(d),
      item_id, season, celebrity_name, placement_final,
      age_z, celebrity_industry, ballroom_partner_c, celebrity_homestate_c, celebrity_homecountry_region_c
    ),
    item_id, .keep_all = TRUE
  )
}

ridge_fit <- function(X, y, pen) {
  A <- crossprod(X) + diag(pen, nrow = ncol(X))
  b <- crossprod(X, y)
  as.numeric(solve(A, b))
}

predict_logworth <- function(beta, X) drop(X %*% beta)

loglik_pl_ranking <- function(scores_in_true_order) {
  # scores_in_true_order: vector s_1..s_n in observed best->worst order
  logsumexp <- function(x) {
    m <- max(x)
    m + log(sum(exp(x - m)))
  }
  ll <- 0
  for (j in seq_along(scores_in_true_order)) {
    ll <- ll + scores_in_true_order[j] - logsumexp(scores_in_true_order[j:length(scores_in_true_order)])
  }
  ll
}

score_fold <- function(model_variant, df_train, df_test, cfg) {
  # cfg: list(npseudo, min_partner, min_state, min_country, lambda)
  npseudo <- cfg$npseudo
  min_partner <- cfg$min_partner
  min_state <- cfg$min_state
  min_country <- cfg$min_country
  lambda <- cfg$lambda

  # Train-derived preprocessing
  rare_partner <- get_rare_levels(df_train$ballroom_partner, min_partner)
  rare_state <- get_rare_levels(df_train$celebrity_homestate, min_state)
  rare_country <- get_rare_levels(df_train$celebrity_homecountry_region, min_country)

  age_center <- mean(df_train$celebrity_age_during_season, na.rm = TRUE)
  age_scale <- sd(df_train$celebrity_age_during_season, na.rm = TRUE)
  if (!is.finite(age_scale) || age_scale == 0) age_scale <- 1

  # Stage A: PL worths on training seasons (needed as targets for ridge)
  rk_train <- prep_rankings(df_train)
  pl_fit <- fit_pl_model(rk_train, npseudo)
  y_logworth <- coef(pl_fit) # named by item_id

  # Build training features and target vector
  feat_train <- build_features(df_train, unique(df_train$season), rare_partner, rare_state, rare_country, age_center, age_scale)
  y <- y_logworth[feat_train$item_id]
  keep <- which(!is.na(y))
  feat_train <- feat_train[keep, , drop = FALSE]
  y <- as.numeric(y[keep])

  # Build test features
  feat_test <- build_features(df_test, unique(df_test$season), rare_partner, rare_state, rare_country, age_center, age_scale)

  # Model matrix for selected factors
  fml <- model_variant$formula
  mf_tr <- model.frame(fml, data = feat_train, drop.unused.levels = TRUE)
  X_tr <- model.matrix(fml, data = mf_tr)

  mf_te <- model.frame(fml, data = feat_test, drop.unused.levels = FALSE)
  X_te <- model.matrix(fml, data = mf_te)

  # Align X_te columns to X_tr columns
  missing_cols <- setdiff(colnames(X_tr), colnames(X_te))
  if (length(missing_cols) > 0) {
    X_te <- cbind(X_te, matrix(0, nrow = nrow(X_te), ncol = length(missing_cols), dimnames = list(NULL, missing_cols)))
  }
  X_te <- X_te[, colnames(X_tr), drop = FALSE]

  # Differential ridge penalty: penalize shrinkage variables only (if present in formula)
  pen <- rep(0, ncol(X_tr))
  if (model_variant$penalize_shrinkage) {
    pen <- rep(lambda, ncol(X_tr))
    pen[1] <- 0
    pen[grep("^age_z$", colnames(X_tr))] <- 0
    pen[grep("^celebrity_industry", colnames(X_tr))] <- 0
    # If shrinkage vars are excluded from formula, their columns won't exist -> no penalty needed.
  }

  beta <- ridge_fit(X_tr, y, pen)
  scores_te <- predict_logworth(beta, X_te)
  names(scores_te) <- feat_test$item_id

  # Score each test season
  seasons_test <- sort(unique(df_test$season))
  per_season <- lapply(seasons_test, function(s) {
    d <- df_test[df_test$season == s, , drop = FALSE]
    d <- d[order(d$placement_final, d$celebrity_name), , drop = FALSE]
    sc <- scores_te[d$item_id]

    # NLL
    ll <- loglik_pl_ranking(sc)
    nll <- -ll

    # ranking metrics
    pred_rank <- rank(-sc, ties.method = "average")
    truth_rank <- d$placement_final

    tibble::tibble(
      season = s,
      n_items = nrow(d),
      nll = nll,
      nll_per_item = nll / nrow(d),
      spearman = suppressWarnings(cor(truth_rank, pred_rank, method = "spearman")),
      kendall_tau = suppressWarnings(cor(truth_rank, pred_rank, method = "kendall")),
      top1 = as.integer(which.min(truth_rank) == which.min(pred_rank)),
      top3_overlap = as.integer(length(intersect(order(truth_rank)[1:3], order(pred_rank)[1:3])) >= 1)
    )
  })
  dplyr::bind_rows(per_season)
}

# --- define model variants ---
fml_intercept <- ~ 1
fml_age_ind <- ~ 1 + age_z + celebrity_industry
fml_full <- ~ 1 + age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c

variants <- list(
  baseline = list(name = "baseline_age_industry", formula = fml_age_ind, penalize_shrinkage = TRUE),
  full = list(name = "full_all", formula = fml_full, penalize_shrinkage = TRUE),

  # plus-one from baseline (only meaningful for the shrinkage vars)
  add_partner = list(name = "baseline_plus_partner", formula = ~ 1 + age_z + celebrity_industry + ballroom_partner_c, penalize_shrinkage = TRUE),
  add_state = list(name = "baseline_plus_state", formula = ~ 1 + age_z + celebrity_industry + celebrity_homestate_c, penalize_shrinkage = TRUE),
  add_country = list(name = "baseline_plus_country", formula = ~ 1 + age_z + celebrity_industry + celebrity_homecountry_region_c, penalize_shrinkage = TRUE),

  # drop-one from full (covers all factors, including age/industry)
  drop_age = list(name = "full_minus_age", formula = ~ 1 + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c, penalize_shrinkage = TRUE),
  drop_industry = list(name = "full_minus_industry", formula = ~ 1 + age_z + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c, penalize_shrinkage = TRUE),
  drop_partner = list(name = "full_minus_partner", formula = ~ 1 + age_z + celebrity_industry + celebrity_homestate_c + celebrity_homecountry_region_c, penalize_shrinkage = TRUE),
  drop_state = list(name = "full_minus_state", formula = ~ 1 + age_z + celebrity_industry + ballroom_partner_c + celebrity_homecountry_region_c, penalize_shrinkage = TRUE),
  drop_country = list(name = "full_minus_country", formula = ~ 1 + age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c, penalize_shrinkage = TRUE),

  # optional: add age/industry from intercept-only (for completeness)
  add_age_only = list(name = "intercept_plus_age", formula = ~ 1 + age_z, penalize_shrinkage = TRUE),
  add_industry_only = list(name = "intercept_plus_industry", formula = ~ 1 + celebrity_industry, penalize_shrinkage = TRUE)
)

# --- run ablation ---
cfg <- readr::read_csv(file.path(ART, "final_config.csv"), show_col_types = FALSE)
cfg_list <- list(
  npseudo = cfg$npseudo[[1]],
  min_partner = cfg$min_partner[[1]],
  min_state = cfg$min_state[[1]],
  min_country = cfg$min_country[[1]],
  lambda = cfg$lambda[[1]]
)

folds <- make_season_folds(df_cs, k = 5, seed = 1)

by_season_rows <- list()
by_fold_rows <- list()

for (v in variants) {
  for (fold in folds) {
    tr <- df_cs[df_cs$season %in% fold$train_seasons, , drop = FALSE]
    te <- df_cs[df_cs$season %in% fold$test_seasons, , drop = FALSE]

    mets <- score_fold(v, tr, te, cfg_list) |>
      dplyr::mutate(
        fold = fold$fold,
        model = v$name
      )
    by_season_rows[[length(by_season_rows) + 1]] <- mets

    by_fold_rows[[length(by_fold_rows) + 1]] <- mets |>
      dplyr::summarise(
        fold = dplyr::first(fold),
        model = dplyr::first(model),
        n_seasons = dplyr::n(),
        nll_mean = mean(nll, na.rm = TRUE),
        nll_sd = sd(nll, na.rm = TRUE),
        spearman_mean = mean(spearman, na.rm = TRUE),
        kendall_mean = mean(kendall_tau, na.rm = TRUE),
        top1 = mean(top1, na.rm = TRUE),
        top3_overlap = mean(top3_overlap, na.rm = TRUE)
      )
  }
}

metrics_by_season_model <- dplyr::bind_rows(by_season_rows) |>
  dplyr::select(model, fold, season, dplyr::everything())

metrics_by_fold_model <- dplyr::bind_rows(by_fold_rows)

readr::write_csv(metrics_by_season_model, file.path(ABL, "metrics_by_season_model.csv"))
readr::write_csv(metrics_by_fold_model, file.path(ABL, "metrics_by_fold_model.csv"))

# --- factor impact summary ---
# Baseline comparison: baseline_age_industry
baseline_name <- "baseline_age_industry"
full_name <- "full_all"

fold_mean <- metrics_by_fold_model |>
  dplyr::select(fold, model, nll_mean, spearman_mean, kendall_mean, top1, top3_overlap)

baseline <- fold_mean |> dplyr::filter(model == baseline_name) |> dplyr::select(fold, nll_mean, spearman_mean, kendall_mean, top1, top3_overlap)
full <- fold_mean |> dplyr::filter(model == full_name) |> dplyr::select(fold, nll_mean, spearman_mean, kendall_mean, top1, top3_overlap)

delta_vs <- function(target_model) {
  tgt <- fold_mean |> dplyr::filter(model == target_model) |> dplyr::select(fold, nll_mean, spearman_mean, kendall_mean, top1, top3_overlap)
  tgt |> dplyr::left_join(baseline, by = "fold", suffix = c("", "_base")) |>
    dplyr::mutate(
      delta_nll_add = nll_mean - nll_mean_base,
      delta_spearman_add = spearman_mean - spearman_mean_base,
      delta_kendall_add = kendall_mean - kendall_mean_base,
      delta_top1_add = top1 - top1_base,
      delta_top3_add = top3_overlap - top3_overlap_base
    ) |>
    dplyr::select(fold, delta_nll_add, delta_spearman_add, delta_kendall_add, delta_top1_add, delta_top3_add)
}

delta_drop_vs_full <- function(drop_model) {
  dm <- fold_mean |> dplyr::filter(model == drop_model) |> dplyr::select(fold, nll_mean, spearman_mean, kendall_mean, top1, top3_overlap)
  dm |> dplyr::left_join(full, by = "fold", suffix = c("", "_full")) |>
    dplyr::mutate(
      delta_nll_drop = nll_mean - nll_mean_full,
      delta_spearman_drop = spearman_mean - spearman_mean_full,
      delta_kendall_drop = kendall_mean - kendall_mean_full,
      delta_top1_drop = top1 - top1_full,
      delta_top3_drop = top3_overlap - top3_overlap_full
    ) |>
    dplyr::select(fold, delta_nll_drop, delta_spearman_drop, delta_kendall_drop, delta_top1_drop, delta_top3_drop)
}

impact_rows <- list()
factor_specs <- list(
  ballroom_partner = list(add = "baseline_plus_partner", drop = "full_minus_partner"),
  celebrity_homestate = list(add = "baseline_plus_state", drop = "full_minus_state"),
  celebrity_homecountry_region = list(add = "baseline_plus_country", drop = "full_minus_country"),
  celebrity_age_during_season = list(add = "intercept_plus_age", drop = "full_minus_age"),
  celebrity_industry = list(add = "intercept_plus_industry", drop = "full_minus_industry")
)

for (fac in names(factor_specs)) {
  add_model <- factor_specs[[fac]]$add
  drop_model <- factor_specs[[fac]]$drop
  da <- delta_vs(add_model)
  dd <- delta_drop_vs_full(drop_model)
  merged <- da |> dplyr::left_join(dd, by = "fold")
  impact_rows[[length(impact_rows) + 1]] <- merged |>
    dplyr::summarise(
      factor = fac,
      delta_nll_add = mean(delta_nll_add, na.rm = TRUE),
      delta_nll_add_sd = sd(delta_nll_add, na.rm = TRUE),
      frac_folds_add_improves = mean(delta_nll_add < 0, na.rm = TRUE),
      delta_nll_drop = mean(delta_nll_drop, na.rm = TRUE),
      delta_nll_drop_sd = sd(delta_nll_drop, na.rm = TRUE),
      frac_folds_drop_hurts = mean(delta_nll_drop > 0, na.rm = TRUE),
      delta_spearman_add = mean(delta_spearman_add, na.rm = TRUE),
      delta_kendall_add = mean(delta_kendall_add, na.rm = TRUE),
      delta_top1_add = mean(delta_top1_add, na.rm = TRUE),
      delta_top3_add = mean(delta_top3_add, na.rm = TRUE)
    )
}

factor_impact_summary <- dplyr::bind_rows(impact_rows) |>
  dplyr::arrange(delta_nll_add)

readr::write_csv(factor_impact_summary, file.path(ABL, "factor_impact_summary.csv"))


