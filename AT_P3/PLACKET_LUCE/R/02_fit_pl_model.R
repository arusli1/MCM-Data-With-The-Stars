## Tune shrinkage hyperparameters (npseudo + rare-level thresholds) by season CV,
## then fit a Plackett–Luce worth model + a regression layer for covariate effects.
##
## Why 2-stage?
## - PlackettLuce::pladmm() (the prototype “ranking regression”) proved numerically unstable
##   for this dataset’s high-dimensional factor design.
## - We therefore fit:
##     Stage A: Plackett–Luce worths on training rankings with `npseudo` (pseudo-rankings shrink log-worths)
##     Stage B: a *ridge-regression* mapping covariates → log-worth, with differential penalties:
##         - no shrink: age, industry
##         - shrink: partner/state/country (penalized toward 0), plus rare-level collapsing
##
## This still produces a covariate-driven worth score for each contestant-season in held-out seasons,
## and keeps the core ranking likelihood Plackett–Luce (no Bradley–Terry conversion).

ART <- file.path(ROOT, "artifacts")
df_cs <- readRDS(file.path(ART, "contestant_season.rds"))
rankings_by_season <- readRDS(file.path(ART, "rankings_by_season.rds"))

# Train/test split by season: last ~20% seasons as test
seasons <- sort(unique(df_cs$season))
n_test <- max(1, ceiling(0.2 * length(seasons)))
test_seasons <- tail(seasons, n_test)
train_seasons <- setdiff(seasons, test_seasons)

readr::write_csv(tibble::tibble(train_season = train_seasons), file.path(ART, "train_seasons.csv"))
readr::write_csv(tibble::tibble(test_season = test_seasons), file.path(ART, "test_seasons.csv"))

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

# Build covariates for a set of seasons with *train-derived* rare-level collapsing and a train-derived age scaler.
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

# Differential-penalty ridge regression (closed form)
ridge_fit <- function(X, y, pen) {
  # beta = (X'X + diag(pen))^-1 X'y
  XtX <- crossprod(X)
  A <- XtX + diag(pen, nrow = ncol(XtX))
  b <- crossprod(X, y)
  as.numeric(solve(A, b))
}

predict_logworth <- function(X, beta) drop(X %*% beta)

eval_season <- function(df_season, scores_named) {
  truth <- df_season |> dplyr::arrange(placement_final, celebrity_name)
  sc <- scores_named[match(truth$item_id, names(scores_named))]
  pred_rank <- rank(-sc, ties.method = "average")
  truth_rank <- truth$placement_final
  tibble::tibble(
    spearman = suppressWarnings(cor(truth_rank, pred_rank, method = "spearman")),
    kendall_tau = suppressWarnings(cor(truth_rank, pred_rank, method = "kendall")),
    top1 = as.integer(which.min(truth_rank) == which.min(pred_rank)),
    top3 = as.integer(any(order(pred_rank)[1:3] %in% order(truth_rank)[1:3])),
    ndcg = ndcg_score(truth_rank, pred_rank)
  )
}

# Season-grouped CV on training seasons to choose npseudo + collapsing thresholds + ridge strength.
set.seed(123)
K <- 5
train_shuf <- sample(train_seasons)
folds <- split(train_shuf, rep(1:K, length.out = length(train_shuf)))

# IMPORTANT: with unique item_id per contestant-season, the comparison network across seasons
# is disconnected unless we add pseudo-rankings. Therefore we do not allow npseudo = 0.
grid_npseudo <- c(0.5, 1, 2)
grid_min_n <- c(1, 2, 3, 5)
grid_lambda <- c(0.1, 1, 10)

cv_rows <- list()
for (npseudo in grid_npseudo) {
  for (min_partner in grid_min_n) {
    for (min_state in grid_min_n) {
      for (min_country in grid_min_n) {
        for (lambda in grid_lambda) {
          fold_mets <- list()
          for (f in seq_along(folds)) {
            val_seasons <- folds[[f]]
            tr_seasons <- setdiff(train_seasons, val_seasons)

            tr_df <- df_cs |> dplyr::filter(season %in% tr_seasons)
            rare_partner <- get_rare_levels(tr_df$ballroom_partner, min_partner)
            rare_state <- get_rare_levels(tr_df$celebrity_homestate, min_state)
            rare_country <- get_rare_levels(tr_df$celebrity_homecountry_region, min_country)

            # age scaler from training fold
            age_center <- mean(tr_df$celebrity_age_during_season, na.rm = TRUE)
            age_scale <- sd(tr_df$celebrity_age_during_season, na.rm = TRUE)
            if (!is.finite(age_scale) || age_scale == 0) age_scale <- 1

            # Stage A: PL worths on training fold
            pl_fit <- fit_pl_worth(rankings_by_season[as.character(tr_seasons)], npseudo)
            logworth <- coef(pl_fit)  # log worths for items in training (named by item)

            # Stage B: ridge regression logworth ~ covariates
            cov_tr <- build_covariates(df_cs, tr_seasons, rare_partner, rare_state, rare_country, age_center, age_scale)
            # align y to cov rows
            y <- logworth[cov_tr$item_id]
            keep <- which(!is.na(y))
            cov_tr2 <- cov_tr[keep, , drop = FALSE]
            y2 <- as.numeric(y[keep])

            # model matrix
            mf <- model.frame(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c, data = cov_tr2, drop.unused.levels = TRUE)
            X <- model.matrix(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c, data = mf)

            # penalty: 0 for intercept + age + industry; lambda for shrinkage variables
            pen <- rep(lambda, ncol(X))
            pen[1] <- 0
            pen[grep("^age_z$", colnames(X))] <- 0
            pen[grep("^celebrity_industry", colnames(X))] <- 0

            beta <- ridge_fit(X, y2, pen)

            # Predict validation seasons
            cov_val <- build_covariates(df_cs, val_seasons, rare_partner, rare_state, rare_country, age_center, age_scale)
            mfv <- model.frame(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c, data = cov_val, drop.unused.levels = FALSE)
            Xv <- model.matrix(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c, data = mfv)
            # align columns (add missing with 0)
            missing_cols <- setdiff(colnames(X), colnames(Xv))
            if (length(missing_cols) > 0) {
              Xv <- cbind(Xv, matrix(0, nrow = nrow(Xv), ncol = length(missing_cols), dimnames = list(NULL, missing_cols)))
            }
            Xv <- Xv[, colnames(X), drop = FALSE]
            pred_logw <- predict_logworth(Xv, beta)
            names(pred_logw) <- cov_val$item_id

            for (s in val_seasons) {
              d_season <- df_cs |> dplyr::filter(season == s) |> dplyr::filter(!is.na(placement_final))
              met <- eval_season(d_season, pred_logw)
              met$season <- s
              met$fold <- f
              fold_mets[[length(fold_mets) + 1]] <- met
            }
          }

          mets <- dplyr::bind_rows(fold_mets)
          cv_rows[[length(cv_rows) + 1]] <- mets |>
            dplyr::summarise(
              npseudo = npseudo,
              min_partner = min_partner,
              min_state = min_state,
              min_country = min_country,
              lambda = lambda,
              spearman_mean = mean(spearman, na.rm = TRUE),
              kendall_mean = mean(kendall_tau, na.rm = TRUE),
              top1_mean = mean(top1, na.rm = TRUE),
              top3_mean = mean(top3, na.rm = TRUE),
              ndcg_mean = mean(ndcg, na.rm = TRUE)
            )
        }
      }
    }
  }
}

cv_grid <- dplyr::bind_rows(cv_rows) |>
  dplyr::arrange(desc(kendall_mean), desc(spearman_mean))

readr::write_csv(cv_grid, file.path(ART, "cv_hyperparams.csv"))
best <- cv_grid |> dplyr::slice(1)
readr::write_csv(best, file.path(ART, "best_hyperparams.csv"))

# Fit final model on all training seasons with best hyperparams
best_npseudo <- best$npseudo[[1]]
best_min_partner <- best$min_partner[[1]]
best_min_state <- best$min_state[[1]]
best_min_country <- best$min_country[[1]]
best_lambda <- best$lambda[[1]]

train_df <- df_cs |> dplyr::filter(season %in% train_seasons)
rare_partner <- get_rare_levels(train_df$ballroom_partner, best_min_partner)
rare_state <- get_rare_levels(train_df$celebrity_homestate, best_min_state)
rare_country <- get_rare_levels(train_df$celebrity_homecountry_region, best_min_country)

age_center <- mean(train_df$celebrity_age_during_season, na.rm = TRUE)
age_scale <- sd(train_df$celebrity_age_during_season, na.rm = TRUE)
if (!is.finite(age_scale) || age_scale == 0) age_scale <- 1

# Stage A: worth model
pl_final <- fit_pl_worth(rankings_by_season[as.character(train_seasons)], best_npseudo)
saveRDS(pl_final, file.path(ART, "pl_worth_train.rds"))

# Stage B: ridge regression on log-worths
cov_train <- build_covariates(df_cs, train_seasons, rare_partner, rare_state, rare_country, age_center, age_scale)
logworth <- coef(pl_final)
y <- logworth[cov_train$item_id]
keep <- which(!is.na(y))
cov_train2 <- cov_train[keep, , drop = FALSE]
y2 <- as.numeric(y[keep])

mf <- model.frame(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c, data = cov_train2, drop.unused.levels = TRUE)
X <- model.matrix(~ age_z + celebrity_industry + ballroom_partner_c + celebrity_homestate_c + celebrity_homecountry_region_c, data = mf)
pen <- rep(best_lambda, ncol(X))
pen[1] <- 0
pen[grep("^age_z$", colnames(X))] <- 0
pen[grep("^celebrity_industry", colnames(X))] <- 0
beta <- ridge_fit(X, y2, pen)

saveRDS(list(beta = beta, colnames = colnames(X), lambda = best_lambda, age_center = age_center, age_scale = age_scale), file.path(ART, "ridge_model.rds"))

coef_tbl <- tibble::tibble(
  term = colnames(X),
  estimate = as.numeric(beta),
  worth_multiplier = exp(as.numeric(beta))
)
readr::write_csv(coef_tbl, file.path(ART, "coefficients.csv"))

# Save the collapsing maps (so 03 can apply consistently)
readr::write_csv(tibble::tibble(level = rare_partner), file.path(ART, "rare_partner_levels.csv"))
readr::write_csv(tibble::tibble(level = rare_state), file.path(ART, "rare_state_levels.csv"))
readr::write_csv(tibble::tibble(level = rare_country), file.path(ART, "rare_country_levels.csv"))

readr::write_csv(tibble::tibble(
  npseudo = best_npseudo,
  min_partner = best_min_partner,
  min_state = best_min_state,
  min_country = best_min_country,
  lambda = best_lambda,
  age_center = age_center,
  age_scale = age_scale
), file.path(ART, "final_config.csv"))


