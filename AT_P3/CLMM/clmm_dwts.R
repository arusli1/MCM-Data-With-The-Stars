#!/usr/bin/env Rscript

## ------------------------------------------------------------
## DWTS Frequentist CLMM (Cumulative Link Mixed Model) Pipeline
## ------------------------------------------------------------
##
## Goal:
##   Predict overall performance (final placement) using contestant characteristics.
##   Primary model excludes season as a predictor (no fixed season effect); sensitivity
##   model adds (1 | season) as an additional random intercept.
##
## Inputs:
##   Data/2026_MCM_Problem_C_Data.csv
##
## Outputs (written to AT_P3/CLMM/):
##   - placement_audit.csv
##   - parsing_warnings.csv
##   - inconsistencies.csv
##   - df_cs.csv
##   - fixed_effects.csv
##   - industry_effects.csv
##   - random_effects_summary.csv
##   - partner_effects.csv (if extractable)
##   - cv_metrics.csv
##   - report.md
##
## NOTE:
##   This script will attempt to install missing packages into a local library:
##     AT_P3/CLMM/R_libs
##
## ------------------------------------------------------------

suppressWarnings(suppressMessages({
  options(stringsAsFactors = FALSE)
}))

here_dir <- normalizePath(dirname(commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]), mustWork = FALSE)
if (is.na(here_dir) || here_dir == "" || grepl("--file=", here_dir)) {
  # Fallback when run interactively
  here_dir <- normalizePath(file.path(getwd(), "AT_P3", "CLMM"), mustWork = FALSE)
}

out_dir <- here_dir
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

## -------------------------
## Local R library handling
## -------------------------
local_lib <- file.path(out_dir, "R_libs")
dir.create(local_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(local_lib, .libPaths()))

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing package: %s", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  suppressWarnings(suppressMessages(library(pkg, character.only = TRUE)))
}

# Required packages (keep lean-ish; we don't strictly need tidyverse meta-package)
pkgs <- c("readr", "dplyr", "tidyr", "stringr", "purrr", "ggplot2", "ordinal")
for (p in pkgs) ensure_pkg(p)

# Optional: tidyverse meta-package (requested). If it fails, we continue with the individual packages above.
tryCatch({
  ensure_pkg("tidyverse")
}, error = function(e) {
  message("NOTE: tidyverse meta-package not available; continuing with dplyr/readr/stringr/etc.")
})

# broom.mixed is nice-to-have for tidy tables; fallback to manual if install fails.
has_broom_mixed <- TRUE
tryCatch({
  ensure_pkg("broom.mixed")
}, error = function(e) {
  has_broom_mixed <<- FALSE
  message("NOTE: broom.mixed not available; will compute coefficient tables manually.")
})

## -------------------------
## Utility: column standardization
## -------------------------
standardize_columns <- function(df) {
  # Normalize column names: lowercase, trim, replace spaces with underscore, drop slashes
  raw_names <- names(df)
  norm <- raw_names |>
    stringr::str_trim() |>
    stringr::str_to_lower() |>
    stringr::str_replace_all("\\s+", "_") |>
    stringr::str_replace_all("[/]", "_") |>
    stringr::str_replace_all("__+", "_")

  # Map to internal schema
  map <- list(
    celebrity_name = c("celebrityname", "celebrity_name"),
    ballroom_partner = c("ballroompartner", "ballroom_partner"),
    celebrity_industry = c("celebrityindustry", "celebrity_industry"),
    celebrity_homestate = c("celebrityhomestate", "celebrity_homestate"),
    celebrity_homecountry_region = c("celebrityhomecountryregion", "celebrity_homecountry_region", "celebrity_homecountry_region", "celebrity_homecountry_region"),
    celebrity_age_during_season = c("celebrityageduringseason", "celebrity_age_during_season"),
    season = c("season"),
    results = c("results"),
    placement = c("placement")
  )

  # Build lookup from normalized names -> original names
  lut <- setNames(raw_names, norm)

  find_col <- function(cands) {
    for (c in cands) {
      if (c %in% names(lut)) return(lut[[c]])
    }
    return(NA_character_)
  }

  out <- df
  # Rename if found
  for (target in names(map)) {
    src <- find_col(map[[target]])
    if (!is.na(src) && src != target) {
      names(out)[names(out) == src] <- target
    }
  }

  # Handle the "celebrity_homecountry/region" original column which may come through as "celebrity_homecountry_region"
  # After normalization, we used slash->underscore, so it's already covered.

  return(out)
}

## -------------------------
## Placement parsing helpers
## -------------------------
parse_place_ordinal <- function(x) {
  # Parse "1st Place", "2nd Place", etc. -> integer
  ifelse(
    is.na(x),
    NA_integer_,
    suppressWarnings(as.integer(stringr::str_match(x, "^(\\d+)(st|nd|rd|th)\\s+Place$")[,2]))
  )
}

parse_elim_week <- function(x) {
  # Parse "Eliminated Week X" -> integer X
  m <- stringr::str_match(x, "Eliminated\\s+Week\\s+(\\d+)")
  w <- suppressWarnings(as.integer(m[,2]))
  w
}

## ------------------------------------------------------------
## Load + standardize
## ------------------------------------------------------------
data_path <- normalizePath(file.path(out_dir, "..", "..", "Data", "2026_MCM_Problem_C_Data.csv"), mustWork = TRUE)
df_raw <- readr::read_csv(data_path, show_col_types = FALSE, na = c("", "NA", "N/A"))
df <- standardize_columns(df_raw)

required_cols <- c(
  "celebrity_name","season","results","placement",
  "ballroom_partner","celebrity_industry","celebrity_homestate","celebrity_homecountry_region","celebrity_age_during_season"
)
missing_cols <- setdiff(required_cols, names(df))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing required columns after standardization: %s", paste(missing_cols, collapse = ", ")))
}

# Clean key string columns
df <- df |>
  dplyr::mutate(
    celebrity_name = stringr::str_squish(as.character(celebrity_name)),
    ballroom_partner = stringr::str_squish(as.character(ballroom_partner)),
    celebrity_industry = stringr::str_squish(as.character(celebrity_industry)),
    celebrity_homestate = stringr::str_squish(as.character(celebrity_homestate)),
    celebrity_homecountry_region = stringr::str_squish(as.character(celebrity_homecountry_region)),
    results = stringr::str_squish(as.character(results)),
    season = as.integer(season),
    placement = suppressWarnings(as.integer(placement)),
    celebrity_age_during_season = suppressWarnings(as.numeric(celebrity_age_during_season))
  )

## ------------------------------------------------------------
## Construct placement_final + audit
## ------------------------------------------------------------

# Compute per-row total judges score by week to support tie-breaking when parsing results
week_cols <- names(df) |> purrr::keep(~ stringr::str_detect(.x, "^week\\d+_judge\\d+_score$"))
weeks <- sort(unique(as.integer(stringr::str_match(week_cols, "^week(\\d+)_")[,2])))

week_total_cols <- c()
if (length(week_cols) > 0) {
  for (w in weeks) {
    cols_w <- week_cols[stringr::str_detect(week_cols, paste0("^week", w, "_"))]
    # Sum across judges for each row
    df[[paste0("week", w, "_judge_total")]] <- rowSums(df[, cols_w, drop = FALSE], na.rm = TRUE)
    week_total_cols <- c(week_total_cols, paste0("week", w, "_judge_total"))
  }
}

# Collapse to contestant-season level early for placement parsing
df_cs0 <- df |>
  dplyr::group_by(season, celebrity_name) |>
  dplyr::summarise(
    results = dplyr::first(na.omit(results)),
    placement_raw = dplyr::first(na.omit(placement)),
    last_nonzero_judge_total = {
      if (length(week_total_cols) == 0) {
        NA_real_
      } else {
        # last positive weekly total; if none positive, use NA
        # Use pick() (cur_data() is deprecated in dplyr >= 1.1).
        row <- dplyr::pick(dplyr::all_of(week_total_cols))
        vals <- as.numeric(row[1, , drop = TRUE])
        pos <- which(vals > 0)
        if (length(pos) == 0) NA_real_ else vals[max(pos)]
      }
    },
    .groups = "drop"
  )

# If placement exists, use it.
df_cs0 <- df_cs0 |>
  dplyr::mutate(
    placement_from_place = parse_place_ordinal(results),
    elim_week = parse_elim_week(results),
    is_withdrew = stringr::str_detect(results, "^Withdrew$"),
    placement_final = dplyr::case_when(
      !is.na(placement_raw) ~ placement_raw,
      !is.na(placement_from_place) ~ placement_from_place,
      TRUE ~ NA_integer_
    ),
    placement_source = dplyr::case_when(
      !is.na(placement_raw) ~ "placement_col",
      !is.na(placement_from_place) ~ "parsed_results_place",
      TRUE ~ NA_character_
    ),
    notes = dplyr::case_when(
      placement_source == "placement_col" ~ "",
      placement_source == "parsed_results_place" ~ "Parsed from '<k>th Place' string.",
      TRUE ~ ""
    )
  )

# For remaining missing placements, reconstruct within each season using elimination week ordering.
warnings <- list()

df_cs0 <- df_cs0 |>
  dplyr::group_by(season) |>
  dplyr::mutate(
    n_cast = dplyr::n()
  ) |>
  dplyr::ungroup()

df_cs0 <- df_cs0 |>
  dplyr::group_by(season) |>
  dplyr::mutate(
    # For reconstruction, define an "exit week" = elimination week if available; for withdrew treat as elim week if parsed.
    exit_week = dplyr::case_when(
      !is.na(elim_week) ~ elim_week,
      is_withdrew ~ elim_week,  # if withdrew has no week info, stays NA and handled below
      TRUE ~ NA_integer_
    )
  ) |>
  dplyr::ungroup()

df_cs0 <- df_cs0 |>
  dplyr::group_by(season) |>
  dplyr::mutate(
    # If we have any remaining NA placement_final in this season, we attempt reconstruction:
    placement_final = dplyr::if_else(
      is.na(placement_final),
      NA_integer_,
      placement_final
    )
  ) |>
  dplyr::ungroup()

reconstruct_season <- function(df_season) {
  # df_season: contestant-season rows for one season
  # We want a complete ordering: better finish => higher elimination week, then higher last_nonzero_judge_total, then name
  n <- nrow(df_season)
  # contestants already have placement_final => keep
  missing <- which(is.na(df_season$placement_final))
  if (length(missing) == 0) return(df_season)

  # If exit_week missing, treat as worst remaining (week 0)
  exit_week <- df_season$elim_week
  exit_week[is.na(exit_week)] <- 0L

  # Larger exit_week => better finish
  # tie-break: larger last_nonzero_judge_total => better
  # tie-break: celebrity_name ascending
  order_idx <- order(
    -exit_week,
    -ifelse(is.na(df_season$last_nonzero_judge_total), -Inf, df_season$last_nonzero_judge_total),
    df_season$celebrity_name
  )

  # Assign placements 1..n according to this ordering
  placement_recon <- integer(n)
  placement_recon[order_idx] <- seq_len(n)

  # Fill only missing placements
  df_season$placement_final[missing] <- placement_recon[missing]
  df_season$placement_source[missing] <- "parsed_results_elimweek"
  df_season$notes[missing] <- "Reconstructed within-season from elimination week + last judge total + name."

  # Warn if we had to use exit_week=0 (no elim week info)
  if (any(df_season$elim_week[missing] == 0)) {
    bad <- df_season[missing & df_season$elim_week == 0, c("celebrity_name","season","results"), drop = FALSE]
    if (nrow(bad) > 0) {
      warnings[[length(warnings) + 1]] <<- bad |>
        dplyr::mutate(warning = "Missing placement and could not parse elimination week; treated as worst remaining.")
    }
  }
  df_season
}

df_cs0 <- df_cs0 |>
  dplyr::group_by(season) |>
  dplyr::group_modify(~ reconstruct_season(.x)) |>
  dplyr::ungroup()

# Write audit + warnings
placement_audit <- df_cs0 |>
  dplyr::transmute(
    celebrity_name,
    season,
    results,
    placement = placement_raw,
    placement_final,
    placement_source,
    notes
  ) |>
  dplyr::arrange(season, placement_final, celebrity_name)

readr::write_csv(placement_audit, file.path(out_dir, "placement_audit.csv"))

parsing_warnings <- if (length(warnings) == 0) {
  tibble::tibble()
} else {
  dplyr::bind_rows(warnings)
}
readr::write_csv(parsing_warnings, file.path(out_dir, "parsing_warnings.csv"))

## ------------------------------------------------------------
## Reduce to one row per celebrity-season with predictor consistency checks
## ------------------------------------------------------------

mode_or_first <- function(x) {
  x <- x[!is.na(x) & x != ""]
  if (length(x) == 0) return(NA_character_)
  tab <- sort(table(x), decreasing = TRUE)
  names(tab)[1]
}

incons <- list()

df_cs <- df |>
  dplyr::group_by(season, celebrity_name) |>
  dplyr::summarise(
    ballroom_partner_vals = list(unique(na.omit(ballroom_partner))),
    celebrity_industry_vals = list(unique(na.omit(celebrity_industry))),
    celebrity_homestate_vals = list(unique(na.omit(celebrity_homestate))),
    celebrity_homecountry_region_vals = list(unique(na.omit(celebrity_homecountry_region))),
    celebrity_age_vals = list(unique(na.omit(celebrity_age_during_season))),
    .groups = "drop"
  ) |>
  dplyr::rowwise() |>
  dplyr::mutate(
    ballroom_partner = {
      v <- unlist(ballroom_partner_vals)
      if (length(unique(v)) > 1) {
        inconsistencies <- paste(unique(v), collapse = " | ")
        inconsist[[length(incons) + 1]] <<- tibble::tibble(
          season = season,
          celebrity_name = celebrity_name,
          field = "ballroom_partner",
          values = inconsistencies,
          resolution = "mode"
        )
      }
      mode_or_first(v)
    },
    celebrity_industry = {
      v <- unlist(celebrity_industry_vals)
      if (length(unique(v)) > 1) {
        inconsist[[length(incons) + 1]] <<- tibble::tibble(
          season = season,
          celebrity_name = celebrity_name,
          field = "celebrity_industry",
          values = paste(unique(v), collapse = " | "),
          resolution = "mode"
        )
      }
      mode_or_first(v)
    },
    celebrity_homestate = {
      v <- unlist(celebrity_homestate_vals)
      if (length(unique(v)) > 1) {
        inconsist[[length(incons) + 1]] <<- tibble::tibble(
          season = season,
          celebrity_name = celebrity_name,
          field = "celebrity_homestate",
          values = paste(unique(v), collapse = " | "),
          resolution = "mode"
        )
      }
      mode_or_first(v)
    },
    celebrity_homecountry_region = {
      v <- unlist(celebrity_homecountry_region_vals)
      if (length(unique(v)) > 1) {
        inconsist[[length(incons) + 1]] <<- tibble::tibble(
          season = season,
          celebrity_name = celebrity_name,
          field = "celebrity_homecountry_region",
          values = paste(unique(v), collapse = " | "),
          resolution = "mode"
        )
      }
      mode_or_first(v)
    },
    celebrity_age_during_season = {
      v <- unlist(celebrity_age_vals)
      if (length(unique(v)) > 1) {
        inconsist[[length(incons) + 1]] <<- tibble::tibble(
          season = season,
          celebrity_name = celebrity_name,
          field = "celebrity_age_during_season",
          values = paste(unique(v), collapse = " | "),
          resolution = "mode"
        )
      }
      # numeric mode: pick the most frequent rounded value, else first
      if (length(v) == 0) NA_real_ else as.numeric(names(sort(table(round(v, 6)), decreasing = TRUE))[1])
    }
  ) |>
  dplyr::ungroup() |>
  dplyr::select(season, celebrity_name, ballroom_partner, celebrity_industry, celebrity_homestate, celebrity_homecountry_region, celebrity_age_during_season)

incons_df <- if (length(incons) == 0) tibble::tibble() else dplyr::bind_rows(incons)
readr::write_csv(incons_df, file.path(out_dir, "inconsistencies.csv"))

# Join placement_final onto df_cs
df_cs <- df_cs |>
  dplyr::left_join(
    placement_audit |> dplyr::select(season, celebrity_name, placement_final, placement_source),
    by = c("season","celebrity_name")
  )

readr::write_csv(df_cs, file.path(out_dir, "df_cs.csv"))

## ------------------------------------------------------------
## Model prep
## ------------------------------------------------------------
df_model <- df_cs |>
  dplyr::filter(!is.na(placement_final)) |>
  # IMPORTANT: clmm() will drop rows with NA in any grouping factor, which can collapse
  # a random-effect factor down to <=2 levels and make the model un-identifiable.
  # We therefore convert missing/blank group labels to an explicit "Unknown" level.
  dplyr::mutate(
    ballroom_partner = dplyr::if_else(is.na(ballroom_partner) | ballroom_partner == "", "Unknown", ballroom_partner),
    celebrity_homestate = dplyr::if_else(is.na(celebrity_homestate) | celebrity_homestate == "", "Unknown", celebrity_homestate),
    celebrity_homecountry_region = dplyr::if_else(is.na(celebrity_homecountry_region) | celebrity_homecountry_region == "", "Unknown", celebrity_homecountry_region),
    celebrity_industry = dplyr::if_else(is.na(celebrity_industry) | celebrity_industry == "", "Unknown", celebrity_industry)
  ) |>
  dplyr::mutate(
    placement_final = as.integer(placement_final),
    placement_ord = factor(placement_final, levels = sort(unique(placement_final)), ordered = TRUE),
    age_z = as.numeric(scale(celebrity_age_during_season)),
    # Industry can have very sparse levels (n=1–2), which often causes quasi-separation and
    # singular Hessians in ordinal models. We lump rare industries into "Other" (min count = 5)
    # to produce stable frequentist standard errors.
    celebrity_industry_raw = celebrity_industry,
    ballroom_partner = factor(ballroom_partner),
    celebrity_homestate = factor(celebrity_homestate),
    celebrity_homecountry_region = factor(celebrity_homecountry_region),
    season = factor(season)
  )

# Lump rare industry levels (min count threshold)
industry_counts <- df_model |>
  dplyr::count(celebrity_industry_raw, sort = TRUE, name = "n")
readr::write_csv(industry_counts, file.path(out_dir, "industry_counts.csv"))

rare_ind <- industry_counts |>
  dplyr::filter(n < 5) |>
  dplyr::pull(celebrity_industry_raw)

df_model <- df_model |>
  dplyr::mutate(
    celebrity_industry = dplyr::if_else(celebrity_industry_raw %in% rare_ind, "Other", celebrity_industry_raw)
  )

# Choose baseline industry as the most frequent (after lumping) for interpretability.
baseline_ind <- df_model |>
  dplyr::count(celebrity_industry, sort = TRUE) |>
  dplyr::slice(1) |>
  dplyr::pull(celebrity_industry)

df_model <- df_model |>
  dplyr::mutate(
    celebrity_industry = factor(celebrity_industry),
    celebrity_industry = stats::relevel(celebrity_industry, ref = baseline_ind)
  )

## Missingness summary for report
missing_summary <- df_model |>
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_seasons = dplyr::n_distinct(season),
    n_contestants = dplyr::n_distinct(celebrity_name),
    missing_age = sum(is.na(celebrity_age_during_season)),
    missing_industry = sum(is.na(celebrity_industry))
  )

## ------------------------------------------------------------
## Fit primary CLMM (NO season random effect)
## ------------------------------------------------------------

fit_primary <- ordinal::clmm(
  placement_ord ~ age_z + celebrity_industry +
    (1 | ballroom_partner) +
    (1 | celebrity_homestate) +
    (1 | celebrity_homecountry_region),
  data = df_model,
  link = "logit",
  Hess = TRUE
)

fit_sens <- ordinal::clmm(
  placement_ord ~ age_z + celebrity_industry +
    (1 | ballroom_partner) +
    (1 | celebrity_homestate) +
    (1 | celebrity_homecountry_region) +
    (1 | season),
  data = df_model,
  link = "logit",
  Hess = TRUE
)

## ------------------------------------------------------------
## Extract fixed effects / random effects summaries
## ------------------------------------------------------------

extract_fixed <- function(fit) {
  # summary.clmm can fail to provide a variance-covariance matrix when the Hessian is singular.
  # In that case, we still return point estimates with NA for SE/z/p/CI.
  s <- tryCatch(summary(fit), error = function(e) NULL)
  if (!is.null(s) && !is.null(s$coefficients)) {
    coefs <- as.data.frame(s$coefficients)
    coefs$term <- rownames(coefs)
    rownames(coefs) <- NULL
    out <- coefs |>
      dplyr::rename(
        estimate = Estimate,
        std_error = `Std. Error`,
        z = `z value`,
        p_value = `Pr(>|z|)`
      ) |>
      dplyr::mutate(
        odds_ratio = exp(estimate),
        ci_low = exp(estimate - 1.96 * std_error),
        ci_high = exp(estimate + 1.96 * std_error)
      ) |>
      dplyr::select(term, estimate, std_error, z, p_value, odds_ratio, ci_low, ci_high)
    return(out)
  }

  # Fallback: estimates only
  beta <- fit$beta
  tibble::tibble(
    term = names(beta),
    estimate = as.numeric(beta),
    std_error = NA_real_,
    z = NA_real_,
    p_value = NA_real_,
    odds_ratio = exp(as.numeric(beta)),
    ci_low = NA_real_,
    ci_high = NA_real_
  )
}

fixed_primary <- extract_fixed(fit_primary)
readr::write_csv(fixed_primary, file.path(out_dir, "fixed_effects.csv"))

# Sensitivity fixed effects (with season random intercept)
fixed_sens <- extract_fixed(fit_sens)
readr::write_csv(fixed_sens, file.path(out_dir, "fixed_effects_sensitivity.csv"))

# Industry coefficients vs baseline (terms start with celebrity_industry)
industry_effects <- fixed_primary |>
  dplyr::filter(stringr::str_starts(term, "celebrity_industry")) |>
  dplyr::mutate(
    industry = stringr::str_replace(term, "^celebrity_industry", ""),
    baseline = levels(df_model$celebrity_industry)[1]
  ) |>
  dplyr::select(industry, baseline, estimate, std_error, z, p_value, odds_ratio, ci_low, ci_high)
readr::write_csv(industry_effects, file.path(out_dir, "industry_effects.csv"))

extract_re <- function(fit) {
  # For ordinal::clmm, VarCorr() returns a *named list* of 1x1 variance matrices, each with attr("stddev").
  vc <- tryCatch(ordinal::VarCorr(fit), error = function(e) NULL)
  if (is.null(vc)) return(tibble::tibble())

  groups <- names(vc)
  out <- purrr::map_dfr(groups, function(g) {
    m <- vc[[g]]
    var <- suppressWarnings(as.numeric(m[1, 1]))
    sd <- suppressWarnings(as.numeric(attr(m, "stddev")[1]))
    tibble::tibble(group = g, variance = var, sd = sd)
  })
  out
}

re_primary_raw <- extract_re(fit_primary)
re_sens_raw <- extract_re(fit_sens)

summarize_re <- function(vdf, model_name) {
  if (nrow(vdf) == 0) return(tibble::tibble())
  vdf |>
    dplyr::transmute(
      model = model_name,
      group = .data$group,
      variance = as.numeric(.data$variance),
      sd = as.numeric(.data$sd)
    ) |>
    dplyr::arrange(group)
}

re_summary <- dplyr::bind_rows(
  summarize_re(re_primary_raw, "primary_no_season"),
  summarize_re(re_sens_raw, "sensitivity_with_season")
)
readr::write_csv(re_summary, file.path(out_dir, "random_effects_summary.csv"))

# Partner BLUPs / conditional modes (if available)
partner_effects <- tibble::tibble()
tryCatch({
  r <- ordinal::ranef(fit_primary)
  if (!is.null(r$ballroom_partner)) {
    pe <- tibble::tibble(
      ballroom_partner = rownames(r$ballroom_partner),
      blup = as.numeric(r$ballroom_partner[,1])
    ) |>
      dplyr::arrange(blup)
    partner_effects <<- pe
  }
}, error = function(e) {
  partner_effects <<- tibble::tibble(note = "partner ranef not extractable", error = as.character(e))
})
readr::write_csv(partner_effects, file.path(out_dir, "partner_effects.csv"))

## ------------------------------------------------------------
## 5-fold CV (row-level)
## ------------------------------------------------------------

set.seed(123)
k <- 5
n <- nrow(df_model)
fold_id <- sample(rep(1:k, length.out = n))

# Fixed-effects-only prediction for ordinal logit:
#   P(Y <= j) = logistic(theta_j - eta), where eta = X beta
predict_probs_fixed_only <- function(fit, newdata, outcome_levels) {
  beta <- fit$beta
  theta <- fit$Theta

  # Build fixed-effect model matrix consistent with fit
  # Use the fit's formula but remove random effects terms by reformulating:
  # We'll use the terms from fit$terms but this includes random effects; instead, we rebuild:
  # placement_ord ~ age_z + celebrity_industry
  mm <- model.matrix(~ age_z + celebrity_industry, data = newdata)

  # Align columns to beta names
  bnames <- names(beta)
  mm2 <- mm[, bnames, drop = FALSE]
  eta <- as.numeric(mm2 %*% beta)

  # theta is length (K-1) for K categories
  K <- length(outcome_levels)
  # Cumulative probs
  cum <- sapply(theta, function(th) plogis(th - eta))
  # cum: n x (K-1)
  cum <- as.matrix(cum)

  # Convert to category probs:
  # p1 = cum1
  # pk = cum_k - cum_{k-1}
  # pK = 1 - cum_{K-1}
  probs <- matrix(NA_real_, nrow = nrow(newdata), ncol = K)
  probs[,1] <- cum[,1]
  if (K > 2) {
    for (j in 2:(K-1)) {
      probs[,j] <- cum[,j] - cum[,j-1]
    }
  }
  probs[,K] <- 1 - cum[,K-1]

  colnames(probs) <- outcome_levels
  probs
}

# Conditional prediction: add random-intercept BLUPs for group levels seen in training.
# Unseen levels in the test fold default to 0 (the population mean).
predict_probs_conditional <- function(fit, newdata, outcome_levels) {
  beta <- fit$beta
  theta <- fit$Theta

  # Fixed effects contribution
  mm <- model.matrix(~ age_z + celebrity_industry, data = newdata)
  mm2 <- mm[, names(beta), drop = FALSE]
  eta <- as.numeric(mm2 %*% beta)

  # Random effects contribution (conditional modes / BLUPs)
  re <- tryCatch(ordinal::ranef(fit), error = function(e) NULL)
  if (!is.null(re)) {
    add_re <- function(name, col_in_newdata) {
      if (is.null(re[[name]])) return(rep(0, nrow(newdata)))
      m <- re[[name]]
      v <- setNames(as.numeric(m[, 1]), rownames(m))
      key <- as.character(newdata[[col_in_newdata]])
      u <- v[key]
      u[is.na(u)] <- 0
      u
    }
    eta <- eta +
      add_re("ballroom_partner", "ballroom_partner") +
      add_re("celebrity_homestate", "celebrity_homestate") +
      add_re("celebrity_homecountry_region", "celebrity_homecountry_region")
  }

  # Cumulative probs under clmm parameterization: P(Y<=k)=logit^{-1}(theta_k - eta)
  K <- length(outcome_levels)
  cum <- sapply(theta, function(th) plogis(th - eta))
  cum <- as.matrix(cum)

  probs <- matrix(NA_real_, nrow = nrow(newdata), ncol = K)
  probs[, 1] <- cum[, 1]
  if (K > 2) {
    for (j in 2:(K - 1)) {
      probs[, j] <- cum[, j] - cum[, j - 1]
    }
  }
  probs[, K] <- 1 - cum[, K - 1]
  colnames(probs) <- outcome_levels
  probs
}

## ------------------------------------------------------------
## Full-data predictions (per contestant-season)
## ------------------------------------------------------------
##
## We compute:
##  - marginal predictions: fixed effects only (random effects = 0)
##  - conditional predictions: fixed effects + BLUPs for partner/state/country
## Then:
##  - predicted class = argmax P(Y=k)
##  - expected placement = sum_k k * P(Y=k)
##  - predicted rank within season = rank(expected placement), lower expected => better rank
##
pred_outcome_levels <- levels(df_model$placement_ord)

P_m_full <- predict_probs_fixed_only(fit_primary, df_model, pred_outcome_levels)
P_c_full <- predict_probs_conditional(fit_primary, df_model, pred_outcome_levels)

lev_int_full <- as.integer(pred_outcome_levels)

pred_class_m_full <- apply(P_m_full, 1, function(p) pred_outcome_levels[which.max(p)])
pred_class_c_full <- apply(P_c_full, 1, function(p) pred_outcome_levels[which.max(p)])

expected_m_full <- as.numeric(P_m_full %*% lev_int_full)
expected_c_full <- as.numeric(P_c_full %*% lev_int_full)

predictions <- df_model |>
  dplyr::transmute(
    season = as.integer(as.character(season)),
    celebrity_name,
    true_placement = as.integer(as.character(placement_ord)),
    pred_class_marginal = as.integer(pred_class_m_full),
    expected_placement_marginal = expected_m_full,
    pred_class_conditional = as.integer(pred_class_c_full),
    expected_placement_conditional = expected_c_full
  ) |>
  dplyr::group_by(season) |>
  dplyr::mutate(
    predicted_rank_marginal = dplyr::min_rank(expected_placement_marginal),
    predicted_rank_conditional = dplyr::min_rank(expected_placement_conditional)
  ) |>
  dplyr::ungroup() |>
  dplyr::arrange(season, predicted_rank_conditional, predicted_rank_marginal, celebrity_name)

readr::write_csv(predictions, file.path(out_dir, "predicted_rankings.csv"))

cv_rows <- list()

for (fold in 1:k) {
  train <- df_model[fold_id != fold, , drop = FALSE]
  test  <- df_model[fold_id == fold, , drop = FALSE]

  # Fit model on training fold.
  fit_cv <- ordinal::clmm(
    placement_ord ~ age_z + celebrity_industry +
      (1 | ballroom_partner) +
      (1 | celebrity_homestate) +
      (1 | celebrity_homecountry_region),
    data = train,
    link = "logit",
    Hess = TRUE
  )

  levs <- levels(df_model$placement_ord)
  probs_marginal <- predict_probs_fixed_only(fit_cv, test, levs)
  probs_cond <- predict_probs_conditional(fit_cv, test, levs)

  # Predicted class = argmax prob
  pred_class_m <- apply(probs_marginal, 1, function(p) levs[which.max(p)])
  pred_class_m_int <- as.integer(pred_class_m)

  pred_class_c <- apply(probs_cond, 1, function(p) levs[which.max(p)])
  pred_class_c_int <- as.integer(pred_class_c)

  true_int <- as.integer(as.character(test$placement_ord))

  # Expected placement
  lev_int <- as.integer(levs)
  expected_m <- as.numeric(probs_marginal %*% lev_int)
  expected_c <- as.numeric(probs_cond %*% lev_int)

  exact_acc_m <- mean(pred_class_m_int == true_int)
  within1_m <- mean(abs(pred_class_m_int - true_int) <= 1)
  mae_m <- mean(abs(expected_m - true_int))
  spear_m <- suppressWarnings(cor(expected_m, true_int, method = "spearman"))

  exact_acc_c <- mean(pred_class_c_int == true_int)
  within1_c <- mean(abs(pred_class_c_int - true_int) <= 1)
  mae_c <- mean(abs(expected_c - true_int))
  spear_c <- suppressWarnings(cor(expected_c, true_int, method = "spearman"))

  cv_rows[[length(cv_rows)+1]] <- tibble::tibble(
    fold = as.character(fold),
    n_test = nrow(test),
    exact_accuracy_marginal = exact_acc_m,
    within1_accuracy_marginal = within1_m,
    mae_expected_marginal = mae_m,
    spearman_expected_vs_true_marginal = spear_m,
    exact_accuracy_conditional = exact_acc_c,
    within1_accuracy_conditional = within1_c,
    mae_expected_conditional = mae_c,
    spearman_expected_vs_true_conditional = spear_c
  )
}

cv_metrics <- dplyr::bind_rows(cv_rows)

cv_summary <- cv_metrics |>
  dplyr::summarise(
    fold = "overall_mean",
    n_test = sum(n_test),
    exact_accuracy_marginal = mean(exact_accuracy_marginal),
    within1_accuracy_marginal = mean(within1_accuracy_marginal),
    mae_expected_marginal = mean(mae_expected_marginal),
    spearman_expected_vs_true_marginal = mean(spearman_expected_vs_true_marginal),
    exact_accuracy_conditional = mean(exact_accuracy_conditional),
    within1_accuracy_conditional = mean(within1_accuracy_conditional),
    mae_expected_conditional = mean(mae_expected_conditional),
    spearman_expected_vs_true_conditional = mean(spearman_expected_vs_true_conditional)
  )

cv_sd <- cv_metrics |>
  dplyr::summarise(
    fold = "overall_sd",
    n_test = NA_real_,
    exact_accuracy_marginal = sd(exact_accuracy_marginal),
    within1_accuracy_marginal = sd(within1_accuracy_marginal),
    mae_expected_marginal = sd(mae_expected_marginal),
    spearman_expected_vs_true_marginal = sd(spearman_expected_vs_true_marginal),
    exact_accuracy_conditional = sd(exact_accuracy_conditional),
    within1_accuracy_conditional = sd(within1_accuracy_conditional),
    mae_expected_conditional = sd(mae_expected_conditional),
    spearman_expected_vs_true_conditional = sd(spearman_expected_vs_true_conditional)
  )

cv_out <- dplyr::bind_rows(cv_metrics, cv_summary, cv_sd)
readr::write_csv(cv_out, file.path(out_dir, "cv_metrics.csv"))

## ------------------------------------------------------------
## Coefficient direction sanity check (computed)
## ------------------------------------------------------------
##
## ordinal::clmm uses the parameterization:
##   logit(P(Y <= k)) = theta_k - eta
## so increasing eta DECREASES P(Y <= k). Because our outcome is ordered 1 < 2 < ...,
## decreasing P(Y <= 3) means *less* probability of finishing top-3 (worse).
##
## We compute P(Y <= 3) at age_z = -1 vs +1 holding other predictors at baseline.
##
mk_new <- function(age_z_val) {
  tibble::tibble(
    age_z = age_z_val,
    celebrity_industry = factor(baseline_ind, levels = levels(df_model$celebrity_industry)),
    ballroom_partner = factor(levels(df_model$ballroom_partner)[1], levels = levels(df_model$ballroom_partner)),
    celebrity_homestate = factor(levels(df_model$celebrity_homestate)[1], levels = levels(df_model$celebrity_homestate)),
    celebrity_homecountry_region = factor(levels(df_model$celebrity_homecountry_region)[1], levels = levels(df_model$celebrity_homecountry_region))
  )
}
nd_dir <- dplyr::bind_rows(mk_new(-1), mk_new(+1))
P_dir <- predict_probs_fixed_only(fit_primary, nd_dir, levels(df_model$placement_ord))
top3_cols <- intersect(colnames(P_dir), c("1", "2", "3"))
p_top3_young <- sum(P_dir[1, top3_cols])
p_top3_old <- sum(P_dir[2, top3_cols])

## ------------------------------------------------------------
## Report.md (real computed values)
## ------------------------------------------------------------

fmt <- function(x, digits = 3) {
  ifelse(is.na(x), "NA", formatC(x, format = "f", digits = digits))
}

report_lines <- c()
report_lines <- c(report_lines, "# DWTS Cumulative Link Mixed Model (CLMM) Report", "")

report_lines <- c(report_lines, "## Data cleaning & construction", "")
report_lines <- c(report_lines, sprintf("- Source rows (raw): **%d**", nrow(df)))
report_lines <- c(report_lines, sprintf("- Contestant-season rows (df_cs): **%d**", nrow(df_cs)))
report_lines <- c(report_lines, sprintf("- Seasons present: **%d**", dplyr::n_distinct(df_cs$season)))
report_lines <- c(report_lines, sprintf("- Unique contestants: **%d**", dplyr::n_distinct(df_cs$celebrity_name)))
report_lines <- c(report_lines, sprintf("- Missing `placement_final` after parsing: **%d**", sum(is.na(df_cs$placement_final))))
report_lines <- c(report_lines, "")

src_counts <- placement_audit |>
  dplyr::count(placement_source) |>
  dplyr::mutate(placement_source = ifelse(is.na(placement_source), "NA", placement_source))
report_lines <- c(report_lines, "Placement source breakdown (`placement_audit.csv`):", "")
for (i in seq_len(nrow(src_counts))) {
  report_lines <- c(report_lines, sprintf("- %s: %d", src_counts$placement_source[i], src_counts$n[i]))
}
report_lines <- c(report_lines, "")

report_lines <- c(report_lines, "## Model specification", "")
report_lines <- c(report_lines, "**Primary model (no season):**", "")
report_lines <- c(report_lines,
  "- Outcome: `placement_final` as ordered factor (1 is best).",
  "- Link: logit.",
  "- Fixed effects: standardized age + industry factor.",
  "- Random intercepts: partner, homestate, homecountry/region.",
  ""
)
report_lines <- c(report_lines, "**Sensitivity model:** same, plus `(1 | season)` random intercept.", "")

report_lines <- c(report_lines, "## Cross-validation (5-fold)", "")
report_lines <- c(report_lines,
  "We report two prediction modes:",
  "- **Marginal (population-level)**: random effects set to 0 (robust for unseen group levels).",
  "- **Conditional (BLUP-assisted)**: adds estimated random intercepts for partner/state/country when that level was seen in the training fold; unseen levels fall back to 0.",
  ""
)
report_lines <- c(report_lines, "Summary (mean ± sd across folds):", "")
report_lines <- c(report_lines,
  "Marginal:",
  sprintf("- Exact accuracy: **%s ± %s**", fmt(cv_summary$exact_accuracy_marginal), fmt(cv_sd$exact_accuracy_marginal)),
  sprintf("- Within-1 accuracy: **%s ± %s**", fmt(cv_summary$within1_accuracy_marginal), fmt(cv_sd$within1_accuracy_marginal)),
  sprintf("- MAE (expected placement): **%s ± %s**", fmt(cv_summary$mae_expected_marginal), fmt(cv_sd$mae_expected_marginal)),
  sprintf("- Spearman(expected, true): **%s ± %s**", fmt(cv_summary$spearman_expected_vs_true_marginal), fmt(cv_sd$spearman_expected_vs_true_marginal)),
  "",
  "Conditional (BLUP-assisted):",
  sprintf("- Exact accuracy: **%s ± %s**", fmt(cv_summary$exact_accuracy_conditional), fmt(cv_sd$exact_accuracy_conditional)),
  sprintf("- Within-1 accuracy: **%s ± %s**", fmt(cv_summary$within1_accuracy_conditional), fmt(cv_sd$within1_accuracy_conditional)),
  sprintf("- MAE (expected placement): **%s ± %s**", fmt(cv_summary$mae_expected_conditional), fmt(cv_sd$mae_expected_conditional)),
  sprintf("- Spearman(expected, true): **%s ± %s**", fmt(cv_summary$spearman_expected_vs_true_conditional), fmt(cv_sd$spearman_expected_vs_true_conditional)),
  ""
)

report_lines <- c(report_lines, "## Fixed effects (primary model)", "")
report_lines <- c(report_lines, "See `fixed_effects.csv` for full table (log-odds scale, OR with 95% CI).", "")

age_row <- fixed_primary |>
  dplyr::filter(term == "age_z")
if (nrow(age_row) == 1) {
  report_lines <- c(report_lines,
    sprintf("- Age (z-scored): estimate=%s, OR=%s, p=%s",
      fmt(age_row$estimate), fmt(age_row$odds_ratio), formatC(age_row$p_value, format="g", digits=3)
    )
  )
}
report_lines <- c(report_lines, "")

report_lines <- c(report_lines, "### Coefficient direction sanity check (computed)", "")
report_lines <- c(report_lines,
  "For `clmm` (logit link), the model uses \\(\\text{logit}(P(Y \\le k)) = \\theta_k - \\eta\\).",
  "With `placement_ord` ordered as 1 < 2 < 3 < ... (1 is best), a **positive** coefficient increases \\(\\eta\\) and therefore **decreases** \\(P(Y \\le k)\\), i.e., shifts mass toward worse placements.",
  ""
)
report_lines <- c(report_lines,
  sprintf("- Holding industry/partner/state/country at baseline and using fixed-effects-only probabilities:"),
  sprintf("  - P(finish top-3) at age_z = -1: **%s**", fmt(p_top3_young)),
  sprintf("  - P(finish top-3) at age_z = +1: **%s**", fmt(p_top3_old)),
  ""
)

report_lines <- c(report_lines, "## Sensitivity: adding `(1 | season)`", "")
report_lines <- c(report_lines, "See `fixed_effects_sensitivity.csv` for the full sensitivity fixed-effects table.", "")
age_sens <- fixed_sens |> dplyr::filter(term == "age_z")
if (nrow(age_sens) == 1) {
  report_lines <- c(report_lines,
    sprintf("- Age (z-scored) sensitivity: estimate=%s, OR=%s, p=%s",
      fmt(age_sens$estimate), fmt(age_sens$odds_ratio), formatC(age_sens$p_value, format="g", digits=3)
    )
  )
}
report_lines <- c(report_lines,
  "",
  "Interpretation: if the age/industry effects keep the **same sign and similar magnitude** under the sensitivity model,",
  "the primary conclusions are not driven purely by between-season differences.",
  ""
)

report_lines <- c(report_lines, "## Random effects", "")
report_lines <- c(report_lines, "See `random_effects_summary.csv` for variance/SD by grouping factor.", "")
report_lines <- c(report_lines, "")

if (nrow(partner_effects) > 0 && "blup" %in% names(partner_effects)) {
  top_best <- partner_effects |>
    dplyr::arrange(blup) |>
    dplyr::slice_head(n = 10)
  top_worst <- partner_effects |>
    dplyr::arrange(desc(blup)) |>
    dplyr::slice_head(n = 10)
  report_lines <- c(report_lines, "### Ballroom partner BLUPs (primary model; interpret cautiously)", "")
  report_lines <- c(report_lines, "**Top 10 (most negative random intercept; tends to better placements):**", "")
  for (i in seq_len(nrow(top_best))) {
    report_lines <- c(report_lines, sprintf("- %s: %s", top_best$ballroom_partner[i], fmt(top_best$blup[i], 4)))
  }
  report_lines <- c(report_lines, "", "**Bottom 10 (most positive random intercept; tends to worse placements):**", "")
  for (i in seq_len(nrow(top_worst))) {
    report_lines <- c(report_lines, sprintf("- %s: %s", top_worst$ballroom_partner[i], fmt(top_worst$blup[i], 4)))
  }
  report_lines <- c(report_lines, "")
} else {
  report_lines <- c(report_lines, "### Ballroom partner BLUPs", "")
  report_lines <- c(report_lines, "Partner random effects were not extractable in this run; see `partner_effects.csv` for details.", "")
}

report_lines <- c(report_lines, "## Limitations", "")
report_lines <- c(report_lines,
  "- Ordinal proportional-odds assumption may be violated (effect sizes assumed constant across thresholds).",
  "- Cross-validation uses fixed-effects-only probabilities for robustness to unseen random-effect levels in test folds.",
  "- Some random-effect levels are sparse; variance estimates and BLUP rankings are shrinkage-regularized and should be interpreted cautiously.",
  "- Placement reconstruction from `results` is implemented for completeness, but this dataset largely provides a numeric `placement` column.",
  ""
)

writeLines(report_lines, con = file.path(out_dir, "report.md"))

## ------------------------------------------------------------
## Console summary (short)
## ------------------------------------------------------------
message("=== CLMM pipeline complete ===")
message(sprintf("Wrote outputs to: %s", out_dir))
message(sprintf("Primary model n=%d; seasons=%d; contestants=%d", nrow(df_model), dplyr::n_distinct(df_model$season), dplyr::n_distinct(df_model$celebrity_name)))
message(sprintf("CV marginal exact accuracy (mean±sd): %s ± %s", fmt(cv_summary$exact_accuracy_marginal), fmt(cv_sd$exact_accuracy_marginal)))
message(sprintf("CV marginal within-1 accuracy (mean±sd): %s ± %s", fmt(cv_summary$within1_accuracy_marginal), fmt(cv_sd$within1_accuracy_marginal)))
message(sprintf("CV conditional exact accuracy (mean±sd): %s ± %s", fmt(cv_summary$exact_accuracy_conditional), fmt(cv_sd$exact_accuracy_conditional)))
message(sprintf("CV conditional within-1 accuracy (mean±sd): %s ± %s", fmt(cv_summary$within1_accuracy_conditional), fmt(cv_sd$within1_accuracy_conditional)))


