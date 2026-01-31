## Shared setup: local package library + common helpers.

options(stringsAsFactors = FALSE)

ROOT <- normalizePath("AT_P3/PLACKET_LUCE", mustWork = TRUE)
dir.create(file.path(ROOT, "artifacts"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(ROOT, "R_libs"), recursive = TRUE, showWarnings = FALSE)

.libPaths(c(file.path(ROOT, "R_libs"), .libPaths()))

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing package: %s", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
  }
  suppressWarnings(suppressMessages(library(pkg, character.only = TRUE)))
}

# Core stack
ensure_pkg("readr")
ensure_pkg("dplyr")
ensure_pkg("tidyr")
ensure_pkg("stringr")
ensure_pkg("purrr")
ensure_pkg("tibble")

# Modeling / metrics
ensure_pkg("PlackettLuce")
ensure_pkg("DescTools")   # KendallTauA
ensure_pkg("rlang")

collapse_rare_levels <- function(x, min_n, other = "Other") {
  x <- as.character(x)
  x[is.na(x) | x == ""] <- "Unknown"
  tab <- sort(table(x), decreasing = TRUE)
  rare <- names(tab)[tab < min_n]
  out <- ifelse(x %in% rare, other, x)
  out
}

standardize_columns <- function(df) {
  # Lowercase, trim, replace spaces with underscore, replace "/" with underscore
  raw <- names(df)
  norm <- raw |>
    stringr::str_trim() |>
    stringr::str_to_lower() |>
    stringr::str_replace_all("\\\\s+", "_") |>
    stringr::str_replace_all("[/]", "_") |>
    stringr::str_replace_all("__+", "_")
  lut <- setNames(raw, norm)

  pick <- function(cands) {
    for (c in cands) if (c %in% names(lut)) return(lut[[c]])
    NA_character_
  }

  map <- list(
    celebrity_name = c("celebrityname", "celebrity_name"),
    ballroom_partner = c("ballroompartner", "ballroom_partner"),
    celebrity_industry = c("celebrityindustry", "celebrity_industry"),
    celebrity_homestate = c("celebrityhomestate", "celebrity_homestate"),
    celebrity_homecountry_region = c("celebrityhomecountryregion", "celebrity_homecountry_region", "celebrity_homecountry_region"),
    celebrity_age_during_season = c("celebrityageduringseason", "celebrity_age_during_season"),
    season = c("season"),
    results = c("results"),
    placement = c("placement")
  )

  out <- df
  for (t in names(map)) {
    src <- pick(map[[t]])
    if (!is.na(src) && src != t) names(out)[names(out) == src] <- t
  }
  out
}

parse_place_ordinal <- function(x) {
  ifelse(
    is.na(x),
    NA_integer_,
    suppressWarnings(as.integer(stringr::str_match(x, "^(\\\\d+)(st|nd|rd|th)\\\\s+Place$")[, 2]))
  )
}

parse_elim_week <- function(x) {
  m <- stringr::str_match(x, "Eliminated\\\\s+Week\\\\s+(\\\\d+)")
  suppressWarnings(as.integer(m[, 2]))
}

ndcg_score <- function(truth_rank, pred_rank, k = NULL) {
  # Relevance = 1 / rank (higher for better placements)
  n <- length(truth_rank)
  if (is.null(k)) k <- n
  k <- min(k, n)
  rel <- 1 / truth_rank
  # DCG on predicted ordering
  ord <- order(pred_rank, decreasing = FALSE)
  rel_ord <- rel[ord][1:k]
  dcg <- sum((2^rel_ord - 1) / log2(seq_len(k) + 1))
  # Ideal DCG
  rel_ideal <- sort(rel, decreasing = TRUE)[1:k]
  idcg <- sum((2^rel_ideal - 1) / log2(seq_len(k) + 1))
  if (idcg == 0) return(NA_real_)
  dcg / idcg
}


