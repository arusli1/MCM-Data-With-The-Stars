## Build modeling dataset + season rankings objects (PlackettLuce).

DATA_PATH <- normalizePath("Data/2026_MCM_Problem_C_Data.csv", mustWork = TRUE)
ART <- file.path(ROOT, "artifacts")

df_raw <- readr::read_csv(DATA_PATH, show_col_types = FALSE, na = c("", "NA", "N/A"))
df <- standardize_columns(df_raw)

required <- c(
  "celebrity_name","season","results","placement",
  "ballroom_partner","celebrity_industry","celebrity_homestate","celebrity_homecountry_region","celebrity_age_during_season"
)
missing <- setdiff(required, names(df))
if (length(missing) > 0) stop(paste("Missing required columns:", paste(missing, collapse=", ")))

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

# One row per contestant-season (dataset is already wide; still enforce uniqueness)
df_cs <- df |>
  dplyr::group_by(season, celebrity_name) |>
  dplyr::summarise(
    ballroom_partner = dplyr::first(na.omit(ballroom_partner)),
    celebrity_industry = dplyr::first(na.omit(celebrity_industry)),
    celebrity_homestate = dplyr::first(na.omit(celebrity_homestate)),
    celebrity_homecountry_region = dplyr::first(na.omit(celebrity_homecountry_region)),
    celebrity_age_during_season = dplyr::first(na.omit(celebrity_age_during_season)),
    results = dplyr::first(na.omit(results)),
    placement_raw = dplyr::first(na.omit(placement)),
    .groups = "drop"
  )

# Construct placement_final: prefer placement column; otherwise parse results.
df_cs <- df_cs |>
  dplyr::mutate(
    placement_from_place = parse_place_ordinal(results),
    elim_week = parse_elim_week(results),
    placement_final = dplyr::case_when(
      !is.na(placement_raw) ~ placement_raw,
      !is.na(placement_from_place) ~ placement_from_place,
      TRUE ~ NA_integer_
    ),
    placement_source = dplyr::case_when(
      !is.na(placement_raw) ~ "placement_col",
      !is.na(placement_from_place) ~ "parsed_results_place",
      TRUE ~ "missing"
    )
  )

# If still missing within a season, reconstruct from elimination week (higher week = better).
reconstruct_season <- function(d) {
  miss <- which(is.na(d$placement_final))
  if (length(miss) == 0) return(d)
  exit_week <- d$elim_week
  exit_week[is.na(exit_week)] <- 0L
  ord <- order(-exit_week, d$celebrity_name)
  rec <- integer(nrow(d))
  rec[ord] <- seq_len(nrow(d))
  d$placement_final[miss] <- rec[miss]
  d$placement_source[miss] <- "parsed_results_elimweek"
  d
}
df_cs <- df_cs |>
  dplyr::group_by(season) |>
  dplyr::group_modify(~ reconstruct_season(.x)) |>
  dplyr::ungroup()

placement_audit <- df_cs |>
  dplyr::transmute(
    season, celebrity_name, results,
    placement = placement_raw,
    placement_final,
    placement_source
  ) |>
  dplyr::arrange(season, placement_final, celebrity_name)

readr::write_csv(placement_audit, file.path(ART, "placement_audit.csv"))

# Create unique item_id per contestant-season (prevents accidental cross-season “item” identity)
df_cs <- df_cs |>
  dplyr::mutate(
    item_id = paste0("S", season, ":", celebrity_name),
    ballroom_partner = ifelse(is.na(ballroom_partner) | ballroom_partner == "", "Unknown", ballroom_partner),
    celebrity_homestate = ifelse(is.na(celebrity_homestate) | celebrity_homestate == "", "Unknown", celebrity_homestate),
    celebrity_homecountry_region = ifelse(is.na(celebrity_homecountry_region) | celebrity_homecountry_region == "", "Unknown", celebrity_homecountry_region),
    celebrity_industry = ifelse(is.na(celebrity_industry) | celebrity_industry == "", "Unknown", celebrity_industry)
  )

readr::write_csv(df_cs, file.path(ART, "contestant_season.csv"))
saveRDS(df_cs, file.path(ART, "contestant_season.rds"))

# Build rankings for each season: best-to-worst by placement_final
make_season_rankings <- function(df_season) {
  df_season <- df_season |>
    dplyr::arrange(placement_final, celebrity_name)
  # PlackettLuce expects rankings in *rank form* (1=best). Build in long form then convert.
  long <- df_season |>
    dplyr::transmute(
      ranking = 1L,
      item = item_id,
      rank = as.integer(placement_final)
    )
  PlackettLuce::rankings(long, id = "ranking", item = "item", rank = "rank", aggregate = FALSE)
}

season_ids <- sort(unique(df_cs$season))
rankings_by_season <- purrr::map(season_ids, function(s) make_season_rankings(df_cs[df_cs$season == s, ]))
names(rankings_by_season) <- as.character(season_ids)

saveRDS(rankings_by_season, file.path(ART, "rankings_by_season.rds"))

# Basic data summary
summary_tbl <- df_cs |>
  dplyr::summarise(
    n_rows = dplyr::n(),
    n_seasons = dplyr::n_distinct(season),
    n_items = dplyr::n_distinct(item_id),
    min_season = min(season),
    max_season = max(season),
    n_missing_placement_final = sum(is.na(placement_final))
  )
readr::write_csv(summary_tbl, file.path(ART, "data_summary.csv"))


