################################################################################
# Data Preparation for Dancing with the Stars Analysis
# Prepares the 'df' object for the brms model
################################################################################

library(dplyr)
library(tidyr)
library(stringr)

# 1. Load raw data
# Update these paths if necessary
data_path <- "/Users/athenagao/Downloads/MCM-Data-With-The-Stars/Data/2026_MCM_Problem_C_Data.csv"
votes_path <- "/Users/athenagao/Downloads/MCM-Data-With-The-Stars/Data/estimate_votes.csv"

raw_data <- read.csv(data_path, stringsAsFactors = FALSE)
votes_data <- read.csv(votes_path, stringsAsFactors = FALSE)

# 2. Pivot scores from wide to long (one row per couple-week)
# We need to extract scores for each week
score_cols <- grep("week[0-9]+_judge[0-9]+_score", names(raw_data), value = TRUE)

df_long <- raw_data %>%
  pivot_longer(
    cols = all_of(score_cols),
    names_to = "week_judge",
    values_to = "score_str"
  ) %>%
  mutate(
    week = as.integer(str_extract(week_judge, "[0-9]+(?=_judge)")),
    judge = as.integer(str_extract(week_judge, "[0-9]+(?=$|_score)")),
    # Convert scores to numeric, treat "N/A" as NA
    score = as.numeric(ifelse(score_str == "N/A" | score_str == "", NA, score_str))
  ) %>%
  filter(!is.na(score) & score > 0) # Only keep actual performance records

# 3. Aggregate scores to one record per couple-week
df_week <- df_long %>%
  group_by(celebrity_name, ballroom_partner, season, week, celebrity_industry, celebrity_age_during_season) %>%
  summarise(
    avg_judge_score = mean(score, na.rm = TRUE),
    .groups = "drop"
  )

# 4. Standardize identifiers and outcomes
# Join with fan vote share estimates
df <- df_week %>%
  left_join(votes_data, by = c("celebrity_name", "season", "week")) %>%
  filter(!is.na(s_share)) %>%
  rename(
    pro_id = ballroom_partner,
    celebrity_id = celebrity_name,
    industry = celebrity_industry,
    age = celebrity_age_during_season,
    fan_vote_share = s_share
  )

# 5. Feature Engineering
df <- df %>%
  arrange(celebrity_id, season, week) %>%
  group_by(celebrity_id, season) %>%
  mutate(
    # improve_judge_z: change in judge score from previous week
    judge_z = (avg_judge_score - mean(avg_judge_score, na.rm = TRUE)) / sd(avg_judge_score, na.rm = TRUE),
    improve_judge_z = judge_z - lag(judge_z, default = 0),
    
    # week_z: standardized week number
    week_z = (week - mean(week)) / sd(week),
    
    # fan_y: centered log fan-vote share
    # We use a small offset if share is 0 to avoid log(0)
    fan_y = log(fan_vote_share + 0.001),
    fan_y = (fan_y - mean(fan_y, na.rm = TRUE)) / sd(fan_y, na.rm = TRUE)
  ) %>%
  ungroup()

# Standardize global predictors
df <- df %>%
  mutate(
    age_z = (age - mean(age, na.rm = TRUE)) / sd(age, na.rm = TRUE),
    # Assuming gender isn't in the raw CSV, we might need a placeholder or 
    # extract it if possible. For now, we'll dummy it if available or use industry as a main factor.
    gender = as.factor("Other") # Placeholder if not in data
  ) %>%
  mutate(
    industry = as.factor(industry),
    pro_id = as.factor(pro_id),
    celebrity_id = as.factor(celebrity_id),
    season = as.factor(season)
  )

cat("✓ Data prepared: ", nrow(df), " records ready for modeling.\n")
