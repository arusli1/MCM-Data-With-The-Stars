################################################################################
# Factor Influence Rankings from Two-Channel Hierarchical Model
# Ranks factors by their standardized effect sizes
################################################################################

# Load required libraries
library(brms)
library(dplyr)
library(tidyr)
library(ggplot2)

# Assume fit_mv is the fitted brms model from brms_two_channel_model.R
# If not in environment, load it:
# fit_mv <- readRDS("brms_two_channel_model.rds")

################################################################################
# Function to extract and rank fixed effects by absolute effect size
################################################################################

rank_fixed_effects <- function(model, response_var) {
  # Extract fixed effects for the specified response
  fixed_eff <- fixef(model, resp = response_var)
  
  # Remove intercept
  fixed_eff <- fixed_eff[rownames(fixed_eff) != "Intercept", , drop = FALSE]
  
  # Create ranking data frame
  ranking_df <- data.frame(
    Factor = rownames(fixed_eff),
    Estimate = fixed_eff[, "Estimate"],
    Lower_CI = fixed_eff[, "Q2.5"],
    Upper_CI = fixed_eff[, "Q97.5"],
    Abs_Effect = abs(fixed_eff[, "Estimate"]),
    stringsAsFactors = FALSE
  )
  
  # Rank by absolute effect size
  ranking_df <- ranking_df %>%
    arrange(desc(Abs_Effect)) %>%
    mutate(Rank = row_number())
  
  return(ranking_df)
}

################################################################################
# Function to extract variance components
################################################################################

extract_variance_components <- function(model, response_var) {
  # Get variance-covariance matrix
  vc <- VarCorr(model)
  
  # Extract standard deviations for the specified response
  var_comp <- data.frame(
    Random_Effect = character(),
    SD = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Loop through random effects
  for (group_name in names(vc)) {
    if (group_name %in% c("pro_id", "celebrity_id", "season")) {
      group_vc <- vc[[group_name]]
      
      # Check if this group has the response variable
      if (response_var %in% names(group_vc)) {
        sd_val <- group_vc[[response_var]]$sd[1, "Estimate"]
        var_comp <- rbind(var_comp, data.frame(
          Random_Effect = group_name,
          SD = sd_val
        ))
      }
    }
  }
  
  # Rank by standard deviation
  var_comp <- var_comp %>%
    arrange(desc(SD)) %>%
    mutate(Rank = row_number())
  
  return(var_comp)
}

################################################################################
# RANKING 1: Overall influence (judges' scores)
################################################################################

cat("\n" , rep("=", 80), "\n", sep = "")
cat("RANKING 1: FACTORS INFLUENCING JUDGES' SCORES (judge_z)\n")
cat(rep("=", 80), "\n\n", sep = "")

# Fixed effects ranking
judge_fixed_ranking <- rank_fixed_effects(fit_mv, "judgez")

cat("FIXED EFFECTS (Most to Least Influential):\n")
cat(rep("-", 80), "\n", sep = "")
for (i in 1:nrow(judge_fixed_ranking)) {
  row <- judge_fixed_ranking[i, ]
  cat(sprintf("%d. %s\n", row$Rank, row$Factor))
  cat(sprintf("   Effect: %.4f [95%% CI: %.4f, %.4f]\n", 
              row$Estimate, row$Lower_CI, row$Upper_CI))
  cat(sprintf("   Absolute Effect: %.4f\n\n", row$Abs_Effect))
}

# Variance components
judge_var_ranking <- extract_variance_components(fit_mv, "judgez")

cat("\nRANDOM EFFECTS VARIANCE (Most to Least Variable):\n")
cat(rep("-", 80), "\n", sep = "")
for (i in 1:nrow(judge_var_ranking)) {
  row <- judge_var_ranking[i, ]
  cat(sprintf("%d. %s: SD = %.4f\n", row$Rank, row$Random_Effect, row$SD))
}

################################################################################
# RANKING 2: Overall influence (fan votes)
################################################################################

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("RANKING 2: FACTORS INFLUENCING FAN VOTES (fan_y)\n")
cat(rep("=", 80), "\n\n", sep = "")

# Fixed effects ranking
fan_fixed_ranking <- rank_fixed_effects(fit_mv, "fany")

cat("FIXED EFFECTS (Most to Least Influential):\n")
cat(rep("-", 80), "\n", sep = "")
for (i in 1:nrow(fan_fixed_ranking)) {
  row <- fan_fixed_ranking[i, ]
  cat(sprintf("%d. %s\n", row$Rank, row$Factor))
  cat(sprintf("   Effect: %.4f [95%% CI: %.4f, %.4f]\n", 
              row$Estimate, row$Lower_CI, row$Upper_CI))
  cat(sprintf("   Absolute Effect: %.4f\n\n", row$Abs_Effect))
}

# Variance components
fan_var_ranking <- extract_variance_components(fit_mv, "fany")

cat("\nRANDOM EFFECTS VARIANCE (Most to Least Variable):\n")
cat(rep("-", 80), "\n", sep = "")
for (i in 1:nrow(fan_var_ranking)) {
  row <- fan_var_ranking[i, ]
  cat(sprintf("%d. %s: SD = %.4f\n", row$Rank, row$Random_Effect, row$SD))
}

################################################################################
# RANKING 3: Comparison of effects between judges and fans
################################################################################

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("RANKING 3: COMPARATIVE INFLUENCE (Judges vs. Fans)\n")
cat(rep("=", 80), "\n\n", sep = "")

# Merge rankings
comparison <- judge_fixed_ranking %>%
  select(Factor, Judge_Effect = Estimate, Judge_Rank = Rank) %>%
  left_join(
    fan_fixed_ranking %>% select(Factor, Fan_Effect = Estimate, Fan_Rank = Rank),
    by = "Factor"
  ) %>%
  mutate(
    Rank_Difference = abs(Judge_Rank - Fan_Rank),
    Effect_Difference = abs(Judge_Effect - Fan_Effect)
  )

cat("FACTORS RANKED BY CONSISTENCY ACROSS JUDGES AND FANS:\n")
cat("(Lower rank difference = more consistent influence)\n")
cat(rep("-", 80), "\n", sep = "")

comparison_sorted <- comparison %>% arrange(Rank_Difference)

for (i in 1:nrow(comparison_sorted)) {
  row <- comparison_sorted[i, ]
  cat(sprintf("\n%s:\n", row$Factor))
  cat(sprintf("  Judge Effect: %.4f (Rank: %d)\n", row$Judge_Effect, row$Judge_Rank))
  cat(sprintf("  Fan Effect:   %.4f (Rank: %d)\n", row$Fan_Effect, row$Fan_Rank))
  cat(sprintf("  Rank Difference: %d\n", row$Rank_Difference))
}

################################################################################
# RANKING 4: By variable type
################################################################################

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("RANKING 4: INFLUENCE BY VARIABLE TYPE\n")
cat(rep("=", 80), "\n\n", sep = "")

# Categorize variables
categorize_variable <- function(var_name) {
  if (grepl("week", var_name, ignore.case = TRUE)) return("Temporal")
  if (grepl("improve", var_name, ignore.case = TRUE)) return("Performance")
  if (grepl("age", var_name, ignore.case = TRUE)) return("Demographics")
  if (grepl("gender", var_name, ignore.case = TRUE)) return("Demographics")
  if (grepl("industry", var_name, ignore.case = TRUE)) return("Background")
  return("Other")
}

# Add categories to rankings
judge_fixed_ranking$Category <- sapply(judge_fixed_ranking$Factor, categorize_variable)
fan_fixed_ranking$Category <- sapply(fan_fixed_ranking$Factor, categorize_variable)

# Aggregate by category for judges
cat("JUDGES' SCORES - Average Influence by Variable Type:\n")
cat(rep("-", 80), "\n", sep = "")

judge_by_category <- judge_fixed_ranking %>%
  group_by(Category) %>%
  summarise(
    Avg_Abs_Effect = mean(Abs_Effect),
    Count = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(Avg_Abs_Effect)) %>%
  mutate(Rank = row_number())

for (i in 1:nrow(judge_by_category)) {
  row <- judge_by_category[i, ]
  cat(sprintf("%d. %s: Avg Effect = %.4f (n=%d variables)\n", 
              row$Rank, row$Category, row$Avg_Abs_Effect, row$Count))
}

# Aggregate by category for fans
cat("\n\nFAN VOTES - Average Influence by Variable Type:\n")
cat(rep("-", 80), "\n", sep = "")

fan_by_category <- fan_fixed_ranking %>%
  group_by(Category) %>%
  summarise(
    Avg_Abs_Effect = mean(Abs_Effect),
    Count = n(),
    .groups = "drop"
  ) %>%
  arrange(desc(Avg_Abs_Effect)) %>%
  mutate(Rank = row_number())

for (i in 1:nrow(fan_by_category)) {
  row <- fan_by_category[i, ]
  cat(sprintf("%d. %s: Avg Effect = %.4f (n=%d variables)\n", 
              row$Rank, row$Category, row$Avg_Abs_Effect, row$Count))
}

################################################################################
# RANKING 5: Industry-specific effects
################################################################################

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("RANKING 5: INDUSTRY-SPECIFIC EFFECTS\n")
cat(rep("=", 80), "\n\n", sep = "")

# Extract industry effects for judges
industry_judge <- judge_fixed_ranking %>%
  filter(grepl("industry", Factor, ignore.case = TRUE)) %>%
  arrange(desc(Abs_Effect))

cat("JUDGES' SCORES - Industry Effects:\n")
cat(rep("-", 80), "\n", sep = "")
if (nrow(industry_judge) > 0) {
  for (i in 1:nrow(industry_judge)) {
    row <- industry_judge[i, ]
    cat(sprintf("%d. %s: %.4f [%.4f, %.4f]\n", 
                i, row$Factor, row$Estimate, row$Lower_CI, row$Upper_CI))
  }
} else {
  cat("No industry effects found (may be reference category)\n")
}

# Extract industry effects for fans
industry_fan <- fan_fixed_ranking %>%
  filter(grepl("industry", Factor, ignore.case = TRUE)) %>%
  arrange(desc(Abs_Effect))

cat("\n\nFAN VOTES - Industry Effects:\n")
cat(rep("-", 80), "\n", sep = "")
if (nrow(industry_fan) > 0) {
  for (i in 1:nrow(industry_fan)) {
    row <- industry_fan[i, ]
    cat(sprintf("%d. %s: %.4f [%.4f, %.4f]\n", 
                i, row$Factor, row$Estimate, row$Lower_CI, row$Upper_CI))
  }
} else {
  cat("No industry effects found (may be reference category)\n")
}

################################################################################
# Save rankings to CSV files
################################################################################

write.csv(judge_fixed_ranking, "judge_fixed_effects_ranking.csv", row.names = FALSE)
write.csv(fan_fixed_ranking, "fan_fixed_effects_ranking.csv", row.names = FALSE)
write.csv(comparison, "judges_vs_fans_comparison.csv", row.names = FALSE)

cat("\n\n", rep("=", 80), "\n", sep = "")
cat("Rankings saved to CSV files:\n")
cat("  - judge_fixed_effects_ranking.csv\n")
cat("  - fan_fixed_effects_ranking.csv\n")
cat("  - judges_vs_fans_comparison.csv\n")
cat(rep("=", 80), "\n", sep = "")
