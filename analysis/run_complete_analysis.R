################################################################################
# MASTER SCRIPT: Complete Factor Influence Analysis
# Run this script to execute the full analysis pipeline
################################################################################

cat("\n")
cat(rep("=", 80), "\n", sep = "")
cat("DANCING WITH THE STARS - FACTOR INFLUENCE ANALYSIS\n")
cat("Two-Channel Hierarchical Bayesian Model\n")
cat(rep("=", 80), "\n\n", sep = "")

# Set working directory to analysis folder
setwd("/Users/athenagao/Downloads/MCM-Data-With-The-Stars/analysis")

# STEP 1: Load and prepare data
################################################################################

cat("STEP 1: Loading and preparing data...\n")
source("load_data.R")

################################################################################
# STEP 2: Fit the two-channel hierarchical model
################################################################################

cat("STEP 2: Fitting two-channel hierarchical Bayesian model...\n")
cat("(This may take several minutes)\n\n")

source("brms_two_channel_model.R")

cat("\n✓ Model fitting complete!\n\n")

################################################################################
# STEP 2: Generate factor rankings
################################################################################

cat("STEP 2: Generating factor influence rankings...\n\n")

source("brms_factor_rankings.R")

cat("\n✓ Rankings generated!\n\n")

################################################################################
# STEP 3: Create visualizations
################################################################################

cat("STEP 3: Creating visualizations...\n\n")

source("brms_visualizations.R")

cat("\n✓ Visualizations created!\n\n")

################################################################################
# STEP 4: Summary of key findings
################################################################################

cat(rep("=", 80), "\n", sep = "")
cat("ANALYSIS COMPLETE - KEY OUTPUTS\n")
cat(rep("=", 80), "\n\n", sep = "")

cat("CSV Files Generated:\n")
cat("  1. judge_fixed_effects_ranking.csv - Factors ranked by influence on judges\n")
cat("  2. fan_fixed_effects_ranking.csv - Factors ranked by influence on fans\n")
cat("  3. judges_vs_fans_comparison.csv - Side-by-side comparison\n\n")

cat("Visualizations Generated:\n")
cat("  1. factor_influence_comparison.png - Coefficient plots for both channels\n")
cat("  2. judges_vs_fans_comparison.png - Direct comparison of effects\n")
cat("  3. top_factors_ranking.png - Top 5 most influential factors\n\n")

cat("Model Object:\n")
cat("  - fit_mv (brms model object, can be saved with saveRDS())\n\n")

cat("Documentation:\n")
cat("  - See RANKINGS_README.md for detailed interpretation guide\n\n")

################################################################################
# STEP 5: Quick summary of top factors
################################################################################

cat(rep("=", 80), "\n", sep = "")
cat("QUICK SUMMARY: TOP 3 FACTORS\n")
cat(rep("=", 80), "\n\n", sep = "")

# Load rankings
judge_rank <- read.csv("judge_fixed_effects_ranking.csv")
fan_rank <- read.csv("fan_fixed_effects_ranking.csv")

cat("TOP 3 FACTORS FOR JUDGES' SCORES:\n")
for (i in 1:min(3, nrow(judge_rank))) {
  cat(sprintf("  %d. %s (Effect: %.3f)\n", 
              i, judge_rank$Factor[i], judge_rank$Estimate[i]))
}

cat("\nTOP 3 FACTORS FOR FAN VOTES:\n")
for (i in 1:min(3, nrow(fan_rank))) {
  cat(sprintf("  %d. %s (Effect: %.3f)\n", 
              i, fan_rank$Factor[i], fan_rank$Estimate[i]))
}

cat("\n")
cat(rep("=", 80), "\n", sep = "")
cat("Analysis complete! Review the generated files for detailed results.\n")
cat(rep("=", 80), "\n", sep = "")
