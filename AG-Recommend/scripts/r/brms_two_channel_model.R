################################################################################
# Two-Channel Hierarchical Bayesian Model for Dancing with the Stars
# Using brms for multivariate regression with correlated random effects
################################################################################

# Load required libraries
library(brms)
library(dplyr)
library(tidyr)

# Assume df is already loaded and preprocessed with the following columns:
# - season, week, couple_id, celebrity_id, pro_id
# - judge_z: standardized judges' scores (within season-week)
# - fan_y: centered log fan-vote share (within season-week)
# - week_z: standardized week number
# - improve_judge_z: change in judge_z from previous week
# - age_z: standardized celebrity age
# - gender: factor (celebrity gender)
# - industry: factor (celebrity industry)

################################################################################
# Step 1: Define formulas for both channels
################################################################################

# Formula for Channel 1: judge_z (judges' scores)
formula_judge <- bf(
  judge_z ~ 1 + week_z + improve_judge_z + age_z + gender + industry +
    (1 | pro_id) + (1 | celebrity_id) + (1 | season)
)

# Formula for Channel 2: fan_y (fan vote share)
formula_fan <- bf(
  fan_y ~ 1 + week_z + improve_judge_z + age_z + gender + industry +
    (1 | pro_id) + (1 | celebrity_id) + (1 | season)
)

################################################################################
# Step 2: Combine formulas into multivariate model
################################################################################

# Multivariate formula with correlated random effects for pro_id
# The set_rescor(FALSE) ensures no residual correlation between channels
mv_formula <- mvbf(
  formula_judge,
  formula_fan,
  rescor = FALSE  # No residual correlation between channels
)

################################################################################
# Step 3: Set weakly informative priors
################################################################################

# Priors for standardized outcomes (judge_z and fan_y)
# - Intercepts: Normal(0, 1) - centered at 0 for standardized outcomes
# - Fixed effects: Normal(0, 0.5) - weakly informative for standardized predictors
# - SD of random effects: Exponential(1) - weakly informative, favors smaller values
# - Correlation between pro random effects: LKJ(2) - weakly informative, slight preference for independence

priors <- c(
  # Intercepts for both channels
  prior(normal(0, 1), class = Intercept, resp = judgez),
  prior(normal(0, 1), class = Intercept, resp = fany),
  
  # Fixed effects for both channels
  prior(normal(0, 0.5), class = b, resp = judgez),
  prior(normal(0, 0.5), class = b, resp = fany),
  
  # Standard deviations of random effects
  prior(exponential(1), class = sd, resp = judgez),
  prior(exponential(1), class = sd, resp = fany),
  
  # Correlation between pro random effects across channels
  # LKJ(2) prior: weakly informative, slight preference for independence
  prior(lkj(2), class = cor)
)

################################################################################
# Step 4: Fit the multivariate Bayesian model
################################################################################

# Fit the model using brms
# - family: gaussian for both continuous outcomes
# - chains: 4 parallel MCMC chains
# - iter: 2000 iterations per chain (1000 warmup + 1000 sampling)
# - cores: use 4 cores for parallel computation
# - seed: for reproducibility

fit_mv <- brm(
  formula = mv_formula,
  data = df,
  family = gaussian(),
  prior = priors,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  cores = 4,
  seed = 123,
  control = list(adapt_delta = 0.95)  # Increase if divergent transitions occur
)

################################################################################
# Step 5: Print model summary
################################################################################

# Display comprehensive model summary
cat("\n=== MODEL SUMMARY ===\n")
summary(fit_mv)

################################################################################
# Step 6: Extract and display key results
################################################################################

# Extract fixed effects for both channels
cat("\n=== FIXED EFFECTS ===\n")
cat("\nChannel 1: Judges' Scores (judge_z)\n")
fixef(fit_mv, resp = "judgez")

cat("\nChannel 2: Fan Votes (fan_y)\n")
fixef(fit_mv, resp = "fany")

# Extract variance components (random effect standard deviations)
cat("\n=== VARIANCE COMPONENTS ===\n")
cat("\nRandom effect standard deviations:\n")
VarCorr(fit_mv)

# Extract correlation between pro random effects across channels
cat("\n=== CORRELATION BETWEEN PRO RANDOM EFFECTS ===\n")
cat("\nCorrelation between pro effects on judges vs. fans:\n")

# Get posterior samples for the correlation
posterior_samples <- as_draws_df(fit_mv)

# Extract correlation parameter for pro_id
# The correlation parameter is named: cor_pro_id__judgez_Intercept__fany_Intercept
cor_cols <- grep("cor_pro_id.*judgez.*fany", names(posterior_samples), value = TRUE)

if (length(cor_cols) > 0) {
  cor_pro <- posterior_samples[[cor_cols[1]]]
  
  # Display posterior summary for pro correlation
  cat(sprintf("  Mean: %.3f\n", mean(cor_pro)))
  cat(sprintf("  Median: %.3f\n", median(cor_pro)))
  cat(sprintf("  SD: %.3f\n", sd(cor_pro)))
  cat(sprintf("  95%% Credible Interval: [%.3f, %.3f]\n", 
              quantile(cor_pro, 0.025), quantile(cor_pro, 0.975)))
} else {
  cat("  Correlation parameter not found. Check model specification.\n")
}

################################################################################
# Step 7: Additional diagnostics and visualizations
################################################################################

# Check MCMC convergence diagnostics
cat("\n=== MCMC DIAGNOSTICS ===\n")
cat("\nRhat values (should be < 1.01):\n")
print(rhat(fit_mv))

cat("\nEffective sample sizes:\n")
print(neff_ratio(fit_mv))

# Plot posterior distributions for key parameters (optional)
# Uncomment to generate plots:
# plot(fit_mv, ask = FALSE)
# mcmc_plot(fit_mv, type = "trace")
# mcmc_plot(fit_mv, type = "dens")

################################################################################
# Step 8: Save model output (optional)
################################################################################

# Save the fitted model object for later use
# saveRDS(fit_mv, "brms_two_channel_model.rds")

# Save summary to text file
# sink("model_summary.txt")
# summary(fit_mv)
# sink()

cat("\n=== MODEL FITTING COMPLETE ===\n")
