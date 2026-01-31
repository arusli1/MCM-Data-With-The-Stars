################################################################################
# Visualization of Factor Influence Rankings
# Creates publication-quality plots for the two-channel model results
################################################################################

library(brms)
library(dplyr)
library(tidyr)
library(ggplot2)
library(patchwork)  # For combining plots

# Assume fit_mv is loaded and rankings are generated

################################################################################
# Function to create coefficient plot
################################################################################

create_coefficient_plot <- function(ranking_df, title, color = "#2E86AB") {
  # Prepare data
  plot_data <- ranking_df %>%
    arrange(Estimate) %>%
    mutate(Factor = factor(Factor, levels = Factor))
  
  # Create plot
  p <- ggplot(plot_data, aes(x = Estimate, y = Factor)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", size = 0.5) +
    geom_errorbarh(aes(xmin = Lower_CI, xmax = Upper_CI), 
                   height = 0.3, size = 0.8, color = color) +
    geom_point(size = 3, color = color) +
    labs(
      title = title,
      x = "Standardized Effect Size",
      y = NULL
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      panel.grid.major.y = element_line(color = "gray90"),
      panel.grid.minor = element_blank(),
      axis.text.y = element_text(size = 10)
    )
  
  return(p)
}

################################################################################
# Generate coefficient plots
################################################################################

# Load rankings (if not already in environment)
judge_ranking <- read.csv("judge_fixed_effects_ranking.csv")
fan_ranking <- read.csv("fan_fixed_effects_ranking.csv")

# Create plots
p_judge <- create_coefficient_plot(
  judge_ranking, 
  "Factors Influencing Judges' Scores",
  color = "#A23B72"
)

p_fan <- create_coefficient_plot(
  fan_ranking, 
  "Factors Influencing Fan Votes",
  color = "#2E86AB"
)

# Combine plots
combined_plot <- p_judge / p_fan

# Save combined plot
ggsave("factor_influence_comparison.png", combined_plot, 
       width = 10, height = 8, dpi = 300, bg = "white")

cat("Saved: factor_influence_comparison.png\n")

################################################################################
# Create comparison plot (judges vs fans)
################################################################################

comparison <- read.csv("judges_vs_fans_comparison.csv")

# Prepare data for comparison plot
comparison_long <- comparison %>%
  select(Factor, Judge_Effect, Fan_Effect) %>%
  pivot_longer(cols = c(Judge_Effect, Fan_Effect),
               names_to = "Channel",
               values_to = "Effect") %>%
  mutate(
    Channel = ifelse(Channel == "Judge_Effect", "Judges", "Fans"),
    Factor = factor(Factor)
  )

# Create comparison plot
p_comparison <- ggplot(comparison_long, aes(x = Effect, y = Factor, color = Channel)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", size = 0.5) +
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  scale_color_manual(values = c("Judges" = "#A23B72", "Fans" = "#2E86AB")) +
  labs(
    title = "Comparison: Judges vs. Fans",
    subtitle = "How do the same factors affect judges and fans differently?",
    x = "Standardized Effect Size",
    y = NULL,
    color = "Channel"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "gray30"),
    panel.grid.major.y = element_line(color = "gray90"),
    panel.grid.minor = element_blank(),
    legend.position = "top"
  )

ggsave("judges_vs_fans_comparison.png", p_comparison, 
       width = 10, height = 6, dpi = 300, bg = "white")

cat("Saved: judges_vs_fans_comparison.png\n")

################################################################################
# Create ranking bar chart
################################################################################

# Top 5 factors for each channel
top_judge <- judge_ranking %>% slice(1:5) %>% mutate(Channel = "Judges")
top_fan <- fan_ranking %>% slice(1:5) %>% mutate(Channel = "Fans")

top_factors <- bind_rows(top_judge, top_fan) %>%
  mutate(
    Factor = reorder_within(Factor, Abs_Effect, Channel),
    Channel = factor(Channel, levels = c("Judges", "Fans"))
  )

p_top <- ggplot(top_factors, aes(x = Abs_Effect, y = Factor, fill = Channel)) +
  geom_col() +
  scale_y_reordered() +
  scale_fill_manual(values = c("Judges" = "#A23B72", "Fans" = "#2E86AB")) +
  facet_wrap(~Channel, scales = "free_y", ncol = 1) +
  labs(
    title = "Top 5 Most Influential Factors",
    x = "Absolute Effect Size",
    y = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    strip.text = element_text(face = "bold", size = 12),
    legend.position = "none",
    panel.grid.major.y = element_blank()
  )

ggsave("top_factors_ranking.png", p_top, 
       width = 10, height = 8, dpi = 300, bg = "white")

cat("Saved: top_factors_ranking.png\n")

################################################################################
# Helper function for reordering within facets
################################################################################

reorder_within <- function(x, by, within, fun = mean, sep = "___", ...) {
  new_x <- paste(x, within, sep = sep)
  stats::reorder(new_x, by, FUN = fun)
}

scale_y_reordered <- function(..., sep = "___") {
  reg <- paste0(sep, ".+$")
  ggplot2::scale_y_discrete(labels = function(x) gsub(reg, "", x), ...)
}

cat("\nAll visualizations created successfully!\n")
