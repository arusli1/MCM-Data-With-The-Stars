################################################################################
# Package Installation Script for DWTS Analysis
# Checks for and installs required packages
################################################################################

required_packages <- c("brms", "dplyr", "tidyr", "ggplot2", "patchwork", "stringr")

install_if_missing <- function(p) {
  if (!require(p, character.only = TRUE)) {
    cat(sprintf("Installing package: %s\n", p))
    install.packages(p, dependencies = TRUE, repos = "https://cloud.r-project.org")
  } else {
    cat(sprintf("Package already installed: %s\n", p))
  }
}

cat("Checking dependencies...\n")
invisible(lapply(required_packages, install_if_missing))

cat("\nAll dependencies are ready!\n")
cat("NOTE: brms requires a C++ compiler (Rtools on Windows, Xcode on Mac).\n")
cat("If you encounter compilation errors, please ensure your system compiler is set up.\n")
