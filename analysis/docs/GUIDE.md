# Quick Start Guide: Running the Bayesian Influence Analysis

This guide explains how to run the statistical modeling pipeline to analyze the impact of pro dancers and celebrity characteristics on judges' scores and fan votes.

## Prerequisites

1.  **R installed** (version 4.0 or higher recommended).
2.  **C++ Compiler**: `brms` (which uses Stan) requires a compiler.
    - **Mac**: Install Xcode Command Line Tools (`xcode-select --install` in terminal).
    - **Windows**: Install [Rtools](https://cran.r-project.org/bin/windows/Rtools/).

## Setup Instructions

1.  **Install R Packages**:
    Open a terminal in the `analysis/` directory and run:
    ```bash
    Rscript install_dependencies.R
    ```

2.  **Verify Data Paths**:
    Open `load_data.R` and ensure the paths to `2026_MCM_Problem_C_Data.csv` and `estimate_votes.csv` are correct on your system.

## Running the Analysis

The entire pipeline—from data cleaning to model fitting and ranking generation—is automated.

**To run via Terminal:**
```bash
Rscript run_complete_analysis.R
```

**To run inside R (or RStudio):**
```r
setwd("/path/to/MCM-Data-With-The-Stars/analysis")
source("run_complete_analysis.R")
```

## What Happens Next?

1.  **Data Loading**: The script will pivot your raw data and calculate standardized features.
2.  **Model Fitting**: A Bayesian hierarchical model will be fitted using 4 MCMC chains. **Note: This may take 5-15 minutes depending on your CPU.**
3.  **Ranking Generation**: Statistical effects will be extracted and ranked.
4.  **Visualizations**: Three PNG charts will be created in the current directory.

## Troubleshooting

- **Sampling Errors**: If you get "divergent transitions," the model may need more warmup. You can increase `adapt_delta` in `brms_two_channel_model.R`.
- **Compiler Issues**: Ensure your R environment can find your C++ compiler. Run `library(brms); example(brm)` to test.
