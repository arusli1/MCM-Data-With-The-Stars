#!/usr/bin/env Rscript

# Entry point: run the full Plackett–Luce pipeline end-to-end.

args <- commandArgs(trailingOnly = TRUE)
do_render <- TRUE
if (length(args) > 0 && any(args %in% c("--no-render"))) do_render <- FALSE

root <- normalizePath("AT_P3/PLACKET_LUCE", mustWork = TRUE)
source(file.path(root, "R", "00_setup.R"))

message("Running: 01_prepare_rankings.R")
source(file.path(root, "R", "01_prepare_rankings.R"))

message("Running: 02_fit_pl_model.R")
source(file.path(root, "R", "02_fit_pl_model.R"))

message("Running: 03_predict_evaluate.R")
source(file.path(root, "R", "03_predict_evaluate.R"))

message("Running: 04_season_5fold_cv.R")
source(file.path(root, "R", "04_season_5fold_cv.R"))

message("Running: 04_ablation_tests.R")
source(file.path(root, "R", "04_ablation_tests.R"))

if (do_render) {
  message("Rendering report/report.Rmd → report/report.md")
  ensure_pkg("knitr")
  ensure_pkg("rmarkdown")
  in_rmd <- file.path(root, "report", "report.Rmd")
  out_md <- file.path(root, "report", "report.md")
  if (isTRUE(rmarkdown::pandoc_available())) {
    rmarkdown::render(
      input = in_rmd,
      output_format = "github_document",
      output_file = "report.md",
      output_dir = file.path(root, "report"),
      quiet = TRUE
    )
  } else {
    message("Pandoc not available; using knitr::knit() to generate markdown.")
    knitr::knit(input = in_rmd, output = out_md, quiet = TRUE, encoding = "UTF-8")
  }
}

message("=== Plackett–Luce pipeline complete ===")
message(sprintf("Artifacts: %s", normalizePath(file.path(root, "artifacts"), mustWork = TRUE)))
message(sprintf("Report: %s", normalizePath(file.path(root, "report", "report.md"), mustWork = TRUE)))


