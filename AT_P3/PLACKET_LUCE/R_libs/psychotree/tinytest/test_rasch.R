### Check npltree with rasch model on the SPISA dataset ###

## load data and fit model ##
data("SPISA", package = "psychotree")
fit_rasch <- npltree(spisa ~ age + gender + semester + elite + spon, data = SPISA, type = "Rasch")

## check that a warning is emitted if impact factors are used with CML estimation
expect_warning(npltree(spisa ~ age + gender + semester + elite + spon | gender, data = SPISA, type = "Rasch"))

## test methods
expect_equal(coef(fit_rasch), readRDS("test_rasch.rds")[[1]],
             tolerance = 0.0001)
expect_equal(itempar(fit_rasch), readRDS("test_rasch.rds")[[2]],
             tolerance = 0.0001)
expect_equal(threshpar(fit_rasch), readRDS("test_rasch.rds")[[3]],
             tolerance = 0.0001)
expect_equal(guesspar(fit_rasch), readRDS("test_rasch.rds")[[4]],
             tolerance = 0.0001)
expect_equal(upperpar(fit_rasch), readRDS("test_rasch.rds")[[5]],
             tolerance = 0.0001)

