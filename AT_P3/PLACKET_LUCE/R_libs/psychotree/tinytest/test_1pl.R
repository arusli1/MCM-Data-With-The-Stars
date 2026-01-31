### Check npltree with 1PL model on the SPISA dataset ###

## load data and fit model ##
data("SPISA", package = "psychotree")
fit_1PL <- npltree(spisa[,1:9] ~ age, data = SPISA, type = "1PL", maxit = 5000)

## test methods
expect_equal(coef(fit_1PL), readRDS("test_1PL.rds")[[1]],
             tolerance = 0.0001)
expect_equal(itempar(fit_1PL), readRDS("test_1PL.rds")[[2]],
             tolerance = 0.0001)
expect_equal(threshpar(fit_1PL), readRDS("test_1PL.rds")[[3]],
             tolerance = 0.0001)
expect_equal(guesspar(fit_1PL), readRDS("test_1PL.rds")[[4]],
             tolerance = 0.0001)
expect_equal(upperpar(fit_1PL), readRDS("test_1PL.rds")[[5]],
             tolerance = 0.0001)

### Check npltree with 1PL model and an impact factor on the SPISA dataset ###

## load data and fit model ##
data("SPISA", package = "psychotree")
fit_1PL_mg <- npltree(spisa[,1:9] ~ gender | gender, data = SPISA, type = "1PL", maxit = 2000)

## test methods
expect_equal(coef(fit_1PL_mg), readRDS("test_1PL_mg.rds")[[1]],
             tolerance = 0.0001)
expect_equal(itempar(fit_1PL_mg), readRDS("test_1PL_mg.rds")[[2]],
             tolerance = 0.0001)
expect_equal(threshpar(fit_1PL_mg), readRDS("test_1PL_mg.rds")[[3]],
             tolerance = 0.0001)
expect_equal(guesspar(fit_1PL_mg), readRDS("test_1PL_mg.rds")[[4]],
             tolerance = 0.0001)
expect_equal(upperpar(fit_1PL_mg), readRDS("test_1PL_mg.rds")[[5]],
             tolerance = 0.0001)

