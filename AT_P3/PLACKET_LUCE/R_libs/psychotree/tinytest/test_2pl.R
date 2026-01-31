### Check npltree with 2PL model on the SPISA dataset ###

## load data and fit model ##
data("SPISA", package = "psychotree")
fit_2PL <- npltree(spisa[,1:9] ~ age + gender, data = SPISA, type = "2PL", maxit = 5000)

## test methods
expect_equal(coef(fit_2PL), readRDS("test_2PL.rds")[[1]],
             tolerance = 0.0001)
expect_equal(itempar(fit_2PL), readRDS("test_2PL.rds")[[2]],
             tolerance = 0.0001)
expect_equal(threshpar(fit_2PL), readRDS("test_2PL.rds")[[3]],
             tolerance = 0.0001)
expect_equal(guesspar(fit_2PL), readRDS("test_2PL.rds")[[4]],
             tolerance = 0.0001)
expect_equal(upperpar(fit_2PL), readRDS("test_2PL.rds")[[5]],
             tolerance = 0.0001)


### Check npltree with 2PL model and an impact factor on the SPISA dataset ###

## load data and fit model ##
data("SPISA", package = "psychotree")
fit_2PL_mg <- npltree(spisa[,1:9] ~ gender | age + gender, data = SPISA, type = "2PL", maxit = 5000)

## test methods
expect_equal(coef(fit_2PL_mg), readRDS("test_2PL_mg.rds")[[1]],
             tolerance = 0.0001)
expect_equal(itempar(fit_2PL_mg), readRDS("test_2PL_mg.rds")[[2]],
             tolerance = 0.0001)
expect_equal(threshpar(fit_2PL_mg), readRDS("test_2PL_mg.rds")[[3]],
             tolerance = 0.0001)
expect_equal(guesspar(fit_2PL_mg), readRDS("test_2PL_mg.rds")[[4]],
             tolerance = 0.0001)
expect_equal(upperpar(fit_2PL_mg), readRDS("test_2PL_mg.rds")[[5]],
             tolerance = 0.0001)
