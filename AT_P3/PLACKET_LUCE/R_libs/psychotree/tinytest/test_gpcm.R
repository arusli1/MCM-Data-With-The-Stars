### Check npltree with GPCM model on a simulated dataset ###

## simulate data and save it (don't run)##
# set.seed(42)
# n_samples <- 1000
# dataset1 <- data.frame(sex = rep(c("M", "F"), n_samples/2),
#                        fac1 = sample(c("A", "B"), n_samples, replace = T),
#                        fac2 = sample(c("C", "D", "E"), n_samples, replace = T))
#
# a_mat <- rep(0.5,8)
# d_mat <- matrix(rep(c(-1,0,1), 8), ncol = 3, byrow = T)
# Theta_vec <- matrix(rnorm(n_samples, sd=2), nrow=n_samples)
#
# DIF_mat_sex <- matrix(rep(0,8*3), ncol=3)
# DIF_mat_sex[1,] <- c(3,1.5,0)
#
# dat_sim1 <- lapply(1:n_samples, function(i){
#   d_mat_temp <- d_mat
#   if(dataset1[i,]$sex == "F"){
#     d_mat_temp <- d_mat_temp - DIF_mat_sex
#   }
#   simdata(a = a_mat, d = d_mat_temp, itemtype = "gpcm", Theta =  matrix(Theta_vec[i]))
# })
# dat_sim1 <- Reduce(function(...) rbind(...), dat_sim1)
# dataset1$response <- dat_sim1
# saveRDS(dataset1, "fit_GPCM_dataset.rds")

## load data and fit model ##
simdata <- readRDS("fit_GPCM_dataset.rds")
fit_GPCM <- gpcmtree(response ~. , data=simdata)

## test methods
expect_equal(coef(fit_GPCM), readRDS("test_GPCM.rds")[[1]],
             tolerance = 0.0001)
expect_equal(itempar(fit_GPCM), readRDS("test_GPCM.rds")[[2]],
             tolerance = 0.0001)
expect_equal(threshpar(fit_GPCM), readRDS("test_GPCM.rds")[[3]],
             tolerance = 0.0001)
expect_equal(guesspar(fit_GPCM), readRDS("test_GPCM.rds")[[4]],
             tolerance = 0.0001)
expect_equal(upperpar(fit_GPCM), readRDS("test_GPCM.rds")[[5]],
             tolerance = 0.0001)

### Check npltree with GPCM and an impact factor on the simulated dataset ###
## fit model ##
fit_GPCM_mg <- gpcmtree(response ~ fac1 | . , data=simdata)

## test methods
expect_equal(coef(fit_GPCM_mg), readRDS("test_GPCM_mg.rds")[[1]],
             tolerance = 0.0001)
expect_equal(itempar(fit_GPCM_mg), readRDS("test_GPCM_mg.rds")[[2]],
             tolerance = 0.0001)
expect_equal(threshpar(fit_GPCM_mg), readRDS("test_GPCM_mg.rds")[[3]],
             tolerance = 0.0001)
expect_equal(guesspar(fit_GPCM_mg), readRDS("test_GPCM_mg.rds")[[4]],
             tolerance = 0.0001)
expect_equal(upperpar(fit_GPCM_mg), readRDS("test_GPCM_mg.rds")[[5]],
             tolerance = 0.0001)
