### R code from vignette source 'raschtree.Rnw'

###################################################
### code chunk number 1: preliminaries
###################################################
options(prompt = "R> ", continue = "+  ")


###################################################
### code chunk number 2: prep_install (eval = FALSE)
###################################################
## install.packages("psychotree")


###################################################
### code chunk number 3: prep_library
###################################################
library("psychotree")


###################################################
### code chunk number 4: prep_data
###################################################
data("SPISA", package = "psychotree")


###################################################
### code chunk number 5: recode (eval = FALSE)
###################################################
## mydata$resp <- as.matrix(mydata[ , 1:5])


###################################################
### code chunk number 6: recode2 (eval = FALSE)
###################################################
## mydata <- mydata[ , -(1:5)]


###################################################
### code chunk number 7: fit_raschtree (eval = FALSE)
###################################################
## my_first_raschtree <- raschtree(spisa ~ age + gender +
##   semester + elite + spon, data = SPISA)


###################################################
### code chunk number 8: fit_raschtree_minsize (eval = FALSE)
###################################################
## my_first_raschtree <- raschtree(spisa ~ age + gender +
##   semester + elite + spon, data = SPISA, minsize = 30)


###################################################
### code chunk number 9: fit_raschtree_cache
###################################################
if(file.exists("raschtree-spisa.rda")) load("raschtree-spisa.rda") else {
my_first_raschtree <- raschtree(spisa ~ age + gender +
  semester + elite + spon, data = SPISA, minsize = 30)
save(my_first_raschtree, file = "raschtree-spisa.rda")
}
file.remove("raschtree-spisa.rda")


###################################################
### code chunk number 10: plot_raschtree
###################################################
plot(my_first_raschtree)


###################################################
### code chunk number 11: plot_raschtree_col
###################################################
plot(my_first_raschtree, 
      col = rep(palette.colors(5), each = 9))


###################################################
### code chunk number 12: coef_raschtree
###################################################
coef(my_first_raschtree, node = 4)


###################################################
### code chunk number 13: itempar_raschtree
###################################################
itempar(my_first_raschtree, node = 4)


###################################################
### code chunk number 14: raschtree.Rnw:316-317 (eval = FALSE)
###################################################
## install.packages("stablelearner")


###################################################
### code chunk number 15: raschtree.Rnw:322-323 (eval = FALSE)
###################################################
## library("stablelearner")


###################################################
### code chunk number 16: raschtree.Rnw:332-334 (eval = FALSE)
###################################################
## set.seed(4321)
## my_first_raschtree_st <- stabletree(my_first_raschtree, B = 50)


###################################################
### code chunk number 17: stabletree_fit
###################################################
if(require("stablelearner", quietly = TRUE)) {

if(!file.exists("my_first_raschtree_st.Rdata")){
set.seed(4321)
my_first_raschtree_st <- stabletree(my_first_raschtree, B = 50)
save(my_first_raschtree_st, file = "my_first_raschtree_st.Rdata")
} else{
load("my_first_raschtree_st.Rdata")
}

spon1 <- summary(my_first_raschtree_st)$vstab["spon", 1]
spon3 <- summary(my_first_raschtree_st)$vstab["spon", 3]

} else {

my_first_raschtree_st <- matrix(1)
spon1 <- spon3 <- 0

}


###################################################
### code chunk number 18: stabletree_summary
###################################################
summary(my_first_raschtree_st)


###################################################
### code chunk number 19: stabletree_barplot
###################################################
barplot(my_first_raschtree_st)


###################################################
### code chunk number 20: stabletree_image
###################################################
image(my_first_raschtree_st)


###################################################
### code chunk number 21: stabletree_plot
###################################################
plot(my_first_raschtree_st)


