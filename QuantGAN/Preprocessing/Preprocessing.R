##INSTALL the package "rstudioapi" to automate working directories.

##Figure out where this file is located, then set the working directory to this filepath
## This lets us open and close files using local paths
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Clear local memory
rm(list=ls())

options(digits=5)
library('LambertW')

###Read Data###
name = "GSPC"
fname = paste(name,".csv", sep = "")
dat.SP500 = read.csv(fname)

###1.Log Return###
n <- length(dat.SP500$Close)
lret.SP500 <- log(dat.SP500$Close[-1]/dat.SP500$Close[-n])

hist(lret.SP500)

###2.Nomalization with zero mean and unit variation###
lret.SP500.norm = ((lret.SP500 - mean(lret.SP500)) / sd(lret.SP500))
hist(lret.SP500.norm)

###3. Inverse Lambert W transform###
lret.SP500.InvLam = Gaussianize(lret.SP500.norm)
hist(lret.SP500.InvLam)

###4.Nomalization again###
lret.SP500.InvLam.Norm = ((lret.SP500.InvLam - mean(lret.SP500.InvLam)) / sd(lret.SP500.InvLam))
hist(lret.SP500.InvLam.Norm)

## this also accomplishes the same thing:
#lambertW2 = Gaussianize(normalize_1, return.u = TRUE)
#hist(lambertW2)

###5.Rolling Window left for python because right now we don't know what it should look like


###6.Export Data###
newname = paste("Normalized_", name, sep="")
fname = paste(newname,".csv", sep = "")

#colnames(lret.SP500.InvLam.Norm) <- "Norm Inv Lam Trans Log Ret"
#colnames(lret.SP500.InvLam.Norm) <- c("Date", "Index", "Data")

dfrm <- data.frame(Index = seq(from = 0, to = length(lret.SP500.InvLam.Norm)-1, by = 1), 
                   Date = dat.SP500$Date[2:length(dat.SP500$Date)],
                   Data = lret.SP500.InvLam.Norm)

colnames(dfrm) <- c("Index", "Date", "Data")




write.csv(dfrm, file = fname, row.names = FALSE)




##7.Compute Delta, if needed
fit.ml <- MLE_LambertW(lret.SP500, type = "h", distname = "normal", hessian=TRUE)
d = fit.ml$params.hat[["delta"]]


dfrm2 <- data.frame(delta = d, mean_lret = mean(lret.SP500), sd_lret = sd(lret.SP500), 
                    mean_InvLam = mean(lret.SP500.InvLam), sd_InvLam = sd(lret.SP500.InvLam))

colnames(dfrm2) <- c("delta", "mean_lret", "sd_lret", "mean_InvLam", "sd_InvLam")

##8. Write Parameters file

newname = paste("Parameters_", name, sep="")
gname = paste(newname,".csv", sep = "")
write.csv(dfrm2, file = gname, row.names = FALSE)


#9. Recreate the data (can be done in Python with parameters file)
#Recreated 2 is original log returns
#Recreated 1 is the zero mean, unit variance log returns (normalized log returns)

Recreated1 = (lret.SP500.InvLam.Norm*exp(d/2*lret.SP500.InvLam.Norm^2))*sd(lret.SP500.InvLam) + mean(lret.SP500.InvLam)
Recreated2 = Recreated1*sd(lret.SP500)+ mean(lret.SP500)


##9. Plots to verify that recreated data is what we want

hist(lret.SP500.norm)
hist(Recreated1)
hist(lret.SP500)
hist(Recreated2)
