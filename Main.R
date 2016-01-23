#Copyright (c) 2016, Krishna Kesavan All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following 
#conditions are met: * Redistributions of source code must retain the above copyright notice, this list of conditions and the 
#following disclaimer. * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and 
#the following disclaimer in the documentation and/or other materials provided with the distribution. * Neither the name of the 
#Krishna Kesavan nor the names of its contributors may be used to endorse or promote products derived from this software without 
#specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT 
#NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
#Krishna Kesavan BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
#AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY 
#WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
# Kaggle "Homesite Quote Conversion"


library(data.table)
library(xgboost)
library(lubridate)
library(caret)
library(genalg)
library(caTools)
library(infotheo)
library(qdapTools)

set.seed(987692)

# Number of Folds for cross validation.

kfolds=3

sink("Ens6.txt", append=T)


#utility functions

factorToNumeric <- function(train, test, response, variables, metrics){
  temp <- data.frame(c(rep(0,nrow(test))), row.names = NULL)

  for (variable in variables){
    for (metric in metrics) {
      x <- tapply(train[, response], train[,variable], metric)
      x <- data.frame(row.names(x),x, row.names = NULL)
      temp <- data.frame(temp,round(lookup(test[,variable], x),2))
      colnames(temp)[ncol(temp)] <- paste(metric,variable, sep = "_")
    }
  }
  return (temp[,-1])
}

# build Gini functions for use in custom xgboost evaluation metric

SumModelGini <- function(solution, submission) {
        df = data.frame(solution = solution, submission = submission)
        df <- df[order(df$submission, decreasing = TRUE),]
        df$random = (1:nrow(df))/nrow(df)
        totalPos <- sum(df$solution)
        df$cumPosFound <- cumsum(df$solution) # this will store the cumulative number of positive examples found (used for computing "Model Lorentz")
        df$Lorentz <- df$cumPosFound / totalPos # this will store the cumulative proportion of positive examples found ("Model Lorentz")
        df$Gini <- df$Lorentz - df$random # will store Lorentz minus random
        return(sum(df$Gini))
}

NormalizedGini <- function(solution, submission) {
        SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

# wrap up into a function to be called within xgboost.train
evalgini <- function(preds, dtrain) {
        labels <- getinfo(dtrain, "label")
        err <- NormalizedGini(as.numeric(labels),as.numeric(preds))
        return(list(metric = "Gini", value = err))
}


cat("reading the train and test data\n")
train <- fread("../input/train.csv")
test  <- fread("../input/test.csv")


# Top 100 Features extracted from xgboost VarInf
# We need to run interactions on them.

topF <- read.csv("topFeat.csv", head=T)
train <- as.data.frame(train)
test  <- as.data.frame(test)

response <- train$QuoteConversion_Flag
train$QuoteConversion_Flag <- NULL


# Find all "character" columns and compute Response metrics for them

charCols <- names(train[, sapply(train, class) == 'character'])

charCols <- charCols[charCols %in% topF$x]

train$QuoteConversion_Flag <- response
ftrain <- factorToNumeric(train, train, "QuoteConversion_Flag", charCols, c("mean","median","sd","IQR"))
ftest  <- factorToNumeric(train, test, "QuoteConversion_Flag", charCols, c("mean","median","sd","IQR"))
train$QuoteConversion_Flag <- NULL
train <- cbind(train,ftrain)
test  <- cbind(test, ftest)


#Extra features from just "date column". output of script from
#https://www.kaggle.com/jesseburstrom/homesite-quote-conversion/extra-features-to-add/code

train1 <- read.csv("train1.csv", head=T)
test1 <- read.csv("test1.csv", head=T)
train <- cbind(train, train1)
test  <- cbind(test, test1)

testquote <- test$QuoteNumber
data <- rbind(train, test)

# find the count of NA columns in a row, before imputing them.

data$rowSumsNA <- rowSums(is.na(data))

# There are some NAs in the integer columns so conversion to zero
data[is.na(data)]   <- -1

# Compute features from date column

data$Original_Quote_Date <- as.Date(data$Original_Quote_Date)

data$month <- month(data$Original_Quote_Date)
data$yday  <- yday(data$Original_Quote_Date)
data$weekd <- week(data$Original_Quote_Date)
data$wday  <- wday(data$Original_Quote_Date)
data$year  <- year(data$Original_Quote_Date)


data$Original_Quote_Date <- NULL

# original tsne.
load("tsne5.RData")
tsn <- as.data.frame(tsne5$Y)
names(tsn) <- c("tsne21","tsne22")
data = cbind(data,tsn)

# tsne on just the date features.
load("tsneTime.RData")
tsn <- as.data.frame(tsneTime$Y)
names(tsn) <- c("tsneTime1","tsneTime2")
data = cbind(data,tsn)


# original tsne with 5 dims.
load("tsne.RData")
tsn <- as.data.frame(tsne$Y)
names(tsn) <- c("tsne51","tsne52","tsne53","tsne54","tsne55")
data = cbind(data,tsn)

# tsne on just the binary features.
load("tsneFac.RData")
tsn <- as.data.frame(tsne$Y)
names(tsn) <- c("tsneF1","tsneF2")
data = cbind(data,tsn)

# tsne on just the "diff" features which are listed below.
load("tsnediff.RData")
tsn <- as.data.frame(tsnediff$Y)
names(tsn) <- c("tsnedff1","tsnedff2")
data = cbind(data,tsn)


# First all 2 combinations of all features extracted and feature engineered to take
# the top N features. 

data$diff1<-data$Field7-data$Field8
data$diff2<-data$Field7-data$Field9
data$diff65<-data$Field8-data$SalesField3
data$diff67<-data$Field8-data$SalesField5
data$diff71<-data$Field8-data$PersonalField1
data$diff72<-data$Field8-data$PersonalField2
data$diff75<-data$Field8-data$PersonalField9
data$diff77<-data$Field8-data$PersonalField10B
data$diff80<-data$Field8-data$PersonalField13
data$diff121<-data$Field9-data$SalesField5
data$diff126<-data$Field9-data$PersonalField2
data$diff129<-data$Field9-data$PersonalField9
data$diff169<-data$CoverageField6A-data$SalesField1B
data$diff222<-data$CoverageField11A-data$SalesField2A
data$diff273<-data$CoverageField11B-data$SalesField2A
data$diff274<-data$CoverageField11B-data$SalesField2B
data$diff325<-data$SalesField1A-data$SalesField3
data$diff326<-data$SalesField1A-data$SalesField4
data$diff330<-data$SalesField1A-data$SalesField10
data$diff331<-data$SalesField1A-data$PersonalField1
data$diff336<-data$SalesField1A-data$PersonalField10A
data$diff337<-data$SalesField1A-data$PersonalField10B
data$diff339<-data$SalesField1A-data$PersonalField12
data$diff343<-data$SalesField1A-data$PersonalField26
data$diff345<-data$SalesField1A-data$PersonalField82
data$diff350<-data$SalesField1A-data$PropertyField16A
data$diff375<-data$SalesField1B-data$SalesField4
data$diff379<-data$SalesField1B-data$SalesField10
data$diff388<-data$SalesField1B-data$PersonalField12
data$diff406<-data$SalesField1B-data$GeographicField5B
data$diff414<-data$SalesField1B-data$tsne51
data$diff416<-data$SalesField1B-data$tsne53
data$diff417<-data$SalesField1B-data$tsne54
data$diff433<-data$SalesField2A-data$PersonalField10A
data$diff498<-data$SalesField2B-data$PropertyField39A
data$diff499<-data$SalesField2B-data$PropertyField39B
data$diff566<-data$SalesField4-data$PersonalField1
data$diff567<-data$SalesField4-data$PersonalField2
data$diff570<-data$SalesField4-data$PersonalField9
data$diff575<-data$SalesField4-data$PersonalField13
data$diff610<-data$SalesField5-data$PersonalField1
data$diff611<-data$SalesField5-data$PersonalField2
data$diff615<-data$SalesField5-data$PersonalField10A
data$diff616<-data$SalesField5-data$PersonalField10B
data$diff617<-data$SalesField5-data$PersonalField11
data$diff622<-data$SalesField5-data$PersonalField26
data$diff624<-data$SalesField5-data$PersonalField82
data$diff780<-data$PersonalField1-data$PersonalField9
data$diff783<-data$PersonalField1-data$PersonalField11
data$diff785<-data$PersonalField1-data$PersonalField13
data$diff819<-data$PersonalField2-data$PersonalField9
data$diff822<-data$PersonalField2-data$PersonalField11
data$diff824<-data$PersonalField2-data$PersonalField13
data$diff933<-data$PersonalField9-data$PersonalField11
data$diff935<-data$PersonalField9-data$PersonalField13
data$diff938<-data$PersonalField9-data$PersonalField26
data$diff939<-data$PersonalField9-data$PersonalField27
data$diff940<-data$PersonalField9-data$PersonalField82
data$diff941<-data$PersonalField9-data$PersonalField84
data$diff970<-data$PersonalField10A-data$PersonalField13
data$diff982<-data$PersonalField10A-data$PropertyField25
data$diff1004<-data$PersonalField10B-data$PersonalField13
data$diff1016<-data$PersonalField10B-data$PropertyField25
data$diff1021<-data$PersonalField10B-data$GeographicField5B
data$diff1069<-data$PersonalField12-data$PersonalField13
data$diff1106<-data$PersonalField13-data$PersonalField84
data$diff1185<-data$PersonalField15-data$tsne52
data$diff1220<-data$PersonalField27-data$PersonalField84
data$diff1232<-data$PersonalField27-data$GeographicField8A
data$diff1270<-data$PersonalField82-data$tsneF1
data$diff1305<-data$PropertyField1A-data$GeographicField1B
data$diff1328<-data$PropertyField1B-data$GeographicField1B
data$diff1596<-data$tsneF1-data$tsneF2
data$diff2000<-data$tsneTime1-data$tsneTime2
data$diff2001<-data$tsnedff1-data$tsnedff2

# all top integer columns from topF and diff now binned into 8 bins 
intCols  <- names(data[, sapply(data, class) %in% c('integer', 'numeric')])
intCols  <- intCols[intCols  %in% topF$x]
intCols <- c(intCols,grep("diff", names(data), value=TRUE))
cat("Adding data diffs for binning")
print(intCols)

dd <- discretize(data[,intCols],nbins=8)
new_col_names <- paste0("d_",colnames(dd[,1:ncol(dd)]))
names(dd) <- new_col_names
dd[] <- lapply(dd, factor)
data <- cbind(data, dd)


# More Features extracted from those 8 bins on row wise.

data$noOf1 <- rowSums(data[,new_col_names] ==1)
data$noOf2 <- rowSums(data[,new_col_names] ==2)
data$noOf3 <- rowSums(data[,new_col_names] ==3)
data$noOf4 <- rowSums(data[,new_col_names] ==4)
data$noOf5 <- rowSums(data[,new_col_names] ==5)
data$noOf6 <- rowSums(data[,new_col_names] ==6)
data$noOf7 <- rowSums(data[,new_col_names] ==7)
data$noOf8 <- rowSums(data[,new_col_names] ==8)
data$r1 <- (1+data$noOf1)/(1+data$noOf2)
data$r2 <- (1+data$noOf1)/(1+data$noOf3)
data$r3 <- (1+data$noOf1)/(1+data$noOf4)
data$r4 <- (1+data$noOf1)/(1+data$noOf5)
data$r5 <- (1+data$noOf2)/(1+data$noOf3)
data$r6 <- (1+data$noOf2)/(1+data$noOf4)
data$r7 <- (1+data$noOf2)/(1+data$noOf5)
data$r8 <- (1+data$noOf3)/(1+data$noOf4)
data$r9 <- (1+data$noOf3)/(1+data$noOf5)
data$r10 <- (1+data$noOf4)/(1+data$noOf5)

data$r11 <- (1+data$noOf6)/(1+data$noOf1)
data$r12 <- (1+data$noOf6)/(1+data$noOf2)
data$r13 <- (1+data$noOf6)/(1+data$noOf3)
data$r14 <- (1+data$noOf6)/(1+data$noOf4)
data$r15 <- (1+data$noOf6)/(1+data$noOf5)
data$r16 <- (1+data$noOf7)/(1+data$noOf1)
data$r17 <- (1+data$noOf7)/(1+data$noOf2)
data$r18 <- (1+data$noOf7)/(1+data$noOf3)
data$r19 <- (1+data$noOf7)/(1+data$noOf4)
data$r20 <- (1+data$noOf7)/(1+data$noOf5)
data$r21 <- (1+data$noOf8)/(1+data$noOf1)
data$r22 <- (1+data$noOf8)/(1+data$noOf2)
data$r23 <- (1+data$noOf8)/(1+data$noOf3)
data$r24 <- (1+data$noOf8)/(1+data$noOf4)
data$r25 <- (1+data$noOf8)/(1+data$noOf5)


data$r26 <- (1+data$noOf8)/(1+data$noOf6)
data$r27 <- (1+data$noOf8)/(1+data$noOf7)
data$r28 <- (1+data$noOf7)/(1+data$noOf6)


# Now convert all categorical values to integers for xgboost

feature.names <- names(data[,-1])
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(data[[f]])=="character") {
    levels <- unique(data[[f]])
    data[[f]] <- as.integer(factor(data[[f]], levels=levels))
  }
}

train <- data[1:length(response),feature.names]
test  <- data[(1+length(response)):nrow(data),feature.names]

# Redo response metrics for new added columns

train$QuoteConversion_Flag <- response

ftrain <- factorToNumeric(train, train, "QuoteConversion_Flag", new_col_names , c("mean","median","sd","IQR"))
ftest  <- factorToNumeric(train, test, "QuoteConversion_Flag", new_col_names, c("mean","median","sd","IQR"))
train <- cbind(train,ftrain)
test  <- cbind(test, ftest)

train$QuoteConversion_Flag <- NULL
data <- rbind(train, test)

# Using Caret Package to Remove Near Zero Variance variables across the entire data

nzv_cols <- nearZeroVar(data)
if(length(nzv_cols) > 0) data <- data[, -nzv_cols]


train <- data[1:length(response),]
test  <- data[(1+length(response)):nrow(data),]

# Data Preparation is done! Now lets do K Fold Cross Vaidation

train$QuoteConversion_Flag <- response
folds = createFolds(train$QuoteConversion_Flag,k=kfolds,list=TRUE,returnTrain=TRUE)
train$QuoteConversion_Flag <- NULL
origtrain <- train 

for (fld in 1:length(folds)) 
{
train <- origtrain
control <- train[folds[[fld]]*-1,]
train <- train[folds[[fld]],]

feature.names <- names(data)

dcontrol<-xgb.DMatrix(data=data.matrix(control),label=response[folds[[fld]]*-1])
dtrain<-xgb.DMatrix(data=data.matrix(train),label=response[folds[[fld]]])
watchlist<-list(val=dcontrol,train=dtrain)

param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 6, "min_child_weight" = 1, "gamma" = 0, "subsample" = .83, "colsample_bytree" = 0.77,"nthread" = 32)
bst1 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)
pred_control1 <- predict(bst1, data.matrix(control[,feature.names]))

param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 7, "min_child_weight" = 3, "gamma" = 1, "subsample" = .8, "colsample_bytree" = 0.7,"nthread" = 32)
bst2 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)
pred_control2 <- predict(bst2, data.matrix(control[,feature.names]))

param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 12, "min_child_weight" = 6, "gamma" = 2, "subsample" = .9, "colsample_bytree" = 0.5,"nthread" = 32)
bst3 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)
pred_control3 <- predict(bst3, data.matrix(control[,feature.names]))


param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 3, "min_child_weight" = 3, "gamma" = 0, "subsample" = .83, "colsample_bytree" = 0.77,"nthread" = 32)
bst4 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)
pred_control4 <- predict(bst4, data.matrix(control[,feature.names]))


param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 8, "min_child_weight" = 5, "gamma" = 1, "subsample" = .8, "colsample_bytree" = 0.7,"nthread" = 32)
bst5 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)
pred_control5 <- predict(bst5, data.matrix(control[,feature.names]))

param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 10, "min_child_weight" = 5, "gamma" = 0, "subsample" = .9, "colsample_bytree" = 0.8,"nthread" = 32)
bst6 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)
pred_control6 <- predict(bst6, data.matrix(control[,feature.names]))

pred_test1 <- predict(bst1, data.matrix(test[,feature.names]))
pred_test2 <- predict(bst2, data.matrix(test[,feature.names]))
pred_test3 <- predict(bst3, data.matrix(test[,feature.names]))
pred_test4 <- predict(bst4, data.matrix(test[,feature.names]))
pred_test5 <- predict(bst5, data.matrix(test[,feature.names]))
pred_test6 <- predict(bst6, data.matrix(test[,feature.names]))


predictions <- data.frame(pred_control1,pred_control2,pred_control3,pred_control4,pred_control5,pred_control6)
yVALIDATE <- response[folds[[fld]]*-1]

 eval1 <- colAUC(pred_control1,yVALIDATE)
 eval2 <- colAUC(pred_control2,yVALIDATE)
 eval3 <- colAUC(pred_control3,yVALIDATE)
 eval4 <- colAUC(pred_control4,yVALIDATE)
 eval5 <- colAUC(pred_control5,yVALIDATE)
 eval6 <- colAUC(pred_control6,yVALIDATE)


performance <- c(eval1,eval2,eval3,eval4,eval5,eval6)
performance <- (performance) / sum(performance)


 evaluate <- function(string = c()) {
            
            stringRepaired <- as.numeric(string)/sum(as.numeric(string))
            
            weightedprediction <- as.numeric(rowSums(t(as.numeric(stringRepaired) * t(predictions))))
            
            returnVal <- -colAUC(weightedprediction,yVALIDATE)

            returnVal
   }
   
  ########################################################################################################################################
  #genalg package: genetic algorithm
    
    cat('   Genetic Algorithm \n') 
    
    tuning <- list(popSize=42,iters=500,mutationChance=1/42,elitism=2)  
    #tuning <- list(popSize=rbga.popSize,iters=rbga.iters,mutationChance=rbga.mutationChance,elitism=rbga.elitism)  
    grid <- expand.grid(tuning)
    
    perf <- data.frame(matrix(nrow=nrow(grid),ncol=7+ncol(grid)))
    #note: everywhere in perf, the auc is not only for auc but also for sens and spec
    names(perf) <- c("weight1","weight2","weight3","weight4","weight5","weight6","auc")
    colnames(perf)[(sum(!is.na(colnames(perf))) + 1) : (sum(!is.na(colnames(perf))) + length(colnames(grid)))] <- colnames(grid)
   
    for (i in 1:nrow(grid)){
      
      rbga.results <- rbga(stringMin=rep(0, ncol(predictions)), 
                           stringMax=rep(1, ncol(predictions)), 
                           popSize = grid[i,]$popSize, 
                           iters = grid[i,]$iters, 
                           mutationChance = grid[i,]$mutationChance,
                           elitism=grid[i,]$elitism,
                           evalFunc = evaluate, showSettings=TRUE, verbose=TRUE)
      
      weightsRBGA <- rbga.results$population[which.min(rbga.results$evaluations),]
      weightsRBGA <- as.numeric(weightsRBGA)/sum(as.numeric(weightsRBGA))
      
      perf[i,] <- c(weightsRBGA,-rbga.results$evaluations[which.min(rbga.results$evaluations)],grid[i,])   
    }
    
    weightsRBGA  <- perf[which.max(perf$auc),c("weight1","weight2","weight3","weight4","weight5","weight6")]
   
fname <- paste0("weightsRBGA_ens6",fld,".RData")
save(weightsRBGA, file=fname)
cat("eval1:")
print(eval1)
cat("eval2:")
print(eval2)
cat("eval3:")
print(eval3)
cat("eval4:")
print(eval4)
cat("eval5:")
print(eval5)
cat("eval6:")
print(eval6)
}

# Save the intermediate work, just in case.
save.image()


# Now lets look at the entire train data and repeat the same process.

set.seed(987692)
train <- origtrain

tra <- train
feature.names <- names(data)

nrow(train)
h<-sample(nrow(train),2000)

dcontrol<-xgb.DMatrix(data=data.matrix(tra[h,]),label=response[h])
dtrain<-xgb.DMatrix(data=data.matrix(train),label=response)
watchlist<-list(val=dcontrol,train=dtrain)


param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 6, "min_child_weight" = 1, "gamma" = 0, "subsample" = .83, "colsample_bytree" = 0.77,"nthread" = 32)
bst1 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)

param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 7, "min_child_weight" = 3, "gamma" = 1, "subsample" = .8, "colsample_bytree" = 0.7,"nthread" = 32)
bst2 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)

param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 12, "min_child_weight" = 6, "gamma" = 2, "subsample" = .9, "colsample_bytree" = 0.5,"nthread" = 32)
bst3 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)

param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 3, "min_child_weight" = 3, "gamma" = 0, "subsample" = .83, "colsample_bytree" = 0.77,"nthread" = 32)
bst4 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)

param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 8, "min_child_weight" = 5, "gamma" = 1, "subsample" = .8, "colsample_bytree" = 0.7,"nthread" = 32)
bst5 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)

param <- list("objective" = "binary:logistic",eta =0.018, "max_depth" = 10, "min_child_weight" = 5, "gamma" = 0, "subsample" = .9, "colsample_bytree" = 0.8,"nthread" = 32)
bst6 <- xgb.train(param, data=dtrain, 3500, watchlist, feval=evalgini)


pred_test1 <- predict(bst1, data.matrix(test[,feature.names]))
pred_test2 <- predict(bst2, data.matrix(test[,feature.names]))
pred_test3 <- predict(bst3, data.matrix(test[,feature.names]))
pred_test4 <- predict(bst4, data.matrix(test[,feature.names]))
pred_test5 <- predict(bst5, data.matrix(test[,feature.names]))
pred_test6 <- predict(bst6, data.matrix(test[,feature.names]))
save.image()

for (i in 1:length(folds)) 
{

fname <- paste0("weightsRBGA_ens6",i,".RData")
load(fname)

ans <- weightsRBGA$weight1*pred_test1 + weightsRBGA$weight2*pred_test2 + weightsRBGA$weight3*pred_test3 + weightsRBGA$weight4*pred_test4 + weightsRBGA$weight5*pred_test5 + weightsRBGA$weight6*pred_test6


submission <- data.frame(QuoteNumber=testquote, QuoteConversion_Flag=ans)
fname1 <- paste0("ga_ens6",i,".csv")
write.csv(submission, file=fname1, row.names=F, quote=F)
}
