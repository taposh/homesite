####################
# Homesite Kaggle Competition
# https://github.com/dmlc/xgboost/blob/master/R-package/demo/custom_objective.R
####################
setwd('/Users/taposh/workspace/mlearning/supervised /classification_code/homesite/')
#Package check function
require <- function(x) { 
  if (!base::require(x, character.only = TRUE)) {
    install.packages(x, dep = TRUE) ; 
    base::require(x, character.only = TRUE)
  } 
}

#Read the input from a webfile
require('RCurl')
getdatafromweb <- function(url,header = FALSE) {
  x <- getURL(url)
  out <- read.delim(textConnection(x),header=header)
  out
}

#Submit to Homesite
submitter <- function(id,predictions,filename)
{
  submission<-cbind(id,predictions)
  colnames(submission) <- c("QuoteNumber", "QuoteConversion_Flag")
  submission <- as.data.frame(submission)
  write_csv(submission, filename)
}


library(readr)
library(xgboost)
library(dplyr)

train <- read.csv("/Users/taposh/workspace/mlearning/supervised /classification_code/homesite/train.csv")
test <- read.csv("/Users/taposh/workspace/mlearning/supervised /classification_code/homesite/test.csv")

#my favorite seed^^
set.seed(1786)

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0


# seperating out the elements of the date column for the train set
train<-cbind(train,
             "day"=format(as.POSIXct(train[,"Original_Quote_Date"], format="%Y-%m-%d"), format="%d"),
             "month"=format(as.POSIXct(train[,"Original_Quote_Date"], format="%Y-%m-%d"), format="%m"),
             "year"=format(as.POSIXct(train[,"Original_Quote_Date"], format="%Y-%m-%d"), format="%Y")
)

# removing the date column
train <- train[,-c(2)]

# seperating out the elements of the date column for the train set
test<-cbind(test,
            "day"=format(as.POSIXct(test[,"Original_Quote_Date"], format="%Y-%m-%d"), format="%d"),
            "month"=format(as.POSIXct(test[,"Original_Quote_Date"], format="%Y-%m-%d"), format="%m"),
            "year"=format(as.POSIXct(test[,"Original_Quote_Date"], format="%Y-%m-%d"), format="%Y")
)

# removing the date column
test <- test[,-c(2)]

colnames(train)
colnames(test)

feature.names <- names(train)[c(3:301)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
#tra<-train[,feature.names]



########################################
# Deal with Categorical
########################################

# Categorical<-c("Field6","Field10","Field12","CoverageField8","CoverageField9","PersonalField7","PersonalField16","PersonalField17","PersonalField18","PersonalField19")
# for(i in Categorical){
#   BT<-train[i]
#   TT<-test[i]
#   BT2<-c()
#   TT2<-c()
#   for(j in unique(BT[,i])){
#     TempMat<-matrix(data=0,nrow=nrow(BT),ncol=1)
#     TempMat[BT==j]<-1
#     BT2<-cbind(BT2,mystr=TempMat)
#     TempMat2<-matrix(data=0,nrow=nrow(TT),ncol=1)
#     TempMat2[TT==j]<-1
#     TT2<-cbind(TT2,TempMat2)
#   }
#   mystr=paste("tmpmat", toString(j),toString(i), sep="_")
#   train<-cbind(train[,colnames(train)!=i],mystr=BT2)
#   test<-cbind(test[,colnames(test)!=i],mystr=TT2)
# }

########################################



