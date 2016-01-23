#Importing libraries
library(readr)
library(xgboost)
library(reshape2)
library(stringr)
library(h2o)

#Initialize H2O
h2o.init()

#Variables to ignore in the model calculations
ignore<-c("QuoteNumber","Original_Quote_Date","QuoteConversion_Flag")

#h2o modeling
train_hex<-as.h2o(train)
test_hex<-as.h2o(test)

#Training set and test set for model building
train_ind<-sample(seq_len(nrow(train_hex)), size = 50000)
train_hex1<-train_hex[train_ind,]
train_hex2<-train_hex[-train_ind,]

#------------------------DEEP LEARNING-------------------------------
# train_dl <- h2o.deeplearning(x = setdiff(colnames(train_hex),ignore), 
#                              y = "QuoteConversion_Flag", 
#                              training_frame = train_hex1,
#                              validation_frame=train_hex2)
# 
# #Model performance
# h2o.performance(model = train_dl, data=train_hex2)
# 
# #Predictions
# predictions_dl <- h2o.predict(train_dl, test_hex)
# 
xvar <- setdiff(colnames(train_hex),ignore)
yvar <- "QuoteConversion_Flag"

#------------------------RANDOM FOREST------------------------------
# train_rf <- h2o.randomForest(x = xvar, 
#                              y = yvar, 
#                              training_frame = train_hex1,
#                              validation_frame=train_hex2,
#                              binomial_double_trees = TRUE)

modelrf1 <- h2o.randomForest(x=xvar,y = yvar,training_frame = train_hex1,validation_frame=train_hex2,
                             mtries = 1,
                             ntree = 100)


#Model performance
h2o.performance(model = modelrf1, data=train_hex2)

#Predictions
predictions_rf <- h2o.predict(modelrf1, test_hex)

#----------------GRADIENT BOOSTED MACHINES-------------------

#Variable Importance GBM
modelgbm <- h2o.gbm(y =yvar, x = xvar, training_frame = train_hex1)
gbm.VI = modelgbm@model$variable_importances
print(gbm.VI)

#Model performance
h2o.performance(model = modelgbm, data=train_hex2)

#Predictions
predictions_gbm <- h2o.predict(modelgbm, test_hex)

h2o.auc(modelgbm)

########
#Optimizing GBM
##########
gbm_models <- c()
for (i in 1:5) {
  rand_numtrees <- sample(50:75,1) ## 1 to 50 trees
  rand_max_depth <- sample(5:15,1) ## 5 to 15 max depth
  gbm_model <- h2o.gbm(y = yvar, x = xvar, training_frame = test_hex,validation_frame=train_hex2, 
                       distribution="bernoulli", ntrees=rand_numtrees, max_depth=rand_max_depth)
  
  gbm_models <- c(gbm_models, gbm_model)
}

gbm_models[1]

best_err <- 0.78
for (i in 1:length(gbm_models)) {
  gbm_err <- h2o.auc(h2o.performance(gbm_models[[i]], test.hex)) 
  if (gbm_err > best_err) {
    best_gbm_auc <- gbm_err
    best_gbm_model <- gbm_models[[i]]
  }
}








#-----------------------ENSEMBLE-----------------------------

#read XGboost
#xgboost <- read.csv("/Users/taposh/workspace/mlearning/supervised /classification_code/homesite/xgb_Shize_stop_3.csv")
#xgboost_hex<-as.h2o(xgboost)

#The final values are the means of the three predictions
# predictions_ensemble<-h2o.cbind(predictions_dl, 
#                                 predictions_rf$predict, 
#                                 predictions_gbm$predict)
# 
# predictions_ensemble$QuoteConversion_Flag<-(predictions_ensemble$predict+predictions_ensemble$predict0+predictions_ensemble$predict1)/3
# 
# #If it's a negative value, then make it zero, if it's larger than 1, make it 1.
# predictions_ensemble$QuoteConversion_Flag<-ifelse(predictions_ensemble$QuoteConversion_Flag<0,
#                                                   0,
#                                                   ifelse(predictions_ensemble$QuoteConversion_Flag>1,
#                                                          1,
#                                                          predictions_ensemble$QuoteConversion_Flag))


#Normalize the results
range01 <- function(x){(x-min(x))/(max(x)-min(x))}
predictions_gbm_nm <- range01(predictions_gbm$predict)


predictions_gbm$predict<-ifelse(predictions_gbm$predict<0,
                                                  0,
                                                  ifelse(predictions_gbm$predict>1,
                                                         1,
                                                         predictions_gbm$predict))


#Submission
submission1<-h2o.cbind(test_hex$QuoteNumber,predictions_gbm$predict)
h2o.exportFile(submission, "/Users/taposh/workspace/mlearning/supervised /classification_code/homesite/h2o_submission_gbm.csv", force = FALSE)
submissiondf <- as.data.frame(submission1)
write_csv(submissiondf, "h2o_submission_gbm.csv")


#read XGboost
xgboost <- read.csv("/Users/taposh/workspace/mlearning/supervised /classification_code/homesite/xgb_Shize_stop_3.csv")

predictions_ensemble<-cbind(xgboost,submissiondf$predict)
predictions_ensemble$QuoteConversion_Flag2<-(0.3*(predictions_ensemble$QuoteConversion_Flag)+1.8*(submissiondf$predict))

submission4<-cbind(submissiondf$QuoteNumber,predictions_ensemble$QuoteConversion_Flag2)
submissiondf4 <- as.data.frame(submission2)
colnames(submissiondf4) <- c("QuoteNumber", "QuoteConversion_Flag")
names(submissiondf4)[1]<-paste("QuoteNumber") 
names(submissiondf4)[2]<-paste("QuoteConversion_Flag") 
write_csv(submissiondf4, "h2o_submission_gbm-xgboost-03-18.csv") #0.96301

# Check AUC



