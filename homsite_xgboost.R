####################
# Homesite Kaggle Competition
# https://github.com/dmlc/xgboost/blob/master/R-package/demo/custom_objective.R
####################

source('/Users/taposh/workspace/mlearning/supervised /classification_code/homesite/homesite_input.R')


nrow(train)
h<-sample(nrow(train),2500)

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$QuoteConversion_Flag[h])
#dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$QuoteConversion_Flag[-h])
dtrain<-xgb.DMatrix(data=data.matrix(tra),label=train$QuoteConversion_Flag)

watchlist<-list(val=dval,train=dtrain)
param <- list(  objective = "binary:logistic", 
                booster = "gbtree",
                eval_metric = "auc",
                eta = 0.03, # 0.06, #0.01,
                max_depth = 6, #changed from default of 8
                subsample = 0.83, # 0.7
                colsample_bytree = 0.77, # 0.7
                num_parallel_tree = 1,
                # alpha = 0.0001, 
                # lambda = 1
)

modelt <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 1800, 
                    verbose             = 1,
                    #early.stop.round    = 150,
                    #watchlist           = watchlist,
                    #nfold = 10,
                    maximize            = FALSE
)

#envc <- matrix(as.numeric(unlist(clf)),1800,4)
pred1 <- predict(modelt, data.matrix(test[,feature.names]))
submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
cat("saving the submission file\n")
write_csv(submission, "xgb_submission4.csv")


