#Copyright (c) 2016, Taposh Roy All rights reserved.

#Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following 
#conditions are met: * Redistributions of source code must retain the above copyright notice, this list of conditions and the 
#following disclaimer. * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and 
#the following disclaimer in the documentation and/or other materials provided with the distribution. * Neither the name of the 
#Krishna Kesavan nor the names of its contributors may be used to endorse or promote products derived from this software without 
#specific prior written permission.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT 
#NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL 
#TAPOSH DUTTA ROY BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT 
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED 
#AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY 
#WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 

# https://github.com/h2oai/h2o-world-2015-training/tree/master/tutorials/ensembles-stacking#h2o-ensemble-super-learning-in-h2o
# http://learn.h2o.ai/content/tutorials/ensembles-stacking/index.html

# Start H2O Cluster
library(h2oEnsemble)  # This will load the `h2o` R package as well
h2o.init(nthreads = -1)  # Start an H2O cluster with nthreads = num cores on your machine
h2o.removeAll() # Clean slate - just in case the cluster was already running

source('/homesite_input.R')

# Load Data into H2O Cluster
# First, import a sample binary outcome train and test set into the H2O cluster.
#Variables to ignore in the model calculations
ignore<-c("QuoteNumber","Original_Quote_Date","QuoteConversion_Flag")


#h2o modeling
train_hex<-as.h2o(factored_train)
test_hex<-as.h2o(factored_test)

#Training set and test set for model building
train_ind<-sample(seq_len(nrow(train_hex)), size = 200000)
train_hex1<-train_hex[train_ind,]
result <-train_hex[-train_ind,]
result_val <- result[2]

labels1 <- as.data.frame(result_val)

val_hex<-train_hex[-train_ind,]
val_hex<-val_hex[-2]

xvar <- setdiff(colnames(train_hex),ignore)
yvar <- "QuoteConversion_Flag"

metalearner <- "h2o.glm.wrapper"

# Specifying new learners
# Custom learner wrappers:

h2o.glm.1 <- function(..., alpha = 0.0) h2o.glm.wrapper(..., alpha = alpha) # adjust alpha
h2o.glm.2 <- function(..., alpha = 0.5) h2o.glm.wrapper(..., alpha = alpha) # adjust alpha
h2o.glm.3 <- function(..., alpha = 1.0) h2o.glm.wrapper(..., alpha = alpha) # adjust alpha
h2o.randomForest.1 <- function(..., ntrees = 200, nbins = 100, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.randomForest.2 <- function(..., ntrees = 200, sample_rate = 0.75, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.3 <- function(..., ntrees = 200, sample_rate = 0.85, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, sample_rate = sample_rate, seed = seed)
h2o.randomForest.4 <- function(..., ntrees = 200, nbins = 50, balance_classes = TRUE, seed = 1) h2o.randomForest.wrapper(..., ntrees = ntrees, nbins = nbins, balance_classes = balance_classes, seed = seed)
h2o.gbm.1 <- function(..., ntrees = 100, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, seed = seed)
h2o.gbm.2 <- function(..., ntrees = 100, nbins = 50, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, nbins = nbins, seed = seed)
h2o.gbm.3 <- function(..., ntrees = 100, max_depth = 10, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.4 <- function(..., ntrees = 100, col_sample_rate = 0.8, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.5 <- function(..., ntrees = 100, col_sample_rate = 0.7, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.6 <- function(..., ntrees = 100, col_sample_rate = 0.6, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, col_sample_rate = col_sample_rate, seed = seed)
h2o.gbm.7 <- function(..., ntrees = 100, balance_classes = TRUE, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, balance_classes = balance_classes, seed = seed)
h2o.gbm.8 <- function(..., ntrees = 100, max_depth = 3, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.9 <- function(..., ntrees = 100, max_depth = 5, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.gbm.10 <- function(..., ntrees = 200, max_depth = 5, seed = 1) h2o.gbm.wrapper(..., ntrees = ntrees, max_depth = max_depth, seed = seed)
h2o.deeplearning.1 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.2 <- function(..., hidden = c(200,200,200), activation = "Tanh", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.3 <- function(..., hidden = c(500,500), activation = "RectifierWithDropout", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.4 <- function(..., hidden = c(500,500), activation = "Rectifier", epochs = 50, balance_classes = TRUE, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, balance_classes = balance_classes, seed = seed)
h2o.deeplearning.5 <- function(..., hidden = c(100,100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.6 <- function(..., hidden = c(50,50), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)
h2o.deeplearning.7 <- function(..., hidden = c(100,100), activation = "Rectifier", epochs = 50, seed = 1)  h2o.deeplearning.wrapper(..., hidden = hidden, activation = activation, seed = seed)

# Customized base learner library

# This learner is from the demo, but takes too long for YHAT
learner <- c("h2o.glm.wrapper",
            "h2o.randomForest.1", "h2o.randomForest.2","h2o.randomForest.3", "h2o.randomForest.4",
            "h2o.gbm.1", "h2o.gbm.2", "h2o.gbm.3","h2o.gbm.4", "h2o.gbm.5", "h2o.gbm.6","h2o.gbm.7", "h2o.gbm.8","h2o.gbm.9","h2o.gbm.10",
            "h2o.deeplearning.1", "h2o.deeplearning.2", "h2o.deeplearning.3","h2o.deeplearning.5","h2o.deeplearning.6","h2o.deeplearning.7")

#learner2 <- c("h2o.deeplearning.5","h2o.deeplearning.6","h2o.deeplearning.7")

#gbmlearner <- c("h2o.glm.wrapper","h2o.gbm.1", "h2o.gbm.2", "h2o.gbm.3","h2o.gbm.4", "h2o.gbm.5", "h2o.gbm.6","h2o.gbm.7", "h2o.gbm.8","h2o.gbm.9","h2o.gbm.10")

# Note that models are run sequentially. H2O is exploring other ways to increase efficiency by adjusting how these are run across the cores.

# Train with new library:

fit <- h2o.ensemble(x = xvar, y = yvar, 
                    training_frame = train_hex1,
                    learner = learner, 
                    metalearner = metalearner,
                    cvControl = list(V = 10))

# Generate predictions on the test set:
pred1 <- predict(fit, val_hex)
predictions1 <- as.data.frame(pred1$pred)
labels <- labels1 #as.data.frame(train_hex2[,y])[,1]

# Evaluate the test set performance:

cvAUC::AUC(predictions = predictions1 , labels = labels)
# 0.9592096
# 0.96329  kaggle site: 0.96815
# We see a slight increase in performance by including a more diverse library.

# Base learner test AUC (for comparison)

L <-length(learner)
auc <- sapply(seq(L), function(l) cvAUC::AUC(predictions = as.data.frame(pred$basepred)[,l], labels = labels)) 
data.frame(learner, auc)

# learner       auc
# 1    h2o.glm.wrapper 0.9324184
# 2 h2o.randomForest.1 0.9568921
# 3 h2o.randomForest.2 0.9536917
# 4          h2o.gbm.1 0.9573190
# 5          h2o.gbm.6 0.9574733
# 6          h2o.gbm.8 0.9561598
# 7 h2o.deeplearning.1 0.9208486
# 8 h2o.deeplearning.6 0.9243657
# 9 h2o.deeplearning.7 0.9152592

pred <- predict(fit, test_hex)
test_predictions<-as.data.frame(as.numeric(pred$pred))

#Normalize
normalizer <- function(x){(x-min(x))/(max(x)-min(x))}
predictions_normalized <- normalizer(test_predictions$predict)
predictions_normalized <- as.data.frame(predictions_normalized)

#negative numbers 
test_predictions$predict <- ifelse(test_predictions$predict <0, 0, ifelse(test_predictions$predict>1,1,test_predictions$predict))

#########################
#Submission
#########################
df1 <- as.data.frame(as.numeric(test_hex$QuoteNumber))

#Normalized Submission
#submitter(df1,predictions_normalized$predict,"h2o_submission_ensemble_normalzied.csv")
#Non-normalized
#submitter(df1,test_predictions$predict,"h2o_submission_ensemble.csv")

xgboost <- read.csv("/Users/taposh/workspace/mlearning/supervised /classification_code/homesite/xgb_Shize_stop_3.csv")

predictions_ensemble<-cbind(xgboost,predictions_normalized$predict)
predictions_ensemble$QuoteConversion_Flag2<-(((predictions_ensemble$QuoteConversion_Flag)+(predictions_ensemble$predict))/2)

submitter(df1,predictions_ensemble$QuoteConversion_Flag2,"Xgboost_h2o_submission_ensemble2.csv")



##############
#read XGboost
##############
ensemble <- read.csv("/Users/taposh/workspace/mlearning/supervised /classification_code/homesite/Xgboost_h2o_submission_ensemble.csv")

xgboost <- read.csv("/Users/taposh/workspace/mlearning/supervised /classification_code/homesite/xgb_Shize_stop_3.csv")

predictions_ensemble<-cbind(xgboost,ensemble$QuoteConversion_Flag)
predictions_ensemble$QuoteConversion_Flag2<-((5.5*(predictions_ensemble$QuoteConversion_Flag)+(ensemble$QuoteConversion_Flag))/2)

submitter(df1,predictions_ensemble$QuoteConversion_Flag2,"Xgboost_h2o_submission_ensemble-ensemble_55.csv")

#######################




