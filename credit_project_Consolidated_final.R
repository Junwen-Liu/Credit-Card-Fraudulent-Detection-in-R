library(tidyr)
library(dplyr)
library(tidyverse)
library(corrplot)
library(ggplot2)
library(caret)
library(car)
library(data.table)
library(ggplot2)
library(plyr)
library(pROC)
library(glmnet)
library(Rtsne)
library(xgboost)
library(randomForest)
library(DMwR)
library(gridExtra)
library(ROSE)
library(class)
library(rpart)
library(rpart.plot)
theme_set(theme_classic())

#keep in mind to change the repository of your dataset
dataset <- read.csv("creditcard.csv", header=TRUE)

#Data-type check
str(dataset)

#Missing values check:
row.has.na <- apply(dataset, 1, function(x){any(is.na(x))})
na <- sum(row.has.na) 
rows <- nrow(dataset) 
(na/rows)*100 #No missing values have been identified.

#Normalizing variables in dataframe:
dataset[,1:30] <- as.data.frame(apply(dataset[,1:30],2,function(x)(x-min(x))/(max(x)-min(x))))

#Checking for duplications in dataframe:
deduped.data <- unique(dataset)
duplicated(deduped.data) #approx. 1081 duplicate entries were removed

#Based on data visualization, Time variable does not have predicting power. Hence will be removed in data modelling.
dataset_all <- subset(deduped.data, select = -c(Time))

#Convert variable Class to vector:
dataset_all$Class <- as.factor(dataset_all$Class)

##Data Under-sampling and Partition:
#Under-sampling our dataset for class = 0 in ratio 1:5 (fraudulent:Non-fraudulent)
non_fraud <- filter(dataset_all, Class == 0)
fraud <- filter(dataset_all, Class == 1)

#Random selection of 2365 obs from non_fraud dataset
set.seed(10)
non_fraud1 <- sample_n(non_fraud, size=2365, replace =F)
unused_non_fraud <- setdiff(non_fraud, non_fraud1)

#Concatenate 2 dataframes 
credit_final <- rbind(fraud, non_fraud1)

#Randomly splitting dataset to Train_val and Test (90:10)
set.seed(12)
train_val <- sample(nrow(credit_final), 0.9*nrow(credit_final))
test_cases_df <- setdiff(1:nrow(credit_final), train_val)

train_val_df <- credit_final[train_val, ]

#Randomly splitting Train_val into Train and Validation(80:10)
set.seed(12)
train_cases_df <- sample(nrow(train_val_df), 0.88*nrow(train_val_df))
val_cases_df <- setdiff(1:nrow(train_val_df), train_cases_df)

train_df <- train_val_df[train_cases_df, ]
test_df <- credit_final[test_cases_df, ]
val_df <- train_val_df[val_cases_df, ]

##=======================================================================================================================
# function to calculate ROC
ROC_func <- function(df, label_colnum, score_colnum, add_on = F, color = "black"){
  # Sort by score (high to low)
  df <- df[order(-df[,score_colnum]),]
  rownames(df) <- NULL  # Reset the row number to 1,2,3,...
  n <- nrow(df)
  # Total # of positive and negative cases in the data set
  P <- sum(df[,label_colnum] == 1)
  N <- sum(df[,label_colnum] == 0)
  
  # Vectors to hold the coordinates of points on the ROC curve
  TPR <- c(0,vector(mode="numeric", length=n))
  FPR <- c(0,vector(mode="numeric", length=n))
  
  # Calculate the coordinates from one point to the next
  AUC = 0
  for(k in 1:n){
    if(df[k,label_colnum] == 1){
      TPR[k+1] = TPR[k] + 1/P
      FPR[k+1] = FPR[k]
    } else{
      TPR[k+1] = TPR[k]
      FPR[k+1] = FPR[k] + 1/N
      AUC = AUC + TPR[k+1]*(1/N)
    }
  }
  
  # Plot the ROC curve
  if(add_on){
    points(FPR, TPR, main=paste0("ROC curve"," (n = ", n, ")"), type = 'l', col=color, cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.2)
  } else{
    plot(FPR, TPR, main=paste0("ROC curve"," (n = ", n, ")"), type = 'l', col=color, cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.2)
  }
  return(AUC)
}

#======================================================================================================================
#This section previewed all selected models via k-fold validation on each type of models, and comparing them in terms of accuracy and ROC value, the detailed process of selected best models through k-fold validation will be provided in following section.

## Train/Retrain the models using selected parameters for each model and evaluating performance:
test_df_new <- rbind(test_df, unused_non_fraud)
actual <- test_df_new$Class #Ground truth of test set

#-----------------------------------------------------------------------------------------------------
#Logistic Regression - Validation set approach: 
log_mod <- glm(Class ~ . -V26-V18-V3-V2-V17-V25-V11-V24-V12-V19-V15, family = "binomial", data = train_val_df[,-c(31)])
scoreLogReg <- predict(log_mod, test_df_new, type='response')
score_lr<- ifelse(scoreLogReg > 0.9, 1, 0)

df <- data.frame(score_lr)
predicted_lr <- c(df$score_lr) #Predicted class of LogReg
(CM_LReg <- table(actual,predicted_lr)) #confusion matrix of LogReg
Accuracy_LReg <- sum(diag(CM_LReg))/sum(CM_LReg)
cat("The model accuracy of Logistic Regression is ", format(round(Accuracy_LReg*100, 2), nsmall = 2), "%")  

#ROC
df_lr <- data.frame(true.class = actual, LR =scoreLogReg)
ROC_func(df_lr, 1, 2, add_on = F) #Log Reg

#-----------------------------------------------------------------------------------------------------
#KNN using K = 3:
knn_mod  <- knn(train = train_val_df[,-c(30)], test = test_df_new[,-30], cl = train_val_df$Class, k = 3, prob=TRUE, use.all=TRUE) #Predicted class of KNN
CM_KNN <- table(actual,knn_mod) #confusion matrix of KNN
Accuracy_KNN <- sum(diag(CM_KNN))/sum(CM_KNN)
cat("The model accuracy of KNN is ", format(round(Accuracy_KNN*100, 2), nsmall = 2),"%")

#ROC
KNN_prob <- attr(knn_mod, "prob")
df_knn <- data.frame(true.class = actual, KNN=KNN_prob)
ROC_func(df_knn, 1, 2, add_on = F) #KNN


#-------------------------------------------------------------------------------------------------------
#Decision Tree with depth = 3:
dTree_mod <- rpart(train_val_df[, 30]~ . , train_val_df[,-c(30,31)], method = 'class', maxdepth = 3)
predict_dTree <- predict(dTree_mod, test_df_new, type = 'class')
result_dTree <- cbind.data.frame(as.data.frame(predict_dTree), as.data.frame(test_df_new[,30]))
rpart.plot(dTree_mod)

predicted_DTree <- result_dTree$predict_dTree
(CM_DTree <- table(test_df_new[,30],predicted_DTree)) #confusion matrix of KNN
Accuracy_dTree <- sum(diag(CM_DTree))/sum(CM_DTree)
cat("The optimal accuracy of decision tree is ", format(round(Accuracy_dTree*100, 2), nsmall = 2),"%")

#ROC
predict_dTree_prob <- predict(dTree_mod, test_df_new, type = 'prob')
df_dtree <- data.frame(true.class = test_df_new[,30], DTree =predict_dTree_prob[,2])
ROC_func(df_dtree, 1, 2, add_on = F) #Decision Tree

#------------------------------------------------------------------------------------------

#Random forest with selected hyperparameters:
model_rf <- train(Class~.,train_val_df,method = "rf",importance = TRUE, nodesize = 14,
                  ntree = 900,maxnodes = 90)
 prediction_rf <-predict(model_rf, test_df_new)
CM_RF <- table(actual,prediction_rf) #Confusion matrix of Random Forest
Accuracy_RF <- sum(diag(CM_RF))/sum(CM_RF)
cat("The model accuracy of Random Forest is ", format(round(Accuracy_RF*100, 2), nsmall = 2), "%")  

#ROC
predict_rf_prob <- predict(model_rf, type = 'prob', test_df_new)
df_rdftree <- data.frame(true.class = test_df_new[,30], RF =predict_rf_prob[,2])
ROC_func(df_rdftree, 1, 2, add_on = F) #Random Forest

#-------------------------------------------------------------------------------------------

# Model comparison using ROC: 
df_all <- data.frame(df$test_df_new.Class, LR = scoreLogReg, KNN =KNN_prob, DTree =predict_dTree_prob[,2], RF = predict_rf_prob[,2])
ROC_func(df_all, 1, 2, add_on = F) #LR
ROC_func(df_all, 1, 3, add_on = T, color = "blue") #KNN
ROC_func(df_all, 1, 4, add_on = T, color = "red") #DTree
ROC_func(df_all, 1, 5, add_on = T, color = "green") #RF

legend("bottomright",legend=c("Logistic Regression", "KNN", "Decision Tree","Random forest"),
       lty=c(1,1,1,1), cex = 0.5,
       lwd=c(.125, 0.125, 0.125, 0.125),col=c("black", "blue", "red", "green"))
#---------------------------------------------------------------------------------------------

#Performance Comparision
Comp <- c("LogReg", "kNN", "DT", "RF")
TP <- c(272653,272339,271940,272686)
FP <- c(415,729,1128,382)
FN <- c(7,4,5,4)
TN <- c(37,40,39,40)
sensitivity = TP / (TP+FN)
specificity = TN / (TN+FP)
precision = TP / (TP+FP)
Accuracy <- (TP+TN) / (TP+TN+FP+FN)
F1 = 2* (precision * sensitivity) / (precision + sensitivity)
FPR = 1 - ( TN / (TN+FP))
Performance_comp_table <- data.frame(Comp,TP,FP,FN,TN,sensitivity,specificity,precision,Accuracy, F1,FPR)
Performance_comp_table

#=============================================================================================================
#Supplementary calculation supporting feature or parameters selected above
#=============================================================================================================

#Feature selection for logistic regression
glm.fits <- glm(Class ~ ., family = "binomial", data = dataset_all) #All data will be used for this procedure
summary(glm.fits)

#collinearity check 
vif(glm.fits)

# Backward selection process - Removing variable based on p-value
BIC(glm.fits)
AIC(glm.fits) #AIC will be chosen as global criteria for variable backward selection process

glm.fits1 <- glm(Class ~ . -V26, family = "binomial", data = dataset_all)
AIC(glm.fits1)

glm.fits2 <- glm(Class ~ . -V26-V18, family = "binomial", data = dataset_all)
AIC(glm.fits2)

glm.fits3 <- glm(Class ~ . -V26-V18-V3, family = "binomial", data = dataset_all)
AIC(glm.fits3)

glm.fits4 <- glm(Class ~ . -V26-V18-V3-V2, family = "binomial", data = dataset_all)
AIC(glm.fits4)

glm.fits5 <- glm(Class ~ . -V26-V18-V3-V2-V17, family = "binomial", data = dataset_all)
AIC(glm.fits5)

glm.fits6 <- glm(Class ~ . -V26-V18-V3-V2-V17-V25, family = "binomial", data = dataset_all)
AIC(glm.fits6)

glm.fits7 <- glm(Class ~ . -V26-V18-V3-V2-V17-V25-V11, family = "binomial", data = dataset_all)
AIC(glm.fits7)

glm.fits8 <- glm(Class ~ . -V26-V18-V3-V2-V17-V25-V11-V24, family = "binomial", data = dataset_all)
AIC(glm.fits8)

glm.fits9 <- glm(Class ~ . -V26-V18-V3-V2-V17-V25-V11-V24-V12, family = "binomial", data = dataset_all)
AIC(glm.fits9)

glm.fits10 <- glm(Class ~ . -V26-V18-V3-V2-V17-V25-V11-V24-V12-V19, family = "binomial", data = dataset_all)
AIC(glm.fits10)

glm.fits11 <- glm(Class ~ . -V26-V18-V3-V2-V17-V25-V11-V24-V12-V19-V15, family = "binomial", data = dataset_all)
AIC(glm.fits11)

glm.fits12 <- glm(Class ~ . -V26-V18-V3-V2-V17-V25-V11-V24-V12-V19-V15-V5, family = "binomial", data = dataset_all)
AIC(glm.fits12)

#By conducting backward selection procedure based on p-value of each variable, we found the model without V26,V18,V3,V2,V17,V25,V11,V24,V12,V19,V15 delivers the best fit.

#-----------------------------------------------------------------------------------------------------------

k=5 #lnumber of fold in cross validation
K=5 #Number of models 



set.seed(1093)
train_val_df$id <- sample(1:k, nrow(train_val_df), replace = TRUE)
list <- 1:k
best_K_value <- 0
best_accurary <- 0



for(m in 1:K) #loop through multiple K value of different knn models
{
  K_value <- strtoi(switch(m,"2","3", "4","5","10"))
  prediction <- data.frame()
  testsetCopy <- data.frame()
  result <- data.frame()
  
  progress.bar <- create_progress_bar("text") #Creating a progress bar to know the status of CV
  progress.bar$init(k)
  
  for(i in 1:k){ #function for k fold
    trainingset <- subset(train_val_df, id %in% list[-i])
    validationset <- subset(train_val_df, id %in% c(i))
    
    mymodel <- knn(train = trainingset[,-30], test = validationset[,-30], cl = trainingset$Class, k = K_value, use.all=TRUE) #run a knn
    prediction <- rbind(prediction, as.data.frame(mymodel)) # append this iteration's predictions to the end of the prediction data frame
    testsetCopy <- rbind(testsetCopy, as.data.frame(validationset[,30])) # append this iteration's test set to the test set copy data frame
    
    progress.bar$step()
  }
  result <- cbind(prediction, testsetCopy[, 1])
  names(result) <- c("predicted", "Class")
  
  confusM <- confusionMatrix(factor(result[,1]), factor(result[,2]), positive = "1") #confusion matrix
  Accuracy <- confusM$overall["Accuracy"]
  
  if (Accuracy > best_accurary) {
    best_accurary <- Accuracy
    best_K_value <- K_value
  } 
}
cat("The best K value is ", best_K_value)
cat("The accuracy is ", best_accurary)

#---------------------------------------------------------------------------------------------------------

# Selecting depth for Decision tree using K-fold validation:
k=5 #Number of fold in cross validation
H=3 #Number of models 

set.seed(1093)
train_val_df$id <- sample(1:k, nrow(train_val_df), replace = TRUE)
list <- 1:k
best_H_value <- 0
best_accurary <- 0

for(m in 1:H) #loop through multiple K value of different depth
{
  H_value <- strtoi(switch(m,"1","3","5"))
  prediction <- data.frame()
  testsetCopy <- data.frame()
  result <- data.frame()
  
  progress.bar <- create_progress_bar("text") #Creating a progress bar to know the status of CV
  progress.bar$init(k)
  
  for(i in 1:k){ #function for k fold
    trainingset <- subset(train_val_df, id %in% list[-i])
    validationset <- subset(train_val_df, id %in% c(i))
    
    decisionTree_model <- rpart(trainingset[, 30]~ . , trainingset[,-30], method = 'class', maxdepth = H_value) #run descrision tree model
    predicted_val <- predict(decisionTree_model, validationset, type = 'class')
    probability <- predict(decisionTree_model, validationset, type = 'prob')
    #rpart.plot(decisionTree_model)
    
    prediction <- rbind(prediction, data.frame(predicted = as.data.frame(predicted_val), score = as.data.frame(probability[, 2])))
    testsetCopy <- rbind(testsetCopy, as.data.frame(validationset[,30]))
    
    progress.bar$step()
  }
  result <- cbind(prediction[, 1], testsetCopy[, 1])
  names(result) <- c("predicted", "Class")
  
  #confusion matrix of KNN
  CM_DT_CV <- table(result[, 2],result[, 1]) 
  Accuracy <- sum(diag(CM_DT_CV))/sum(CM_DT_CV)
  cat("\n The accuracy of Decision Tree when H=", H_value, " is ", format(round(Accuracy*100, 2), nsmall = 2),"%")
  
  if (Accuracy > best_accurary) {
    best_accurary <- Accuracy
    best_H_value <- H_value
  } 
  
  #plot ROC Curve
  roc_result <- cbind.data.frame(testsetCopy[, 1],prediction[, 2])
  if (m == 1) #plot ROC
  {
    ROC_func(roc_result, 1, 2, add_on = F)
  }else if(m == 2){
    ROC_func(roc_result, 1, 2, add_on = T, color = 'blue')
  }else{
    ROC_func(roc_result, 1, 2, add_on = T, color = 'red')
  }
}
legend("bottomright",legend=c("Decision Tree Cross Validation ROC", "H=1", "H=3","H=5"),
       lty=c(1,1,1,1), cex = 0.5,
       lwd=c(.125, 0.125, 0.125, 0.125),col=c("black", "blue", "red"))

#The best depth value is  3 with accuracy 97%
cat("The best depth value is ", best_H_value) 
cat("/n The accuracy is ", best_accurary)

#---------------------------------------------------------------------------------------------------------
##Defining hyperparameters for Random Forest using train_val dataset:

X_train <- train_df[,-30]
X_test <- val_df[,-30] #Validation predictors
y_train <- train_df[,30]
y_test <- val_df[,30] #Validation response


customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("maxnodes", "ntree"), class = rep("numeric", 2), label = c("maxnodes", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, maxnodes = param$maxnodes, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

metric<-"accuracy"
control <- trainControl(method="cv", number=5, search='grid')

# Grid of parameters to select optimal combination:
tunegrid <- expand.grid(.maxnodes=c(80,90,100,110,120), .ntree=c(600,800, 900, 1000, 1100))

set.seed(100)
rf_gridsearch <- train(x=X_train, y=y_train, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
plot(rf_gridsearch)
rf_gridsearch$bestTune 




