##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## Project: SNOT-22
## Created: July 15, 2021
## Author: Ingrid Shu 
## Collaborators: Vijay Ramikrishnan, CoSIBS SNOT22 group 
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

source("SNOT_Data_Cleaning.R")

# Predictive modeling -----------------------------------------------------
# Models we will compare: 
# Random Forests: allows for non-linear and interaction effects.  Conditional inference trees with permutation-based variable importance scores for unbiased variable selection.
# Support vector machine with a radial basis kernel ("SVM-Radial") 
# Step-wise Logistic regression: traditional linear/logistic regression but uses backwards step-wise AIC variable selection ("LogReg-StepAIC")
# LASSO: similar to traditional linear/logistic regression except the regression coefficients are shrunk towards zero to perform variable selection (unimportant variables are given coefficients exactly equal to 0, effectively removing these variables from the model).  Assumes only linear and additive effects (i.e. no interactions)
# MARS: similar to traditional linear/logistic regression but uses stepwise methods for variable selection, and allows for non-linear and interaction effects (unlike LASSO) 

#Evaluating classification accuracy: Repeated 10-fold cross validation (100 repeats) was used to tune and evaluate the classification accuracy of each model (AUC, sensitivity, specificity)

mldat_bin <- pdat %>% dplyr::select(SNOT22Change_bin,all_of(mlpredictors)) 
mldat_bin <-  data.frame(mldat_bin[-which(is.na(mldat_bin$SNOT22Change_bin)),])

mldat_cts <- pdat %>% dplyr::select(SNOT22Change_cts,all_of(mlpredictors)) 
mldat_cts <-  data.frame(mldat_cts[-which(is.na(mldat_cts$SNOT22Change_cts)),])

#method: boot, cv, repeatedcv, LOOCV
#repeates: number of resampling iterations (resamples in boot, repetitions of the cv)
#number: folds in cv
#classProbs: should class probabilities be computed for classification models (long iwth predicted values)
#twoClassSummary: computes sensitivity, specificity, AUC (other options: defaultSummary)
#sampling: to handle class imbalances (https://topepo.github.io/caret/subsampling-for-class-imbalances.html)
ctrl_bin <- trainControl(method="repeatedcv", classProbs=TRUE, repeats=10,  number=10,
                         sampling = "down",
                         summaryFunction = twoClassSummary, savePredictions = TRUE) 

ctrl_cts <- trainControl(method="repeatedcv", number=10, repeats=10) 

# methods accepted by caret
# http://topepo.github.io/caret/train-models-by-tag.html#logistic-regression


# Impute missing covariate data -------------------------------------------
set.seed(123)
X <- data.frame(mldat_bin[,-which(colnames(mldat_bin)=="SNOT22Change_bin")])
X <- data.frame(mldat_bin[,-1])
X[sapply(X, is.character)] <- lapply(X[sapply(X, is.character)], 
                                     as.factor)
imputeX <- missForest(X)
imputedata_bin <- data.frame(SNOT22Change_bin=mldat_bin[,which(colnames(mldat_bin)=="SNOT22Change_bin")],imputeX$ximp)
imputedata_cts <- data.frame(SNOT22Change_cts=mldat_cts[,which(colnames(mldat_cts)=="SNOT22Change_cts")],imputeX$ximp)

#Models: change code to appropriate set 

# Multiple Regression -----------------------------------------------------
set.seed(seed)
mult_reg <- train(SNOT22Change_cts ~., data = imputedata_cts, method = "lm", 
                  trControl = ctrl_cts,
                  metric="RMSE")

# Logistic Regression -----------------------------------------------------
set.seed(seed)
log_reg <- train(SNOT22Change_bin ~., data = imputedata_bin, method = "glm", family = "binomial",
                 trControl = ctrl_bin, metric = "ROC")


# Ridge Regression -------------------------------------------------------

# Binary
tuneGrid.lambda <- seq(0.001, 10, length.out=10)

X2 <- model.matrix(SNOT22Change_bin ~ ., data=imputedata_bin)
Y <- imputedata_bin$SNOT22Change_bin

set.seed(seed)
ridge_bin <- train(X2, Y, method = "glmnet",family="binomial",
                   trControl = ctrl_bin ,
                   tuneGrid = expand.grid(alpha=0, lambda=tuneGrid.lambda),
                   metric="ROC")

# Cts
X2 <- model.matrix(SNOT22Change_cts ~ ., data=imputedata_cts)
Y <- imputedata_cts$SNOT22Change_cts

set.seed(seed)
ridge_cts <- train(X2, Y, method = "glmnet",
                   trControl = ctrl_cts ,
                   tuneGrid = expand.grid(alpha=0, lambda=tuneGrid.lambda),
                   metric="RMSE"
)

# LASSO ------------------------------------------------------------------

# Binary
tuneGrid.lambda <- seq(0.001, 10, length.out=10)

X2 <- model.matrix(SNOT22Change_bin ~ ., data=imputedata_bin)
Y <- imputedata_bin$SNOT22Change_bin

set.seed(123)
lasso_bin <- train(X2, Y, method = "glmnet",family="binomial",
                   trControl = ctrl_bin ,
                   tuneGrid = expand.grid(alpha=1, lambda=tuneGrid.lambda),
                   metric="ROC"
)

#Cts
set.seed(123)
lambda <- 10^seq(-3,3, length.out = 100)

lasso_cts <- train(
  SNOT22Change_cts ~., 
  data = imputedata_cts, 
  method = "glmnet",
  trControl = ctrl_cts,
  tuneGrid = expand.grid(alpha = 1, lambda = lambda),
  na.action = na.omit,
  metric="RMSE",
  preProcess = "scale"
)





# Principle Components Regression ----------------------------------------
set.seed(123)
tuneGrid = expand.grid(ncomp = seq(1, 31, by =1))

pcr_cts = train(
  SNOT22Change_cts ~., data = imputedata_cts, method = "pcr",
  scale = TRUE,
  preProcess = c("center", "scale"),
  trControl = trainControl("repeatedcv", number = 10, repeats = 10),
  tuneGrid = tuneGrid, 
  savePredictions = TRUE,
  returnResamp="all",
  metric = "RMSE"
)

# Partial Least Squares --------------------------------------------------
set.seed(123)
pls_cts <- train(
  SNOT22Change_cts ~.,
  data = imputedata_cts,
  method = "pls",
  preProcess = c("center", "scale"),
  trControl = trainControl(method = "repeatedcv", repeats = 10, number = 10),
  tuneGrid = tuneGrid,
  savePredictions = TRUE,
  returnResamp="all",
  metric = "RMSE"
)

# SVM --------------------------------------------------------------------
set.seed(123)
tune_svm_bin <- train(SNOT22Change_bin ~ ., data=imputedata_bin, method='svmRadialCost',
                      tuneLength=5,
                      trControl=ctrl_bin, metric="ROC", preProcess = c("center","scale"))

tune_svm_cts <- train(SNOT22Change_cts ~ ., data=imputedata_cts, method='svmRadialCost',
                      tuneLength=5,
                      trControl=ctrl_cts, metric="RMSE", preProcess = c("center","scale"))

# MARS --------------------------------------------------------------------
hyper_grid <- expand.grid(
  degree = 1:5, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)

# Binary
set.seed(seed)
cv_mars_bin <- train(
  SNOT22Change_bin ~ .,
  data=imputedata_bin,
  method = "earth",
  metric = "ROC",
  trControl = ctrl_bin,
  tuneGrid = hyper_grid
)

# Cts
set.seed(seed)
cv_mars_cts <- train(
  SNOT22Change_cts ~ .,
  data=imputedata_cts,
  method = "earth",
  metric = "RMSE",
  trControl = ctrl_cts,
  tuneGrid = hyper_grid
)

# ADABoost ----------------------------------------------------------------
#Only applicable for classification
nIter=2:3 #want to choose reasonable values for this (just as a test)
method.ada <- "Adaboost.M1"
tunegrid <- expand.grid(.nIter=nIter, .method=method.ada)

# Binary
set.seed(seed)
adaboost_bin <- train(
  SNOT22Change_bin ~ .,
  data=imputedata_bin,
  method = "adaboost",
  trControl = ctrl_bin,
  tuneGrid = tunegrid,
  metric = "ROC"
)

# # Conditional Inference Random Forest -------------------------------------
mtry=2:10 #2:(ncol(imputedata_bin)-1)
tunegrid_mtry <- expand.grid(.mtry=mtry)

# # Binary
set.seed(123)
rf_bin <- train(SNOT22Change_bin ~ .,
                method = "cforest",
                data= imputedata_bin,
                trControl = ctrl_bin ,
                tuneGrid = tunegrid_mtry,
                metric="ROC"
)

# Cts
set.seed(seed)
rf_cts <- train(SNOT22Change_cts ~ .,
                method = "cforest",
                data= imputedata_cts,
                trControl = ctrl_cts ,
                tuneGrid = tunegrid_mtry,
                metric="RMSE"
)
# 
# 
# # StepAIC -----------------------------------------------------------------
# # Binary
set.seed(seed)
step_bin <- train(SNOT22Change_bin ~ .,
                  method = "glmStepAIC",
                  data= imputedata_bin,
                  trControl = ctrl_bin ,
                  metric="ROC"
)

# Cts
set.seed(seed)
step_cts <- train(SNOT22Change_cts ~ .,
                  method = "glmStepAIC",
                  data= imputedata_cts,
                  trControl = ctrl_cts ,
                  metric="RMSE"
)


# Comparison of models ----------------------------------------------------
set.seed(123)
model_list_bin <- list(lasso = lasso_bin, ridge = ridge_bin, logistic_regression = log_reg, svm_bin = tune_svm_bin)
res_bin <- resamples(model_list_bin)
summary(res_bin)

model_list_cts <- list(pcr = pcr_cts, pls = pls_cts, multipleRegression = mult_reg, lasso = lasso_cts, ridge = ridge_cts, svm_cts = tune_svm_cts) #cforest = tune_cf_cts, svm = tune_svm_cts, )
res_cts <- resamples(model_list_cts)
summary(res_cts)


# Variable importance -----------------------------------------------------
library(RColorBrewer)
library(viridis)
# Random forest--------------------
# Binary 
vimp_bin <- varImp(rf_bin)
vimp_bin2 <- data.frame(Feature=rownames(vimp_bin$importance), Importance=vimp_bin$importance$Overall)
vimp_bin2 <- vimp_bin2[order(vimp_bin2$Importance, decreasing = FALSE), ]
vimp_bin2$Feature <- factor(vimp_bin2$Feature, levels=vimp_bin2$Feature)

p_bin <- ggplot(vimp_bin2, aes(y = Feature, x = Importance)) +
  geom_bar(stat="identity", aes(fill = Importance), color = "black")+
  xlab("Importance")+ylab("Feature")+ggtitle("Random Forest")+
  scale_fill_viridis()+
  theme(legend.position = "none") +
  theme(text = element_text(size=8))

p_bin

# Cts 
vimp_cts <- varImp(rf_cts)
vimp_cts2 <- data.frame(Feature=rownames(vimp_cts$importance), Importance=vimp_cts$importance$Overall)
vimp_cts2 <- vimp_cts2[order(vimp_cts2$Importance, decreasing = FALSE), ]
vimp_cts2$Feature <- factor(vimp_cts2$Feature, levels=vimp_cts2$Feature)

p_cts <- ggplot(vimp_cts2, aes(y = Feature, x = Importance)) +
  geom_bar(stat="identity")+
  xlab("Importance")+ylab("Feature")+ggtitle("Random Forest")


gridExtra::grid.arrange(p_bin, p_bin, ncol = 2)


# LASSO (continuous)
vimp_cts <- varImp(lasso_cts)
vimp_cts2 <- data.frame(Feature=rownames(vimp_cts$importance), Importance=vimp_cts$importance$Overall)
vimp_cts2 <- vimp_cts2[order(vimp_cts2$Importance, decreasing = FALSE), ]
vimp_cts2$Feature <- factor(vimp_cts2$Feature, levels=vimp_cts2$Feature)


p_cts <- ggplot(vimp_cts2[vimp_cts2$Importance != 0,], aes(y = Feature[Importance != 0], x = Importance[Importance != 0])) +
  geom_bar(stat="identity", aes(fill = Importance), color = "black")+
  xlab("Importance")+ylab("Feature")+ggtitle("LASSO (continuous)") + 
  #scale_fill_distiller(palette = "YlOrRd", trans = "reverse")+
  scale_fill_viridis()+
  theme(legend.position = "none") 
  
p_cts
