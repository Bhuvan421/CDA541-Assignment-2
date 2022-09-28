getwd()
setwd("C:/Users/jonna/OneDrive/Desktop/CDA541 Statistical Data Mining 1/Assignments/Answers/Assignment2")

# Question 1:

library(ISLR)

data("College")

df = College

head(df, 10)
#View(College)
dim(df)     #College data is very small dataset
str(df)
summary(df)

#checking missing values?
sum(is.na(df))  #no missing values!!

set.seed(41)
library(tidyverse)
library(caret)

#(a) Split data into train-test:
training_samples = df$Apps %>% 
  createDataPartition(p = .8, list = FALSE)

train_data = df[training_samples, ]
test_data = df[-training_samples, ]

linear_model = train(Apps ~ ., data = train_data, method = 'lm')
summary(linear_model)

# The model is is good as R-squared and adjusted R-squared is higher and
# p-value is comparitively lower overall.

pred = predict(linear_model, newdata = test_data)

compare = data.frame(actual = test_data$Apps, predicted = pred)
head(compare,10)

# Error obtained:
linear_model$results

#(b) Ridge model

# Parameter tuning:
param_control = trainControl(method = 'repeatedcv', number = 10, repeats = 5)

ridge_reg_model = train(Apps ~ ., data = train_data, method = 'glmnet', 
                        trControl = param_control, 
                        tuneGrid = expand.grid(alpha =0, 
                                               lambda = seq(0.01, 500, 
                                                          length = 25)))

ridge_reg_model
plot(ridge_reg_model)

# Error obtained:
ridge_reg_model$results

#(d) Lasso Model

# Parameter tuning:
param_control = trainControl(method = 'repeatedcv', number = 10, repeats = 5)

lasso_model = train(Apps ~ ., data = train_data, method = 'glmnet', 
                    trControl = param_control, 
                    tuneGrid = expand.grid(alpha = 1, 
                                           lambda = seq(0.01, 100, length = 25)))

lasso_model
plot(lasso_model)

# Error obtained:
lasso_model$results

# no.of non-zero estimates:
coef(lasso_model$finalModel, lasso_model$bestTune$lambda)

#(g) Which is best?

cbind(linear_model$results$Rsquared, 
      ridge_reg_model$results$Rsquared, 
      lasso_model$results$Rsquared)
cbind(linear_model$results$RMSE, 
      ridge_reg_model$results$RMSE, 
      lasso_model$results$RMSE)

# Based on the error values and accuracy of all 3 models, 
# I would say, all models performed very well.



################################################################################
# Question 2:
################################################################################

df = read.csv("caravan-insurance-challenge.csv")
#head(df)
#str(df)   #origin column is 'str'.
#summary(df)

train_df = subset(df, ORIGIN == 'train')
test_df = subset(df, ORIGIN == 'test')
train_df = subset(train_df, select = -c(ORIGIN))
test_df = subset(test_df, select = -c(ORIGIN))
#str(train_df)
#str(test_df)

dim(train_df)
dim(test_df)

#missing values?
sum(is.na(train_df))  #no missing values
sum(is.na(test_df))   #no missing values


set.seed(41)

############
# OLS model:
############

ols_model = train(CARAVAN ~ ., data = train_df, method = 'lm')
summary(ols_model)

# The model is good as R-squared and adjusted R-squared is higher and
# p-value is comparitively lower overall.

pred = predict(ols_model, newdata = test_df)

compare = data.frame(actual = test_df$CARAVAN, predicted = pred)
head(compare,10)

# Error obtained:
ols_model$results

#####################
# Backward Selection:
#####################

bwd_model = lm(CARAVAN ~ ., data = train_df)
summary(bwd_model)

step(bwd_model, direction = 'backward')

# final AIC for backward elimination: AIC=-17108.4

# lm(formula = CARAVAN ~ MOSTYPE + MGEMLEEF + MOSHOOFD + MGODRK + 
# MRELGE + MOPLMIDD + MOPLLAAG + MBERBOER + MBERMIDD + MSKC + 
# MHHUUR + MZFONDS + MZPART + MINK123M + MINKGEM + MKOOPKLA + 
# PWAPART + PWALAND + PPERSAUT + PLEVEN + PGEZONG + PWAOREG + 
# PBRAND + PINBOED + ALEVEN + AGEZONG + AWAOREG + ABRAND + 
# APLEZIER + AFIETS + AINBOED + ABYSTAND, data = train_df)

# Suggested 32 independent variable out of 86 variables by backward selection 

bwd_reg_model = lm(formula = CARAVAN ~ MOSTYPE + MGEMLEEF + MOSHOOFD + MGODRK + 
                  MRELGE + MOPLMIDD + MOPLLAAG + MBERBOER + MBERMIDD + MSKC + 
                  MHHUUR + MZFONDS + MZPART + MINK123M + MINKGEM + MKOOPKLA + 
                  PWAPART + PWALAND + PPERSAUT + PLEVEN + PGEZONG + PWAOREG + 
                  PBRAND + PINBOED + ALEVEN + AGEZONG + AWAOREG + ABRAND + 
                  APLEZIER + AFIETS + AINBOED + ABYSTAND, data = train_df)

pred_bwd = predict(bwd_reg_model, newdata = test_df)

compare = data.frame(actual = test_df$CARAVAN, predicted = pred_bwd)
head(compare,10)

summary(bwd_reg_model)

rss_error_bwd = mean((test_df$CARAVAN - pred_bwd)^2)
rss_error_bwd

rmse_error_bwd = sqrt(mean(test_df$CARAVAN - pred_bwd)^2)
rmse_error_bwd

####################
# Forward Selection:
####################

fwd_model = lm(CARAVAN ~ 1, data = train_df)
summary(fwd_model)

step(fwd_model, direction = 'forward', scope = formula(bwd_model))

# final AIC for forward selection: AIC=-17108.56

# lm(formula = CARAVAN ~ PPERSAUT + APLEZIER + MKOOPKLA + PWAPART + 
# MOPLLAAG + MRELGE + PBRAND + ABYSTAND + MBERBOER + AFIETS + 
# PWALAND + PWAOREG + MGEMLEEF + MINK123M + MINKGEM + ABRAND + 
# AWAOREG + MOPLHOOG + MGODPR + MZPART + MZFONDS + PGEZONG + 
# AGEZONG, data = train_df)

# Suggested 23 independent variables out of 86 variables by forward selection

fwd_reg_model = lm(formula = CARAVAN ~ PPERSAUT + APLEZIER + MKOOPKLA + PWAPART + 
                  MOPLLAAG + MRELGE + PBRAND + ABYSTAND + MBERBOER + AFIETS + 
                  PWALAND + PWAOREG + MGEMLEEF + MINK123M + MINKGEM + ABRAND + 
                  AWAOREG + MOPLHOOG + MGODPR + MZPART + MZFONDS + PGEZONG + 
                  AGEZONG, data = train_df)

pred_fwd = predict(fwd_reg_model, newdata = test_df)

compare = data.frame(actual = test_df$CARAVAN, predicted = pred_fwd)
head(compare,10)

summary(fwd_reg_model)

rss_error_fwd = mean((test_df$CARAVAN - pred_fwd)^2)
rss_error_fwd

rmse_error_fwd = sqrt(mean(test_df$CARAVAN - pred_fwd)^2)
rmse_error_fwd

#####################
# Stepwise Selection: (Combining both forward and backward)
#####################

step(fwd_model, direction = 'both', scope = formula(bwd_model))

# final AIC for stepwise selection: AIC=-17108.56

# lm(formula = CARAVAN ~ PPERSAUT + APLEZIER + MKOOPKLA + PWAPART + 
# MOPLLAAG + MRELGE + PBRAND + ABYSTAND + MBERBOER + AFIETS + 
# PWALAND + PWAOREG + MGEMLEEF + MINK123M + MINKGEM + ABRAND + 
# AWAOREG + MOPLHOOG + MGODPR + MZPART + MZFONDS + PGEZONG + 
# AGEZONG, data = train_df)

# Suggested 23 independent variables out of 86 variables by stepwise selction

# Here, we can see that, although both backward, forward selection and 
# stepwise selection got similar 'AIC' value, forward selection done better
# in reducing the model complexity by reducing no.of variables.


###################
# Ridge regression:
###################

param_control = trainControl(method = 'repeatedcv', number = 10, repeats = 5)

ridge_reg_model_2 = train(CARAVAN ~ ., data = train_df, method = 'glmnet', 
                        trControl = param_control, 
                        tuneGrid = expand.grid(alpha =0, 
                                               lambda = seq(0.01, 100, 
                                                            length = 20)))

ridge_reg_model_2
plot(ridge_reg_model_2)


###################
# Lasso Regression:
###################

param_control = trainControl(method = 'repeatedcv', number = 10, repeats = 5)

lasso_model_2 = train(CARAVAN ~ ., data = train_df, method = 'glmnet', 
                      trControl = param_control, 
                      tuneGrid = expand.grid(alpha = 1, 
                                             lambda = seq(0.01, 100, length = 20)))

lasso_model_2
plot(lasso_model_2)

#############
# Conclusion:
#############

# Forward selection/stepwise selection performed well overall. That is because, 
# apart from the model not being complex with just 23 independent variables, 
# unlike in backward selection with 32 variables or ridge or lasso with around 
# 86 independent variable, the error value is better for forward selection.

################################################################################
# Question 3:
################################################################################

# Load train data

X <- as.matrix(read.table(gzfile("zip.train")))
head(X)
dim(X)

X_7_9 <- which(X[, 1] == 7 | X[, 1] == 9)

X.train <- X[X_7_9, -1]
y.train <- X[X_7_9, 1] == 7

table(y.train)

# Load test data
X <- as.matrix(read.table(gzfile("zip.test")))
head(X)
dim(X)

X_7_9 <- which(X[, 1] == 7 | X[, 1] == 9)

X.test <- X[X_7_9, -1]
y.test <- X[X_7_9, 1] == 7

table(y.test)

# Linear Regression:

L <- lm(y.train ~ X.train)
yhat <- (cbind(1, X.test) %*% L$coef) >= 0.5
L.error <- mean(yhat != y.test)

# KNN:

library(class)
k <- c(1, 3, 5, 7, 9, 11, 13, 15)
k.error <- rep(NA, length(k))
for (i in 1:length(k)) {
    yhat <- knn(X.train, X.test, y.train, k[i])
    k.error[i] <- mean(yhat != y.test)
}

# Lets compare:

error <- matrix(c(L.error, k.error), ncol = 1)
colnames(error) <- c("Error Rate")
rownames(error) <- c("Linear Regression", paste("k-NN with k =", k))
error

plot(c(1, 15), c(0, 1.1 * max(error)), type = "n", main = "SLR vs KNN", 
     ylab = "Error Rate", xlab = "k")
abline(h = 0.04121, col = 2, lty = 3)
points(k, k.error, col = 4)
lines(k, k.error, col = 4, lty = 2)

# Conclusion:

# Here, both linear regression and KNN are performing nearly same with
# __`red line`__ indicating __`SLR`__ and __`blue line`__ indicating 
# __`KNN`__ error values respectively. Both models error rate is close to zero.












