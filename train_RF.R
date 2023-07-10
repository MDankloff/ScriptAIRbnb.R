##### INIT #############

setwd(dirname(rstudioapi::getSourceEditorContext()$path)) 

if(!require('tidyverse')) install.packages('tidyverse') 
library(tidyverse)
if(!require('randomForest')) install.packages('randomForest') 
library (randomForest)
if(!require('ggplot2')) install.packages('ggplot2') 
library (ggplot2)
if(!require('cowplot')) install.packages('cowplot') 
library (cowplot)
if(!require('caTools')) install.packages('caTools') 
library(caTools)

# options(stringsAsFactors = FALSE)

##### LOAD DATA #############
data <- read.csv('new_listings.csv') %>% as_tibble
#data_orig <- data_orig %>% 
#  mutate(last_review = last_review %>% as_date()) 

data %>% glimpse

data_w_bias$fraud_label <- as.factor(data_w_bias$fraud_label)

####Split data set into train and test set ###
set.seed(1234)

split <- sample.split(data_w_bias$fraud_label, SplitRatio = 2/3)
trainset <- subset(data_w_bias, split == TRUE)
testset <- subset(data_w_bias, split == FALSE)

trainset$fraud_label <- as.factor(train$fraud_label)
testset$fraud_label <- as.factor(test$fraud_label)

##### FITTING RANDOM FOREST REGRESSION TO DATA SET ####
set.seed(1234)
regressor = randomForest(x = trainset[1:14],
                         y = trainset$fraud_label,
                         ntree = 100)

####Check optimal number of variables at each internal node in the tree####
oob.values <- vector(length=10)
for (i in 1:10) {
  temp.regressor <- randomForest(fraud_label ~ ., data=trainset, mtry=i, ntree = 10)
  oob.values[i] <- temp.regressor$err.rate[nrow(temp.regressor$err.rate),1] 
}
oob.values


##make predictions on TEST data
pred_test <- predict(regressor, newdata = testset, type = "class")

pred_test




