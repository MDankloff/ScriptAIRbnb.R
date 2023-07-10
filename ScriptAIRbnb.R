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

#options(stringsAsFactors = FALSE)

if(!require('caret')) install.packages('caret')
library(caret)

##### LOAD DATA #############
data_orig <- read.csv('listings.csv') %>% as_tibble
#data_orig <- data_orig %>% 
#  mutate(last_review = last_review %>% as_date()) 

data_orig %>% glimpse

##### ADD FRAUD LABELS #############

fraud_rate <- 0.1
n_case <- data_orig %>% nrow

fraud <- rep(1, (n_case*fraud_rate) %>% round)
nonfraud <- rep(0, n_case - (fraud %>% length) ) 
all_label <- c(fraud, nonfraud) %>% sample

all_label %>% length

data_w_fraud <- data_orig %>%
  mutate(fraud_label = all_label)

data_w_fraud %>% glimpse
# glimpse(new_label)

##### ADD BIAS IN FRAUD LABELS for neigbourhood #############

### Separate subset of the data
hood <- "Oostelijk Havengebied - Indische Buurt"

data_in_hood <- data_w_fraud %>% filter(neighbourhood == hood) 
data_not_in_hood <- data_w_fraud %>% filter(neighbourhood != hood) 

### Make new fraud labels for one neighborhood
fraud_rate_hood <- 0.4
n_case <- data_in_hood %>% nrow

fraud <- rep(1, (n_case*fraud_rate) %>% round)
nonfraud <- rep(0, n_case - (fraud %>% length) ) 
new_label <- c(fraud, nonfraud) %>% sample

data_in_hood <- data_in_hood %>%
  mutate(fraud_label = new_label)


### Merge modified data subset with the unmodified one.

data_w_bias <- bind_rows(data_in_hood, data_not_in_hood)
data_w_bias %>% nrow

##### EXPORT NEW DATA #############

data_w_bias %>% write.csv('new_listings.csv', row.names = FALSE)

#### REMOVE NA PREDICTORS ######
data_w_bias = subset(data_w_bias, select = -c(neighbourhood_group))
data_w_bias <- data_w_bias %>% drop_na

#### CONVERTING OUTCOME TO FACTOR (otherwise random forest doesn't work) & assign text to outcome  ####
data_w_bias <- data_w_bias %>%
  mutate(fraud_label_text = ifelse(fraud_label == 0, "Not_Fraud", "Fraud") )

data_w_bias$fraud_label <- as.factor(data_w_bias$fraud_label)
glimpse(data_w_bias)

####Split data set into train and test set ###
set.seed(1234)

split <- sample.split(data_w_bias$fraud_label, SplitRatio = 0.67)
trainset <- subset(data_w_bias, split == TRUE)
testset <- subset(data_w_bias, split == FALSE)

#trainset$fraud_label <- as.factor(train$fraud_label)
#testset$fraud_label <- as.factor(test$fraud_label)

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

predictions <-cbind(data.frame(train_preds= pred_test, testset$fraud_label))

##create confusion matrix object

cm <- caret::confusionMatrix(predictions$train_preds, predictions$testset.fraud_label)
print(cm)


