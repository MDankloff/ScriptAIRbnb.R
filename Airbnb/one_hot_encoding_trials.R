##### INIT #############

setwd(dirname(rstudioapi::getSourceEditorContext()$path)) 

if(!require('tidyverse')) install.packages('tidyverse') 
library(tidyverse)
if(!require('randomForest')) install.packages('randomForest') 
library (randomForest)
if(!require('cowplot')) install.packages('cowplot') 
library (cowplot)
if(!require('caTools')) install.packages('caTools') 
library(caTools)
if(!require('mltools')) install.packages('mltools') 
library(mltools)
if(!require('data.table')) install.packages('data.table') 
library(data.table)
if(!require('rpart')) install.packages('rpart') 
library(rpart)
if(!require('DALEX')) install.packages('DALEX')
library(DALEX)

options(rstudio.help.showDataPreview = FALSE)
# options(stringsAsFactors = FALSE)

##### LOAD DATA #############
data <- read.csv('new_listings.csv') %>% as_tibble
data %>% glimpse

##### PREPARE DATA #############
data <- data  %>% 
  select(-neighbourhood_group, -latitude, -longitude, -name, -host_name, -host_id) %>%
  drop_na

data <- data %>%
  mutate(last_review = last_review %>% as_date %>% as.integer) 

data_prep <- data %>%
  mutate(fraud_label = fraud_label %>% as.factor) %>%
  mutate(neighbourhood = neighbourhood %>% 
           str_replace_all("[ -]", "_") %>% str_replace_all("___", "_") %>% 
           as.factor) %>%
  mutate(room_type = room_type %>% 
           str_replace_all("[ -/]", "_") %>% str_replace_all("___", "_") %>% 
           as.factor) 

data_prep %>% glimpse

data_1hot <- one_hot(data_prep %>% setDT) %>%
  as_tibble %>%
  mutate(fraud_label = fraud_label_1 %>% as.factor) %>%
  select(-fraud_label_0, -fraud_label_1)

data_1hot %>% glimpse

##### SPLIT DATA - RANDOM #############
set.seed(1234)

split <- sample.split(data_1hot$fraud_label, SplitRatio = 2/3)
trainset_random <- subset(data_1hot, split == TRUE)
testset_random <- subset(data_1hot, split == FALSE)

##### SPLIT DATA - BALANCED #############
set.seed(1234)

trainset_1 <- data_1hot %>% filter(fraud_label == 1) %>% head(1000) 
trainset_0 <- data_1hot %>% filter(fraud_label == 0) %>% head(1000) 

trainset <- rbind(trainset_0, trainset_1)
testset <- data_1hot %>% filter(! id %in% trainset$id)

trainset %>% nrow
testset %>% nrow
data_1hot %>% nrow

##### FIT MODEL ####
set.seed(1234)
rf_model = randomForest(x = trainset %>% select(-fraud_label),
                         y = trainset$fraud_label,
                         ntree = 100, mtry = 10)

tree_model <- rpart(fraud_label ~ ., data = trainset, method = "class")

plot(tree_model)

##### TEST MODEL ####
pred_test <- rf_model %>% 
  predict(newdata = testset)

predictions <-cbind(data.frame(train_preds = pred_test, testset$fraud_label))
predictions %>% glimpse

cm <- caret::confusionMatrix(predictions$train_preds, predictions$testset.fraud_label)
print(cm)

pred_test <- tree_model %>% 
  predict(newdata = testset) %>% 
  as_tibble %>% 
  mutate(score = `0`) %>% 
  select(score)
pred_test %>% glimpse

predictions <-cbind(data.frame(train_preds = ifelse(pred_test$score<0.5, 1, 0) %>% as.factor, 
                               testset$fraud_label))
predictions %>% glimpse

cm <- caret::confusionMatrix(predictions$train_preds, predictions$testset.fraud_label)
print(cm)

#### APPLY SHAP ###
#trainset <- as.numeric(unlist(trainset))
#testset <- as.numeric(unlist(testset))


#Shap_dum <- explain(predictions, data = testset, y= testset$fraud_label, label = "Fraud model")

#plot(Shap_dum)


