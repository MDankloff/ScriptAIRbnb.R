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

options(rstudio.help.showDataPreview = FALSE)
#options(stringsAsFactors = FALSE)

##### LOAD DATA #############
data_orig <- read.csv('listings.csv') %>% as_tibble
#data_orig <- data_orig %>% 
#  mutate(last_review = last_review %>% as_date()) 

data <- data_orig

#### PREPARE DATA ######
data <- data  %>% 
  select(-neighbourhood_group, -latitude, -longitude, -name, -host_name, -host_id) %>%
  drop_na

data <- data %>%
  mutate(last_review = last_review %>% as_date %>% as.integer)  %>%
  mutate(neighbourhood = neighbourhood %>% 
           str_replace_all("[ -]", "_") %>% str_replace_all("___", "_") %>% 
           as.factor) %>%
  mutate(room_type = room_type %>% 
           str_replace_all("[ -/]", "_") %>% str_replace_all("___", "_") %>% 
           as.factor) 

data %>% glimpse

##### LABELLING PREP #############
# Room & Hood (categories)
hi_room <- data$room_type %>% fct_unique %>% sort
hi_room <- hi_room[1]

hi_hood <- data$neighbourhood %>% fct_unique %>% sort
hi_hood <- hi_hood[c(4, 2, 19, 13)]

# Price & Review (percentiles)
p_price <- ecdf(data$price)
p_review <- ecdf(data$reviews_per_month)

##### LABELLING FUNCTION #############
get_label <- function(price, review, room, hood){
  p_fraud <- mean(p_price(price), p_review(review))
  if(room %in% hi_room) { p_fraud <- mean(p_fraud, 0.9) }
  if(hood %in% hi_hood) { p_fraud <- mean(p_fraud, 0.8) }
  # if(room %in% hi_room) { p_fraud <- min(p_fraud + 0.4, 0.95) }
  # if(hood %in% hi_hood) { p_fraud <- min(p_fraud + 0.4, 0.95) }
  p_fraud <- min(p_fraud, 0.95)
  p_fraud <- max(p_fraud, 0)
  rbinom(1, 1, p_fraud) %>% return
}

data <- data %>% 
  rowwise %>%
  mutate(fraud_label = get_label(price, number_of_reviews, room_type, neighbourhood)) %>%
  mutate(fraud_label = factor(fraud_label))

data$fraud_label %>% summary
# data %>% glimpse

##### EXPORT NEW DATA #############
data %>% write.csv('data_synth_v2.csv', row.names = FALSE)

##### 1-HOT ENCODING #############
data_1hot <- one_hot(data %>% setDT) %>%
  as_tibble %>%
  mutate(fraud_label = fraud_label_1 %>% as.factor) %>%
  select(-fraud_label_0, -fraud_label_1)

# data_1hot %>% glimpse

##### SPLIT DATA - BALANCED #############
set.seed(1234)

trainset_1 <- data_1hot %>% filter(fraud_label == 1) %>% head(2000) 
trainset_0 <- data_1hot %>% filter(fraud_label == 0) %>% head(2000) 

trainset <- rbind(trainset_0, trainset_1)
testset <- data_1hot %>% filter(! id %in% trainset$id)

trainset %>% nrow
testset %>% nrow
data %>% nrow

##### FIT MODEL ####
set.seed(1234)
rf_model = randomForest(x = trainset %>% select(-fraud_label),
                        y = trainset$fraud_label,
                        ntree = 5, mtry = 6)

tree_model <- rpart(fraud_label ~ ., data = trainset, method = "class")
# plot(tree_model)

##### TEST MODEL ####
pred_test <- rf_model %>% predict(newdata = testset)

predictions <-cbind(data.frame(train_preds = pred_test, testset$fraud_label))
# predictions %>% glimpse

cm <- caret::confusionMatrix(predictions$train_preds, predictions$testset.fraud_label)
print(cm)


pred_test <- tree_model %>% 
  predict(newdata = testset) %>% 
  as_tibble %>% 
  mutate(score = `0`) %>% 
  select(score)
# pred_test %>% glimpse

predictions <-cbind(data.frame(train_preds = ifelse(pred_test$score<0.5, 1, 0) %>% as.factor, 
                               testset$fraud_label))
# predictions %>% glimpse

cm <- caret::confusionMatrix(predictions$train_preds, predictions$testset.fraud_label)
print(cm)





##### SHAP ####
exp_shap <- explain(regressor, data = testset, y= testset$fraud_label, label = "Regressor model")
exp_shap

fi_shap <- model_parts(exp_shap, B= 10, loss_function = loss_one_minus_auc)
plot(fi_shap)


