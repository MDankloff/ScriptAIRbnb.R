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
if(!require('DALEX')) install.packages('DALEX')
library(DALEX)
if(!require('data.table')) install.packages('data.table')
library(data.table)
if(!require('caret')) install.packages('caret')
library(caret)
if(!require('mltools')) install.packages('mltools')
library(mltools)
if(!require('shapper')) install.packages('shapper')
library(shapper)
if(!require('lime')) install.packages('lime')
library(lime)

#options(stringsAsFactors = FALSE)

##### LOAD DATA #############
data_orig <- read.csv('listings.csv') %>% as_tibble
#data_orig <- data_orig %>% 
#  mutate(last_review = last_review %>% as_date()) 

#data_orig %>% glimpse
#-------------------------------------------------------------------------------
################ ADD FRAUD LABELS ####################
fraud_rate <- 0.1
n_case <- data_orig %>% nrow

fraud <- rep(1, (n_case*fraud_rate) %>% round)
nonfraud <- rep(0, n_case - (fraud %>% length) )
all_label <- c(fraud, nonfraud) %>% sample

all_label %>% length

data_w_fraud <- data_orig %>%
  mutate(fraud_label = all_label)

data_w_fraud %>% glimpse
#-------------------------------------------------------------------------------
################# ADD PRIOR IN FRAUD LABELS FOR NEIGHBOURHOOD ################
### Separate subset of the data
hood <- "Oostelijk Havengebied - Indische Buurt"

data_in_hood <- data_w_fraud %>% filter(neighbourhood == hood)
data_not_in_hood <- data_w_fraud %>% filter(neighbourhood != hood)

### Make fraud labels for hood ####
fraud_rate_hood <- 0.4
#fraud_rate_hood <- 1
n_case <- data_in_hood %>% nrow

fraud <- rep(1, (n_case*fraud_rate_hood) %>% round)
nonfraud <- rep(0, n_case - (fraud %>% length) )
new_label <- c(fraud, nonfraud) %>% sample

data_in_hood <- data_in_hood %>%
  mutate(fraud_label = new_label)

### Merge modified data subset with the unmodified data subset

data_w_bias <- bind_rows(data_in_hood, data_not_in_hood)
data_w_bias %>% nrow
#-------------------------------------------------------------------------------
##### EXPORT NEW DATA #############

data_w_bias %>% write.csv('output/new_listings.csv', row.names = FALSE)

#### REMOVE NA PREDICTORS ######
data_w_bias = subset(data_w_bias, select = -c(neighbourhood_group))
data_w_bias <- data_w_bias %>% drop_na

#### CONVERTING OUTCOME TO FACTOR (otherwise random forest doesn't work)  ####
data_w_bias$fraud_label <- as.factor(data_w_bias$fraud_label)
glimpse(data_w_bias)

#### CONVERTING LAST REVIEW TO NUMERIC  ####
data_w_bias <- data_w_bias %>%
  mutate(last_review = as.numeric(as_date(last_review))) 
#data_w_bias %>% glimpse

####REMOVE VARIABLES NOT NEEDED ######
data_w_bias <- data_w_bias %>% select (-id, -name, -host_id, -host_name, -latitude, -longitude)
#data_w_bias %>% glimpse

####### CHANGE CHARACTERS TO FACTORS ####
data_w_bias <- data_w_bias %>% mutate(neighbourhood = as.factor(neighbourhood), room_type = as.factor(room_type))
#data_w_bias %>% glimpse
#-------------------------------------------------------------------------------
############### ONE HOT ENCODING room_type & neighbourhood ##############

dum_data <- one_hot(data_w_bias %>% setDT) %>% as_tibble %>%
  mutate(fraud_label = fraud_label_1 %>% as.factor) %>%
  select(-fraud_label_0, -fraud_label_1)

#dum_data <- cbind(dum_data1, dum_data2, data_w_bias)
#dum_data <- dum_data %>% select (-neighbourhood, -room_type) 
#dum_data %>% glimpse 

###change Doubles into factors##
#double_columns <- sapply(dum_data, is.double)
#dum_data[double_columns] <- lapply(dum_data[double_columns], as.factor)

#######CHANGE NAMES - TO . SO RANDOM FOREST RECOGNIZES OBJECTS #####
names(dum_data) <- make.names(names(dum_data))
#-------------------------------------------------------------------------------

############### SPLIT DATA SET INTO TRAIN AND TEST SET -- separate columns for FRAUD ########
set.seed(1234)

split <- sample.split(dum_data$fraud_label, SplitRatio = 0.67)
trainset <- subset(dum_data, split == TRUE)
testset <- subset(dum_data, split == FALSE)
#testset %>% glimpse
#trainset %>% glimpse

#### CHANGE TO FACTORS
#trainset <- trainset %>% mutate(neighbourhood = as.factor(neighbourhood), room_type=as.factor(room_type))
#testset <- testset %>% mutate(neighbourhood = as.factor(neighbourhood), room_type=as.factor(room_type))

#-------------------------------------------------------------------------------
############# FITTING RANDOM FOREST CLASSIFICATION MODEL TO DATA SET ############
set.seed(123)

modelfraud <- randomForest(x = trainset %>% select(-fraud_label),
                        y = trainset$fraud_label,
                        ntree = 25, mtry = 10)

#plot(modelfraud)
#-------------------------------------------------------------------------------
# #####Check optimal number of trees through error #####
# oob.error.data <-data.frame(Trees=rep(1:nrow(modelfraud$err.rate), times = 3),
#                             Type=rep(c("OOB", "0", "1"), each=nrow(modelfraud$err.rate)),
#                             Error=c(modelfraud$err.rate[,"OOB"], 
#                                     modelfraud$err.rate[,"0"],
#                                     modelfraud$err.rate[,"1"]))
# ggplot(data= oob.error.data, aes(x=Trees, y=Error)) +
#   geom_line(aes(color=Type))
# 
# #####Check for optimal no of variables at each split####
# oob.values <- vector(length = 10)
# for (i in 1:10) {
#   temp_modelfraud <- randomForest((trainset$fraud_label) ~ ., data = trainset, mtry = i, ntree = 5)
#   oob.values[i] <- temp_modelfraud$err.rate[nrow(temp_modelfraud$err.rate),1]
# }
# oob.values
#-------------------------------------------------------------------------------

############# MAKE PREDICTIONS ON TEST DATA  #################

pred_test <- predict(modelfraud, newdata = testset, type = "class")
#pred_test

predictions <-cbind(data.frame(pred_train = pred_test, testset$fraud_label))
#predictions %>% glimpse

#-------------------------------------------------------------------------------
####Confusion matrix ####
cm <- caret::confusionMatrix(predictions$pred_train, testset$fraud_label)

print(cm)
#-------------------------------------------------------------------------------
####### APPLY SHAP and LIME #########
# #take a smaller sample otherwise loading takes long
small_sample <- slice_sample(trainset, n = 100)

# #convert data to a data frame
small_sample_df <- as.data.frame(small_sample)
 
# ## change Y to numeric otherwise shap doesnt work 
y_numeric <- as.numeric(small_sample$fraud_label)

# ########create SHAP EXPLAINER ###########
# exp_rf <- explain(modelfraud, data = small_sample_df, y= y_numeric, label = "Classification model")

# #take one individual prediction from testset
df_indiv <- data.frame(testset[3,])

df_indiv$fraud_label <- as.numeric(df_indiv$fraud_label)

df_indiv <- df_indiv %>% select(-fraud_label)

# 
# #EXPLAIN AGGREGATED SHAP 
# ive_rf <- shap_aggregated(exp_rf, new_observation = df_indiv)
# plot(ive_rf)
# 
# #EXPLAIN INSTANCE LV SHAP  
# bd_rf <- predict_parts(exp_rf, new_observation = df_indiv)
# plot(bd_rf)

########create LIME EXPLAINER #########
#LIME not applicable for RANDOM FOREST - convert it to PREDICT FUNCTION
#uses the predict method of random forest model to generate predictions
 
rf_predict <- function(trainset) {
   return(predict(modelfraud, trainset))
}

#attempt 1 DALEXtra package: lime_rf <- DALEXtra::predict_surrogate_lime(rf_predict, new_observation = df_indiv)
#attempt 1 Lime package: lime_explainer <- lime::lime(x = small_sample_df, model = rf_predict, model_type = "classification", feature_names = colnames(trainset))

## EXPLAIN INSTANCE LV LIME 
#lime_result <- DALEXtra::predict_surrogate_lime(rf_predict, new_observation = df_indiv)
lime_result <- DALEXtra::predict_surrogate_lime(modelfraud, new_observation = df_indiv)
# 
plot_explanations(lime_result)
plot(lime_result)
