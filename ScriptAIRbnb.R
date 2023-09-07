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

#options(stringsAsFactors = FALSE)

##### LOAD DATA #############
data_orig <- read.csv('listings.csv') %>% as_tibble
#data_orig <- data_orig %>% 
#  mutate(last_review = last_review %>% as_date()) 

data_orig %>% glimpse

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
glimpse(new_label)

################# ADD BIAS IN FRAUD LABELS for neigbourhood ################
### Separate subset of the data
hood <- "Oostelijk Havengebied - Indische Buurt"

data_in_hood <- data_w_fraud %>% filter(neighbourhood == hood) 
data_not_in_hood <- data_w_fraud %>% filter(neighbourhood != hood) 

### MAKE FRAUD LABELS FOR ONE NEIGHBOURHOOD ####
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

#### CONVERTING LAST REVIEW TO NUMERIC  ####
data_w_bias <- data_w_bias %>%
  mutate(last_review = as.numeric(as_date(last_review))) 

data_w_bias %>% glimpse


############### SPLIT DATA SET INTO TRAIN AND TEST SET ########
set.seed(1234)

split <- sample.split(data_w_bias$fraud_label, SplitRatio = 0.67)
trainset <- subset(data_w_bias, split == TRUE)
testset <- subset(data_w_bias, split == FALSE)

trainset %>% glimpse

###Separate Fraud_label/ fraud_label_text columns from training and test set and remove variables that are not needed####
trainset_X <- trainset %>%  select(-id, -name, -host_id, -host_name, -latitude, -longitude, 
                                         -fraud_label, -fraud_label_text)
testset_X <- testset %>%  select(-id, -name, -host_id, -host_name, -latitude, -longitude, 
                                         -fraud_label, -fraud_label_text)

#### change character variables to factors
trainset_X <- trainset_X %>% mutate(neighbourhood = as.factor(neighbourhood), room_type=as.factor(room_type))
testset_X <- testset_X %>% mutate(neighbourhood = as.factor(neighbourhood), room_type=as.factor(room_type))

trainset_X %>% glimpse
testset_X %>% glimpse

###Make fraud_label/fraud_label_text columns as separate dataset columnns ##
trainset_Y <- trainset %>% select(fraud_label)
testset_Y <- testset %>% select(fraud_label)

trainset_Y %>% glimpse
testset_Y %>% glimpse

############# FITTING RANDOM FOREST REGRESSION TO DATA SET ############
set.seed(123)
#modelfraud = randomForest(unlist(fraud_label) ~ ., data = trainset_X %>% mutate(fraud_label=trainset_Y),
                         #ntree = 500, mtry=1, classwt=c(10000,1))


modelfraud = randomForest(unlist(trainset_Y) ~ ., data = trainset_X,
                          ntree = 5, mtry=1, classwt=c(10000,1))

modelfraud


####Check optimal number of trees through error ####
oob.error.data <-data.frame(Trees=rep(1:nrow(modelfraud$err.rate), times = 3),
                            Type=rep(c("OOB", "0", "1"), each=nrow(modelfraud$err.rate)),
                            Error=c(modelfraud$err.rate[,"OOB"], 
                                    modelfraud$err.rate[,"0"],
                                    modelfraud$err.rate[,"1"]))
ggplot(data= oob.error.data, aes(x=Trees, y=Error)) +
  geom_line(aes(color=Type))

####Check for optimal no of variables at each split####
oob.values <- vector(length = 10)
for (i in 1:10) {
  temp_modelfraud <- randomForest(unlist(trainset_Y) ~ ., data = trainset_X, mtry = i, ntree = 5)
  oob.values[i] <- temp_modelfraud$err.rate[nrow(temp_modelfraud$err.rate),1]
}
oob.values

####Feature scaling ####
#head(scale(trainset_X[3]))

############# MAKE PREDICTIONS ON TEST DATA  #################

pred_train <- predict(modelfraud, data = trainset_X, type = "class")
#
pred_test <- predict(modelfraud, newdata = testset_X, type = "class")
#
fraud_pred <-cbind(data.frame(pred_train = pred_test, testset_Y))

#fraud_pred <- predict(modelfraud, newdata = testset_X, type= "response")
fraud_pred

##create confusion matrix object
cm <- caret::confusionMatrix(pred_test, testset_Y$fraud_label)
cm

tble = table(pred_test, testset_Y$fraud_label)
tble

####### APPLY SHAP #########
## change this back to numeric otherwise shap doesnt work
trainset_Y <- as.numeric(unlist(trainset_Y))
testset_Y<- as.numeric(unlist(testset_Y))

exp_shap <- explain(fraud_pred, data = testset_X, y= testset_Y, label = "Regressor model")
exp_shap

fi_shap <- model_parts(exp_shap, B= 10, loss_function = loss_one_minus_auc)
plot(fi_shap)

################### ONE HOT ENCODING FOR CATEGORICAL VARIABLES room_type & neighbourhood ##############
dttrain <- data.table(trainset_X)
dttest <- data.table(testset_X)

one_hot(dttrain, cols = "room_type")
one_hot(dttrain, cols = "neighbourhood")
trainset_Xdummy <- one_hot(dttrain, c("room_type", "neighbourhood"))


one_hot(dttest, cols = "room_type")
one_hot(dttest, cols = "neighbourhood")
testset_Xdummy <- one_hot(dttest, c("room_type", "neighbourhood"))

#### SEE IF PREDICTIONS GOT BETTER WITH ONE HOT ENDCODING####
#
set.seed(123)

#modelfraud_dum = randomForest(unlist(trainset_Y) ~ ., data = trainset_Xdummy,
                          #ntree = 5, mtry=1, classwt=c(10000,1))
modelfraud_dum = randomForest(unlist(fraud_label) ~ ., data = trainset_Xdummy %>% mutate(fraud_label=trainset_Y),
ntree = 500, mtry=1,classwt=c(10000,1))

modelfraud_dum
#
pred_train_dum <- predict(modelfraud_dum, data = trainset_Xdummy, type = "class")
pred_test_dum <- predict(modelfraud_dum, newdata = testset_Xdummy, type = "class")
fraud_pred_dum <-cbind(data.frame(pred_train_dum = pred_test_dum, testset_Y))

fraud_pred_dum

cmdum <- caret::confusionMatrix(pred_test_dum, testset_Y$fraud_label)
cmdum

tbledum = table(pred_test_dum, testset_Y$fraud_label)
tbledum

## APPLY SHAP 
## change this back to numeric otherwise shap doesnt work
trainset_Y <- as.numeric(unlist(trainset_Y))
testset_Y<- as.numeric(unlist(testset_Y))

exp_shap_dum <- explain(fraud_pred_dum, data = testset_Xdummy, y= testset_Y, label = "Fraud model")
exp_shap_dum

#fi_shap <- model_parts(exp_shap, B= 10, loss_function = loss_one_minus_auc)
#plot(fi_shap)


