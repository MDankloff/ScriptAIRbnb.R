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
#options(stringsAsFactors = FALSE)

##### LOAD DATA #############
data_compas <- read.csv('german_processed.csv') %>% as_tibble

data_compas %>% glimpse