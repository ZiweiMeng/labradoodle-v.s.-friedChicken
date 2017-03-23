set.seed(400)
setwd("~/Documents/2017Spring/ADS/Proj3")

runXGB<-function(train_X_path, train_Y_path, max.depth = 100, eta = 1, 
                 nthread = 3, nround = 1000, objective = "binary:logistic"){
  library(xgboost)
  library(dplyr)
  features <- read.csv(train_X_path, header = TRUE, as.is=TRUE)
  labels <- read.csv(train_Y_path, header = TRUE, as.is=TRUE)[[1]]
  test_v3 <- read.csv("prediction_inceptionV3.csv", header = TRUE, as.is=TRUE)
  features <- t(features)
  features <- data.frame(features)
  dataset <- cbind(features, labels)
  dataset$image <- rownames(features)
  train_set <- dataset %>% filter(! dataset$image %in% test_v3$image)
  
  k <- dim(train_set)[2]
  colnames(train_set)[k-1]<-"label"
  
  train_data<- as.matrix(train_set[,1:(k-2)])
  train_label<- as.numeric(train_set$label)
  dtrain <- xgb.DMatrix(data = train_data, label = train_label)
  
  bstSparse <- xgb.train(eval,data = dtrain, max.depth = max.depth, eta = eta, 
                         nthread = nthread, nround = nround, objective = objective)
  return(bstSparse)
}

## The part below is used for test the function, not included in the function
features <- read.csv("cnn_features_350.csv", header = TRUE, as.is=TRUE)
labels <- read.csv("labels.csv", header = TRUE, as.is=TRUE)
test_v3 <- read.csv("prediction_inceptionV3.csv", header = TRUE, as.is=TRUE)
features <- t(features)
features <- data.frame(features)
colnames(labels) <- "label"
dataset <- cbind(features, labels)
n <- ncol(dataset)
#rownames(dataset) <- paste(rownames(dataset),".jpg",sep="")
dataset$image <- rownames(dataset)
#Set test data and training data 
test_data <- dataset %>% filter(dataset$image %in% test_v3$image)
n_col<-ncol(test_data)
test_label<-test_data[,n_col-1]
test_data<-test_data[,1:(n_col-2)]
test_data<-as.matrix(test_data)
### Start running the training function
t1<-Sys.time()
xgb_fit<-runXGB(train_X_path = "cnn_features_350.csv",train_Y_path = "labels.csv")
t2<-Sys.time()
### End running the training function


### Start predicting
t3<-Sys.time()
pred<-predict(xgb_fit,test_data)>0.5
t4<-Sys.time()
### End training 

# Prediction accuracy
accuracy<-mean(test_label==pred)

train_time<-t2-t1
pred_time<-t4-t3

accuracy
train_time
pred_time


