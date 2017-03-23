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
  
  bstSparse <- xgb.train(data = dtrain, max.depth = max.depth, eta = eta, 
                         nthread = nthread, nround = nround, objective = objective)
  return(bstSparse)
}

## The part below is used for test the function, not included in the function
features <- read.csv("sift_features_1100.csv", header = TRUE, as.is=TRUE)
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
test_data1<-test_data[,1:(n_col-2)]
test_data1<-as.matrix(test_data1)
### Start running the training function
t1<-Sys.time()
xgb_fit<-runXGB(train_X_path = "sift_features_1100.csv",train_Y_path = "labels.csv")
t2<-Sys.time()
### End running the training function


### Start predicting
t3<-Sys.time()
pred<-predict(xgb_fit,test_data1)>0.5
t4<-Sys.time()
### End training 

# Prediction accuracy
accuracy<-mean(test_label==pred)

train_time<-t2-t1
pred_time<-t4-t3

accuracy
train_time
pred_time

friedChicken<-predict(xgb_fit,test_data1,n.trees=100,type="response")
labradoodle<-1-friedChicken
prediction_csv<-cbind(image=test_data[,n+1],friedChicken=friedChicken,labradoodle=labradoodle)
write.csv(prediction_csv,file="prediction_xgb_sift_1100.csv",row.names = FALSE)

