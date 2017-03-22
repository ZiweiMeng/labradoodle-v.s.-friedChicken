logisticRegression_adv<-function(fpath){
  # library the LASSO package
  # library the filter package
  if(!require("glmnet")) install.packages('glmnet')
  library("glmnet")
  library(dplyr)
  
  # finding the fixed test index
  sift_features<-read.csv("/Users/jinruxue/Documents/ADS/spr2017-proj3-group3/output/sift_features.csv",header = TRUE)
  sift_features<-data.frame(t(sift_features))
  n<-dim(sift_features)[1]
  sift_features$image<-seq(1:n)
  
  test_data<-read.csv("/Users/jinruxue/Documents/ADS/spr2017-proj3-group3/output/prediction_inceptionV3.csv")
  test_data<-data.frame(test_data)
  test_data<-sift_features%>%filter(paste(rownames(sift_features),".jpg",sep="")%in%test_data$image)
  test_index<- test_data$image
  
  # Read cnn150 csv file
  sift_features<-read.csv(fpath,header = TRUE)
  sift_features<-data.frame(t(sift_features))
  class<-read.csv("/Users/jinruxue/Documents/ADS/spr2017-proj3-group3/output/labels.csv")
  class<-unlist(class)
  sift_features$Class<-factor(class)
  
  # Separete trainning data and test data from orignial data set:
  train_data<-sift_features[-test_index,]
  test_data<-sift_features[test_index,]
  n<-dim(train_data)[2]-1
  # start trainning logistic regression model:
  logictRegression_cnn_featureSelect<-cv.glmnet(as.matrix(train_data[,1:n]),train_data$Class,alpha = 1,family = "binomial",type.measure='auc')
  
  return(logictRegression_cnn_featureSelect)
}

