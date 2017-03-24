# k number of predictors in logistic regression
lr_adv<-function(fpath,k,run.cv){
  # library the LASSO package
  # library the filter package
  #if(!require("glmnet")) install.packages('glmnet')
  
  # finding the fixed test index
  test_data<-fread("output/prediction_inceptionV3.csv")
  #test_data<-data.frame(test_data)
  #test_data<-sift_features%>%filter(paste(rownames(sift_features),".jpg",sep="")%in%test_data$image)
  test_index<- test_data$image
  
  # Read cnn150 csv file
  features<-fread(fpath)
  features<-data.frame(t(features))
  labels<-fread("data/labels.csv")
  #class<-unlist(class)
  features$label<-as.factor(labels$V1)
  features$image <- rownames(features)
  
  # Separete trainning data and test data from orignial data set:
  train_data<-features%>%filter(!image%in%test_index)
  # test_data<-sift_features[test_index,]
  
  # start trainning logistic regression model:
  logictRegression_cnn_featureSelect<-cv.glmnet(as.matrix(train_data[,1:k]),train_data$label,alpha = 1,family = "binomial",type.measure='auc')
  
  return(logictRegression_cnn_featureSelect)
}



