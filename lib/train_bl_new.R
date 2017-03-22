library(dplyr)
library(gbm)
setwd("/Users/Bowen/Desktop")
train_bl <- function(trainD,par=NULL){
  # Training function for the baseline model (GBM)
  # INPUT: 
  #     trainD =  data.frame n_images*(n_features + 1*label + 1*image_id)
  #     par = list of values for params depth, shrinkage and n.trees
  #
  # OUTPUT: trained model object
  
  library('gbm')
  
  k <- dim(trainD)[2]-1
  
  if(is.null(par)){
    depth = 1
    shrinkage = 0.03
    n.trees = 250
  }
  else {
    eval(parse(text = paste(names(par), par, sep = '=', collapse = ';')))
  }
  #trainD$label <- trainD$label
  t1 <- Sys.time()
  gbm_fit <- gbm(label~ ., data = trainD[,1:k],distribution = "bernoulli",interaction.depth = depth, n.trees = n.trees, shrinkage = shrinkage)
  t2 <- Sys.time() 
  t2 - t1
  
  return(gbm_fit)
  
}

features <- read.csv("sift_features_1000.csv", header = TRUE, as.is=TRUE)
labels <- read.csv("labels.csv", header = TRUE, as.is=TRUE)
test_v3 <- read.csv("prediction_inceptionV3.csv", header = TRUE, as.is=TRUE)
features <- t(features)
features <- data.frame(features)
colnames(labels) <- "label"
dataset <- cbind(features, labels)
n <- ncol(dataset)
rownames(dataset) <- paste(rownames(dataset),".jpg",sep="")
dataset$image <- rownames(dataset)
dim(dataset)
#Set test data and training data
test_data <- dataset %>% filter(dataset$image %in% test_v3$image)

train_data <- dataset %>% filter(! dataset$image %in% test_v3$image)


gbm_fit<-train_bl(trainD = train_data)
# Get the estimation of n.trees
gbm.perf(gbm_fit)

pred<-predict(gbm_fit,test_data[,1:n-1],n.trees=250)>0
# Prediction accuracy
mean(test_data$label==pred)

# Output csv
friedChicken<-predict(gbm_fit,test_data[,1:n-1],n.trees=100,type="response")
labradoodle<-1-friedChicken
prediction_gbm_sift_1000<-cbind(image=test_data[n+1],friedChicken=friedChicken,labradoodle=labradoodle)
write.csv(prediction_gbm_sift_1000,file="prediction_gbm_sift_1000.csv",row.names = FALSE)



