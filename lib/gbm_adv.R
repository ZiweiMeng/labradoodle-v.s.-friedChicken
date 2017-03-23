set.seed(400)
setwd("~/Documents/2017Spring/ADS/Proj3")

## This is the gbm function
gbm_adv <- function(fpath,par=NULL){
  # Training function for the baseline model (GBM)
  # INPUT: 
  #     trainD =  data.frame n_images*(n_features + 1*label + 1*image_id)
  #     par = list of values for params depth, shrinkage and n.trees
  #
  # OUTPUT: trained model object
  # read the data
  features <- read.csv(fpath, header = TRUE, as.is=TRUE)
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
  train_data <- dataset %>% filter(! dataset$image %in% test_v3$image)
  
  k <- dim(train_data)[2]-1
  
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
  gbm_fit <- gbm(label~ ., data = train_data[,1:k],distribution = "bernoulli",interaction.depth = depth, n.trees = n.trees, shrinkage = shrinkage)
  t2 <- Sys.time() 
  t2 - t1
  
  return(gbm_fit)
}

## The part below is used for test the function, not included in the function
features <- read.csv("cnn_features_150.csv", header = TRUE, as.is=TRUE)
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

gbm_fit<-gbm_adv("cnn_features_150.csv")
# Get the estimation of n.trees
gbm.perf(gbm_fit)
pred<-predict(gbm_fit,test_data[,1:n-1],n.trees=250)>0
head(predict(gbm_fit,test_data[,1:n-1],n.trees=250)>0)
# Prediction accuracy
mean(test_data$label==pred)
