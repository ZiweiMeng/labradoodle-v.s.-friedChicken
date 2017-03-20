#########################################################
### Train a classification model with training images ###
#########################################################

### Author: Nanjun Wang
### Project 3
### ADS Spring 2017


train.fn <- function(dat_train, label_train, c=1000){
  
  ### Train a support vector machine (svm) using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  library(e1071)
  
  ### Train with svm model
  fit_svm <- svm(dat_train, as.factor(label_train), data = cbind(dat_train, label_train),
                 cost = c,
                 kernel = "linear",
                 scale = F,
                 probability = T)
  
  return(fit = fit_svm)

}