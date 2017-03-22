##########################
### svm advanced model ###
##########################

### Author: Nanjun Wang
### Project 3
### ADS Spring 2017


svm_adv <- function(file_dir){
  
  ###load libraries    
  library(dplyr)
  library(e1071)
  
  ###read feature files
  prediction_inception <- read.csv("/Users/ouminamikun/Desktop/Columbia/Spring 2017/ADS/spr2017-proj3-group3/output/prediction_inceptionV3.csv")
  label <- read.csv("/Users/ouminamikun/Desktop/Columbia/Spring 2017/ADS/Project 3/labels.csv")
  label$V1 <- as.numeric(label$V1)
  dataset <- read.csv(file_dir)
  dataset <- data.frame(t(dataset))
  dataset <- cbind(label,dataset)
  file_rowname <- rownames(dataset)[1]
  value <- "jpg"
  if(grepl(value, file_rowname)){
    dataset$image <- rownames(dataset)
  } else{
    dataset$image <- paste(rownames(dataset),".jpg", sep = "")
  }
  
  ###set test and train data 
  
  test_data <- dataset %>% filter(dataset$image %in% prediction_inception$image)
  train_data <- dataset %>% filter(!dataset$image %in% prediction_inception$image)
  
  ###train model 
  #Cross validaiton---tunning parameters
  tune_par_cnn150 <- cv.function(X.train = train_data[,-c(1,152)], 
                                 y.train = train_data[,1],
                                 c = c(0.001,0.01,0.1),
                                 K = 5)
  
  #prediction error after tunning parameters
  bestfit_cnn150 <- train.fn(dat_train = train_data[,-c(1,152)], 
                             label_train = train_data[,1], 
                             c = tune_par_cnn150$best_cost)
  return(model = bestfit_cnn150)
}