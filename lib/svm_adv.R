##########################
### svm advanced model ###
##########################

### Author: Nanjun Wang
### Project 3
### ADS Spring 2017
cv.function <- function(X.train, y.train, c, K){
  n <- length(y.train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  error <- data.frame(NA)
  cost <- c
  
  for(i in 1:length(c)){
    for (j in 1:K){
      train.data <- X.train[s != j,]
      train.label <- y.train[s != j]
      test.data <- X.train[s == j,]
      test.label <- y.train[s == j]
      
      fit <- svm(X.train, as.factor(y.train), data = cbind(X.train, y.train),
                 cost = c[i],
                 kernel = "linear",
                 scale = F)
      
      pred <- predict(fit, test.data)  
      cv.error[j] <- mean(pred != test.label)  
    }	
    error[i,1] <- c[i]
    error[i,2] <- mean(cv.error)
    error[i,3] <- sd(cv.error)
    colnames(error) <- c("cost","mean","sd")
  }
  return(list( error, best_cost = error[which.min(error[,2]),1]))
}

train.fn <- function(dat_train, label_train, c=1000){
  
  ### Train a support vector machine (svm) using processed features from training images
  
  ### Input: 
  ###  -  processed features from images 
  ###  -  class labels for training images
  ### Output: training model specification
  
  ### load libraries
  #library(e1071)
  
  ### Train with svm model
  fit_svm <- svm(dat_train, as.factor(label_train), data = cbind(dat_train, label_train),
                 cost = c,
                 kernel = "linear",
                 scale = F,
                 probability = T)
  
  return(fit = fit_svm)
  
}

svm_adv <- function(file_dir,k,run.cv){
  # file_dir: path to features.csv
  # k: number of features
  
  ###load libraries    
  #library(dplyr)
  #library(e1071)
  #library(data.table)
  
  ###read feature files
  #prediction_inception <- read.csv("/output/prediction_inceptionV3.csv")
  test_ids <- fread("output/prediction_inceptionV3.csv")
  label <- fread("data/labels.csv")
  label$V1 <- as.numeric(label$V1)
  dataset <- fread(file_dir)
  dataset <- data.frame(t(dataset))
  #dataset <- cbind(label,dataset)
  dataset$label <- label$V1
  #file_rowname <- rownames(dataset)
  dataset$image <- rownames(dataset)
  # value <- "jpg"
  # if(grepl(value, file_rowname[1])){
  #   dataset$image <- rownames(dataset)
  # } else{
  #   dataset$image <- paste(rownames(dataset),".jpg", sep = "")
  # }
  
  ###set test and train data 
  
  #test_data <- dataset %>% filter(dataset$image %in% test_ids$image)
  train_data <- dataset %>% filter(!dataset$image %in% test_ids$image)
  
  ###train model 
  #Cross validaiton---tunning parameters
  if(run.cv){
    tune_par_cnn150 <- cv.function(X.train = train_data[,1:k], 
                                   y.train = train_data[,'label'],
                                   c = c(0.0001,0.0003,0.001),
                                   K = 5)
    best.cost <- tune_par_cnn150$best_cost
  } else{
    best.cost <- 0.001
  }
  
  
  #prediction error after tunning parameters
  bestfit_cnn150 <- train.fn(dat_train = train_data[,1:k], 
                             label_train = train_data[,'label'], 
                             c = best.cost)
                             
                             
  return(bestfit_cnn150)
}
