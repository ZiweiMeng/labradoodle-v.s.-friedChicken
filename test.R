######################################################
### Fit the classification model with testing data ###
######################################################
# Author: Ziwei Meng
# date: 2017-03-22

split.train <- function(fpath){
  test_ids <- fread("output/prediction_inceptionV3.csv")
  label <- fread("data/labels.csv")
  label$V1 <- as.numeric(label$V1)
  dataset <- fread(fpath)
  dataset <- data.frame(t(dataset))
  dataset$label <- label$V1
  file_rowname <- rownames(dataset)
  value <- "jpg"
  if(grepl(value, file_rowname[1])){
    dataset$image <- rownames(dataset)
  } else{
    dataset$image <- paste(rownames(dataset),".jpg", sep = "")
  }
  test_data <- dataset %>% filter(dataset$image %in% test_ids$image)
  
  return(test_data)
}

process.test <- function(fpath){
  test.data <- fread(fpath)
  test.data <- data.frame(t(test.data))
  test.data$image <- rownames(test.data)
  return(test.data)
}


test = function(fit_train, dat_test,k){
  # Fit the classfication model with testing data
  # INPUT: 
  #     fit_train = trained model object, either gbm or xgb.Booster
  #     dat_test = processed features from testing images 
  #
  # OUTPUT: training model specification
  
  #library('gbm')
  #library('xgboost')
  
  #k <- dim(dat_test)[2]-2
  
  pred = switch(class(fit_train), 
                gbm = predict(fit_train, 
                              newdata = dat_test[,1:k+1], 
                              n.trees = fit_train$n.trees, 
                              type="response"),
                
                # xgb.Booster = predict(fit_train, 
                #                       newdata = dat_test)
                svm = attr(predict(fit_train, dat_test[,1:k], probability = T),'probabilities')[,2]
  )
  
  
  return(as.numeric(pred> 0.5))
}
