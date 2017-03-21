# Random Forest Function

rf_adv <- function(fpath){
  setwd(fpath)
  features <- read.csv("cnn_features_150.csv", header = TRUE, as.is=TRUE)
  labels <- read.csv("labels.csv", header = TRUE, as.is=TRUE)
  test_v3 <- read.csv("prediction_inceptionV3.csv", header = TRUE, as.is=TRUE)
  features <- t(features)
  features <- data.frame(features)
  colnames(labels) <- "y"
  dataset <- cbind(features, labels)
  n <- ncol(dataset)
  #rownames(dataset) <- paste(rownames(dataset),".jpg",sep="")
  dataset$image <- rownames(dataset)
  
  #Set test data and training data
  test_data <- dataset %>% filter(dataset$image %in% test_v3$image)
  train_data <- dataset %>% filter(! dataset$image %in% test_v3$image)
  
  #Random Forest model
  system.time(model.rf <- randomForest(train_data[, 1:(n-1)], as.factor(train_data[, n]),mtry=19))
  return(model.rf)
}

library(dplyr)
library(randomForest)
model.rf <- rf_adv("/Users/Michelle/Documents/Michelle/Columbia stat/ads/project 3")
features <- read.csv("cnn_features_150.csv", header = TRUE, as.is=TRUE)
labels <- read.csv("labels.csv", header = TRUE, as.is=TRUE)
test_v3 <- read.csv("prediction_inceptionV3.csv", header = TRUE, as.is=TRUE)
features <- t(features)
features <- data.frame(features)
colnames(labels) <- "y"
dataset <- cbind(features, labels)
n <- ncol(dataset)
#rownames(dataset) <- paste(rownames(dataset),".jpg",sep="")
dataset$image <- rownames(dataset)

#Set test data and training data
test_data <- dataset %>% filter(dataset$image %in% test_v3$image)
train_data <- dataset %>% filter(! dataset$image %in% test_v3$image)
system.time(rf.pred <- predict(model.rf, test_data[, 1:(n-1)], "prob"))

#Write csv file with probability
df.pred <- data.frame(image=test_data$image,rf.pred)
names(df.pred) <- c("image","friedChicken","labradoodle")
write.csv(df.pred,"prediction_rf_cnn150.csv",row.names=FALSE)

#Calculate the precision
rf.pred1 <- predict(model.rf, test_data[, 1:(n-1)], type="class")
table(rf.pred1, test_data[, "y"])
mean(rf.pred1 == test_data[, "y"])