## Random Forest Model

library(randomForest)
features <- read.csv("sift_features.csv", header = TRUE, as.is=TRUE)
labels <- read.csv("labels.csv", header = TRUE, as.is=TRUE)
features <- t(features)
features <- data.frame(features)
colnames(labels) <- "y"
dataset <- cbind(features, labels)

#Set aside 20% test data
index <- sample(1:2000,400)
train <- dataset[-index,]
test <- dataset[index,]

#Build random forest model
model.rf <- randomForest(train[, 1:5000], as.factor(train[, 5001]))

#Predict on test data
rf.pred <- predict(model.rf, test[, 1:5000], type = "class")

#Calculate the precision
table(rf.pred, test[, "y"])
mean(rf.pred == test[, "y"])