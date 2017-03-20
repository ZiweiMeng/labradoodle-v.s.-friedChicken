########################
### Cross Validation ###
########################

### Author: Nanjun Wang
### Project 3
### ADS Spring 2017

cv.function <- function(X.train, y.train, c, K){
  
  library(e1071)
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