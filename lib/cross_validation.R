########################
### Cross Validation ###
########################

cv.f <- function(train_df, par, K=5, train_f){
  
  n <- dim(train_df)[1]
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- train_df[s != i,]
    train.label <- train_df$label[s != i]
    test.data <- train_df[s == i,]
    test.label <- train_df$label[s == i]
    
    par <- par
    fit <- train_f(train.data, par)
    pred <- test(fit, test.data)  
    cv.error[i] <- mean(pred != test.label)  
    
  }			
  return(c(mean(cv.error),sd(cv.error)))
  
}
