train_bl <- function(fpath,run.evaluation,k=5000,K=5,run.cv=FALSE){
  # Training function for the baseline model (GBM)
  # INPUT: 
  #     trainD =  data.frame n_images*(n_features + 1*label + 1*image_id)
  #     par = list of values for params depth, shrinkage and n.trees
  #
  # OUTPUT: trained model object
  
  # process data
  bl_df <- fread(fpath,stringsAsFactors = FALSE)
  bl_labels <- fread(train_labels)
  bl_df <- data.frame(t(bl_df))
  bl_df$label <- factor(bl_labels$V1)
  bl_df$image <- rownames(bl_df)
  
  # For evaluation test error purpose, separate train imgs into train set and test set.
  if(run.evaluation){
    # separate training and test data sets
    
    #library(stringi)
    test_imgs <- fread('output/prediction_inceptionV3.csv')
    test_imgs <- test_imgs$image
    #test_imgs <- lapply(test_imgs,function(x) stri_sub(x,1,-5))
    
    #testD <- bl_df[bl_df$image%in%test_imgs,]
    trainD <- bl_df[!(bl_df$image%in%test_imgs),]
  } else{
    trainD <- bl_df
  }
  
  trainD$label <- as.numeric(trainD$label) - 1

  
  #Cross validaiton---tunning parameters
  if(run.cv){
    list_ntrees <- c(150,250,320)
    n.pars <- length(list_ntrees)
    errors <- rep(NA,n.pars)
    
    for(j in 1:n.pars){
      errors[j] <- cv.f(trainD, list_ntrees[j], k, K)
    }
    
    best.n.trees <- list_ntrees[which.min(errors)]
  } else{
    best.n.trees <- 250
  }

  
  # best model after tunning parameters
  bestfit_baseline <- gbm(label~ ., data = trainD[,1:k+1],interaction.depth = 1, n.trees = best.n.trees, shrinkage = 0.03)
  
  return(bestfit_baseline)

}

cv.f <- function(train_df, n.trees, k=5000, K=5){
  
  n <- dim(train_df)[1]
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.error <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- train_df[s != i,]
    train.label <- train_df$label[s != i]
    test.data <- train_df[s == i,]
    test.label <- train_df$label[s == i]
    
    n.trees <- n.trees
    fit <- gbm(label~ ., data = train.data[,1:k+1],interaction.depth = 1, n.trees = n.trees, shrinkage = 0.03)
    pred <- predict(fit, 
                    newdata = test.data[,1:k+1], 
                    n.trees = fit$n.trees, 
                    type="response")
    pred <- as.numeric(pred> 0.5)
    cv.error[i] <- mean(pred != test.label)  
    
  }			
  return(mean(cv.error))
  
}
