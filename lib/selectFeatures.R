####################################################
#   use pca to select sub set of features to use   #
####################################################
# Author: Ziwei Meng
# date: 2017-03-22

selectFeatures <- function(fpath,save.path,n.features=350,run.feature.test=FALSE,run.feature.train=TRUE,train.sd=NULL,prin_comp=NULL){
  if(run.feature.test){
    run.feature.train <- FALSE
  }
  if(run.feature.train){
    cat('start PCAing features...')
    # process data  
    features <- fread(fpath,stringsAsFactors = FALSE)
    features <- data.frame(t(features))
    n <- dim(features)[2]
    # format of image names in test_ids and features rownames should be the same
    test_ids <- fread("output/prediction_inceptionV3.csv")
    features$image <- rownames(features)
    # split into training and test sets
    test.features <- features %>% filter(features$image %in% test_ids$image)
    train.features <- features %>% filter(!features$image %in% test_ids$image)
    
    # fit a pca model on training data
    # remove columns where sd is 0
    train.sd <- apply(train.features[,1:n],2,sd)
    train.pca <- train.features[,1:n][,train.sd!=0]
    test.pca <- test.features[,1:n][,train.sd!=0]
    set.seed(59)
    prin_comp <- prcomp(train.pca,scale. = T)
    # select number of features to be remained
    # std_dev <- prin_comp$sdev
    # pr_var <- std_dev^2
    # prop_varex <- pr_var/sum(pr_var)
    # plot(prop_varex, xlab = "Principal Component",
    #            ylab = "Proportion of Variance Explained",
    #            type = "b")
    
    reduced.train <- data.frame(image=train.features$image,prin_comp$x[,1:n.features])
    # transform test data into the new space by trained matrix
    reduced.test <- predict(prin_comp,newdata = test.pca)
    reduced.test <- as.data.frame(reduced.test)
    reduced.test <- reduced.test[,1:n.features]
    reduced.test <- data.frame(image=test.features$image,reduced.test)
    # combine training and test data
    reduced.data <- rbind(reduced.train,reduced.test)
    reduced.data$image <- as.character(reduced.data$image)
    reduced.data <- reduced.data %>% arrange(image)
    reduced.data <- data.frame(t(reduced.data))
    # save new features into save.path, same as advanced_train_features
    fwrite(reduced.data,file = save.path,row.names = F,col.names = F)
  }
  if(run.feature.test){
    #cat('test')
    # process data  
    test.features <- fread(fpath,stringsAsFactors = FALSE)
    test.features <- data.frame(t(test.features))
    n <- dim(test.features)[2]
    test.pca <- test.features[,1:n][,train.sd!=0]
    test.features$image <- rownames(test.features)
    # transform test data into the new space by trained matrix
    reduced.test <- predict(prin_comp,newdata = test.pca)
    reduced.test <- as.data.frame(reduced.test)
    reduced.test <- reduced.test[,1:n.features]
    reduced.test <- data.frame(image=test.features$image,reduced.test)
    #reduced.test <- reduced.test %>% arrange(image)
    reduced.test <- data.frame(t(reduced.test))
    # save new features into save.path, same as advanced_train_features
    fwrite(reduced.test,file = save.path,row.names = F,col.names = F)
  }
  return(list(prin_comp,train.sd))
  
}
