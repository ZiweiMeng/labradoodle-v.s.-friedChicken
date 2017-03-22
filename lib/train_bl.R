train_bl <- function(trainD,par=NULL){
  # Training function for the baseline model (GBM)
  # INPUT: 
  #     trainD =  data.frame n_images*(n_features + 1*label + 1*image_id)
  #     par = list of values for params depth, shrinkage and n.trees
  #
  # OUTPUT: trained model object
  
  library('gbm')
  
  k <- dim(trainD)[2]-1
  
  if(is.null(par)){
    depth = 1
    shrinkage = 0.03
    n.trees = 250
  }
  else {
    eval(parse(text = paste(names(par), par, sep = '=', collapse = ';')))
  }
  trainD$label <- as.numeric(trainD$label) - 1
  t1 <- Sys.time()
  gbm_fit <- gbm(label~ ., data = trainD[,1:k],interaction.depth = depth, n.trees = n.trees, shrinkage = shrinkage)
  t2 <- Sys.time() 
  t2 - t1
  
  return(gbm_fit)

}