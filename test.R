######################################################
### Fit the classification model with testing data ###
######################################################


test = function(fit_train, dat_test){
  # Fit the classfication model with testing data
  # INPUT: 
  #     fit_train = trained model object, either gbm or xgb.Booster
  #     dat_test = processed features from testing images 
  #
  # OUTPUT: training model specification
  
  library('gbm')
  library('xgboost')
  
  k <- dim(dat_test)[2]-2
  
  pred = switch(class(fit_train), 
                gbm = predict(fit_train, 
                              newdata = dat_test[,1:k], 
                              n.trees = fit_train$n.trees, 
                              type="response"),
                
                xgb.Booster = predict(fit_train, 
                                      newdata = dat_test)
  )
  
  
  return(as.numeric(pred> 0.5))
}
