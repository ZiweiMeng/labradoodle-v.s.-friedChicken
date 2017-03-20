######################################################
### Fit the classification model with testing data ###
######################################################

### Author:Nanjun Wang
### Project 3
### ADS Spring 2017

test.fn <- function(data_test, fit_model){
  
  ### Fit the classfication model with testing data
  
  ### Input: 
  ###  - the fitted classification model using training data
  ###  -  processed features from testing images 
  ### Output: training model specification
  
  ### load libraries
    library(e1071)
    
    pred <- predict(fit_model, data_test, probability = T)
    pred.prob <- attr(pred, 'probabilities')
    return(pred.prob)
  }