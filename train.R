##################################################
### Train the baseline and the advanced model  ###
##################################################


train <- function(trainD, saveas=NULL){
  # INPUT: 
  #     trainD = data.frame n_images*(n_features + 1*label + 1*image_id) 
  #     saveas (optional): path and file name to save trained models, e.g. './output/models.RData'
  #
  # OUTPUT: trained model objects for both basline and advanced models
  
  source('./lib/cross_validation.R')
  source("./lib/train_bl.R")
  source("./lib/train_adv.R")
  
}

############################################### BASELINE MODEL ###################################################### 

# First, tune n.trees parameter in BL model:
n.trees = seq(100, 300, 50) 
test_err_BL = numeric(length(n.trees))
for(j in 1:length(n.trees)){
  cat("BL model CV: j =", j, "of",length(n.trees), "\n")
  
  par = list(depth=1, 
             shrinkage=0.03, 
             n.trees=n.trees[j])
  
  test_err_BL[j] = cv.f(train_df, par=par, K=5, train_f = train_bl)
}

cat('Grid of test error for BL model tuning: \n') 
print(test_err_BL)   

# Now train BL model on the whole data using optimal shrinkage value
t = proc.time()
n.trees = n.trees[which.min(test_err_BL)]
par = list(depth=1, shrinkage=0.03, n.trees=n.trees)
BL_model = train_bl(train_df, par)

cat("Baseline model training time:", (proc.time()-t)[3], " seconds \n")

