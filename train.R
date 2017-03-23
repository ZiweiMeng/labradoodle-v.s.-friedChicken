##################################################
### Train the baseline and the advanced model  ###
##################################################


##################
# baseline model #
##################

source('lib/gbm_bl.R')

###################
# advanced models #
###################

source('lib/selectFeatures.R')
source('lib/svm_adv.R')
# source('lib/rf_adv.R')
