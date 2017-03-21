################################################
###             preprocess data              ###
################################################
library(dplyr)
library(data.table)
library(stringi)

source('train.R')
source('test.R')

# read features
baseline_features <- '../data/sift_features.csv'
advance_features <- '../data/cnn_features_150.csv'
# set model saved path
bl_save <- '../output/blModel.RData'
adv_save <- '../output/advModel.RData'
# set test data path
test_bl_path <- '../data/test_sift_features.csv'
test_adv_path <- '../data/test_cnn_features.csv'
test_labels <- '../data/test_labels.csv'

bl_df <- fread(baseline_features,stringsAsFactors = FALSE)
adv_df <- fread(advance_features, stringsAsFactors = FALSE)

# set index as a column
labels <- fread('../data/labels.csv')

bl_df <- t(bl_df)
bl_df <- data.frame(bl_df)
bl_df$label <- factor(labels$V1)
bl_df$image <- rownames(bl_df)

adv_df <- t(adv_df)
adv_df <- data.frame(adv_df)
adv_df$label <- factor(labels$V1)
adv_df$image <- lapply(rownames(adv_df),function(x) stri_sub(x,1,-5))

# separate training and test data sets
test_imgs <- fread('../output/prediction_inceptionV3.csv')
test_imgs <- test_imgs$image
test_imgs <- lapply(test_imgs,function(x) stri_sub(x,1,-5))

bl_test <- bl_df[bl_df$image%in%test_imgs,]
bl_train <- bl_df[!(bl_df$image%in%test_imgs),]
adv_test <- adv_df[adv_df$image%in%test_imgs,]
adv_train <- adv_df[!(adv_df$image%in%test_imgs),]

#################################
##### training model        #####
#################################
train(bl_train,saveas=bl_save)
train(adv_train,saveas=adv_save)

#################################
###       test model          ###
#################################
load(bl_save)
load(adv_save)
test_bl <- fread(test_bl_path, stringsAsFactors = FALSE)
test_adv <- fread(test_adv_path, stringsAsFactors = FALSE)

# add label and image ids
labels <- fread(test_labels)

test_bl <- t(test_bl)
test_bl <- data.frame(test_bl)
test_bl$label <- factor(labels$V1)
test_bl$image <- rownames(test_bl)

test_adv <- t(test_adv)
test_adv <- data.frame(test_adv)
test_adv$label <- factor(labels$V1)
test_adv$image <- lapply(rownames(test_adv),function(x) stri_sub(x,1,-5))

test(gbm_fit,test_bl)
#test(ensenmble_fit, test_adv)














