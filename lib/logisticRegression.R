setwd('/Users/Zoe/Documents/Spring2017/GR5243/MyPrjs/spr2017-proj3-group3/lib/')

library(dplyr)
library(data.table)

# set features path and features number
k <- 150
finput <- '../data/cnn_features_150.csv'
foutput <- '../output/prediction_logisticRegression_cnn150.csv'

# read features and label
# set index as a column
df <- fread(finput,stringsAsFactors = FALSE)
df <- t(df)
df <- data.frame(df)
labels <- fread('../data/labels.csv')
df$label <- factor(labels$V1)
df$image <- rownames(df)


# separate training and test data sets
test_imgs <- fread('../output/prediction_inceptionV3.csv')
test_imgs <- test_imgs$image

test_df <- df[rownames(df)%in%test_imgs,]
train_df <- df[!(rownames(df)%in%test_imgs),]
start <- Sys.time()
logistic.fit <- glm(label ~ ., data = train_df[,c(1:k+1)], family = "binomial")
end <- Sys.time()
(end - start)

labradoodle_preds <- predict(logistic.fit,newdata = test_df,type = 'response')
friedChicken_preds <- 1 - labradoodle_preds

output <- data.frame(image=test_df$image,labradoodle=labradoodle_preds,friedChicken=friedChicken_preds)
fwrite(output,foutput)

