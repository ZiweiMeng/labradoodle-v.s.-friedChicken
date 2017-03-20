install.packages("e1701")
library(e1071)
sift<-data.frame(read.csv("~/sift_features/sift_features.csv"))
label<-as.matrix(read.csv("labels.csv"))[,1]
label<-as.factor(label)
sift<-t(sift)
ncol(sift)
# Name each column
colnames(sift)<-paste("pix",1:ncol(sift))
sift<-cbind.data.frame(sift,class=label)
# Create indicators to divide data into different groups
ind<-sample(1:nrow(sift),nrow(sift))
length(ind)
n_test<-round(length(ind)*0.2,0)
test<-sift[ind[1:n_test],]

train<-sift[ind[n_test+1:length(ind)], ]
# Train the model 

bayes.model<-naiveBayes(train[,-5001],train[,5001])

# Test the model on test data
pred<-predict(bayes.model,test[,-5001])
error<-mean(pred!=test[,5001])
error
