######################################################
### Read data and set it into test and train data  ###
######################################################

### Author:Nanjun Wang
### Project 3
### ADS Spring 2017

read_set_fn <- function(file_name){
  
  library(dplyr)
  ###read feature files
  file_dir<- paste("/Users/ouminamikun/Desktop/Columbia/Spring 2017/ADS/Project 3/",file_name, sep = "")
  prediction_inception <- read.csv("/Users/ouminamikun/Desktop/Columbia/Spring 2017/ADS/spr2017-proj3-group3/output/prediction_inceptionV3.csv")
  label <- read.csv("/Users/ouminamikun/Desktop/Columbia/Spring 2017/ADS/Project 3/labels.csv")
  label$V1 <- as.numeric(label$V1)
  dataset <- read.csv(file_dir)
  dataset <- data.frame(t(dataset))
  dataset <- cbind(label,dataset)
  file_rowname <- rownames(dataset)[1]
  value <- "jpg"
  if(grepl(value, file_rowname)){
    dataset$image <- rownames(dataset)
  } else{
    dataset$image <- paste(rownames(dataset),".jpg", sep = "")
  }
  
  ###set test and train data 
  
  test_data <- dataset %>% filter(dataset$image %in% prediction_inception$image)
  train_data <- dataset %>% filter(!dataset$image %in% prediction_inception$image)
  
  return(list(dataset,test_data, train_data))
}