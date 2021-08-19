# 1
library(readr)
library(caTools)
library(dplyr)
library(ggplot2)
install.packages("corrplot")
library(corrplot)
install.packages("class")
library(class)
# Importing the dataset
glass <- read.csv('C:\\Users\\WIN10\\Desktop\\LEARNING\\DS\\glass.csv')
# Standardizing the data
glassn <- scale(glass[1:9])
summary(glassn)
#Join the standardized data with the target column
glassf <- cbind(glassn,glass[10])
# Check if there are any missing values to impute. 
anyNA(glassf)
head(glassf)
# Splitting the model
set.seed(101)
sample <- sample.split(glassf$Type,SplitRatio = 0.70)
train <- subset(glassf,sample==TRUE)
test <- subset(glassf,sample==FALSE)
# Training the model
glassp <- knn(train[1:9],test[1:9],train$Type,k=1)
# Finding error in prediction
error <- mean(glassp!=test$Type)
# Confusion Matrix
library(caret)
confusionMatrix(factor(glassp, levels=1:7), factor(test$Type, levels=1:7))
glassp <- NULL
errorrate <- NULL
for (i in 1:10) {
  glassp <- knn(train[1:9],test[1:9],train$Type,k=i)
  errorrate[i] <- mean(glassp!=test$Type)
  
}
knnerror <- as.data.frame(cbind(k=1:10,error.type =errorrate))
ggplot(knnerror,aes(k,error.type))+ 
  geom_point()+ 
  geom_line() + 
  scale_x_continuous(breaks=1:10)+ 
  theme_bw() +
  xlab("Value of K") +
  ylab('Error')
# from the graph we can say that the error is minimum for K=1
# Therefore we use 1 for K value
glassp <- knn(train[1:9],test[1:9],train$Type,k=1)
# Error in prediction
error <- mean(glassp!=test$Type)
# Confusion Matrix
confusionMatrix(factor(glassp, levels=1:7), factor(test$Type, levels=1:7))


predicted.type <- knn(train[1:9],test[1:9],train$Type,k=1)
#Error in prediction
error <- mean(predicted.type!=test$Type)
#Confusion Matrix
confusionMatrix(factor(predicted.type, levels=1:7), factor(test$Type, levels=1:7))

