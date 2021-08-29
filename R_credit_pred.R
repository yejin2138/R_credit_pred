library(dplyr)
library(MASS)
library(nnet)
suppressMessages(library(magrittr))
suppressMessages(library(skimr)) %>% suppressWarnings()

# 데이터 가져오기
data <- read.csv("./train.csv", header=T, na.strings = c("."))

#DAYS_BIRTH
data$DAYS_BIRTH <- as.integer(-(data$DAYS_BIRTH/30))
data$DAYS_BIRTH <- as.numeric(data$DAYS_BIRTH)

#DAYS_EMPLOYED
data[data$DAYS_EMPLOYED>0, "DAYS_EMPLOYED"]=0
data$DAYS_EMPLOYED <- as.integer(-(data$DAYS_EMPLOYED/30))
data$DAYS_EMPLOYED <- as.numeric(data$DAYS_EMPLOYED)

#begin_month
data$begin_month <- -(data$begin_month)

#occyp_type, FLAG_MOBIL, index
data <- subset(data, select=-c(occyp_type))
data <- subset(data, select = -c(FLAG_MOBIL))
data <- subset(data, select = -c(index))

#family_size
data[data$family_size>=15, "family_size"]=15

#child_num
data[data$child_num>=14, "child_num"]=14

#income_type
data$income_type=factor(data$income_type,
                        levels=c("Commercial associate","Working", "State servant", "Pensioner", "Student"),
                        labels=c(0,1,2,3,4))
data$income_type <- as.numeric(data$income_type)

#edu_type
data$edu_type=factor(data$edu_type,
                     levels=c("Lower secondary","Secondary / secondary special",
                              "Incomplete higher", "Higher education", "Academic degree"),
                     labels=c(0,1,2,3,4))
data$edu_type <- as.numeric(data$edu_type)

#family_type
data$family_type=factor(data$family_type,
                        levels=c("Civil marriage","Married", "Separated", "Single / not married", "Widow"),
                        labels=c(0,1,2,3,4))
data$family_type <- as.numeric(data$family_type) 

#house_type
data$house_type=factor(data$house_type,
                       levels=c("House / apartment","With parents", "Co-op apartment", 
                                "Municipal apartment", "Rented apartment", "Office apartment"),
                       labels=c(0,1,2,3,4,5))
data$house_type <- as.numeric(data$house_type)

#factor
data$gender = as.integer(factor(data$gender))
data$car = as.integer(factor(data$car))
data$reality = as.integer(factor(data$reality))
data$credit = factor(data$credit)
str(data)

# data 분할
data_rand <- data[order(runif(26457)),]
data_train <- data_rand[1:21166, ]
data_test <- data_rand[21167:26457, ]
nrow(data_train)
nrow(data_test)

# test data
test_credit <- data_test$credit

# Logistic regression
lr_fit<-multinom(credit~car+reality+child_num+income_total+income_type
                 +edu_type+family_type+house_type+DAYS_BIRTH+DAYS_EMPLOYED
                 +work_phone+email+family_size+begin_month,data=data_train)

lr_pred <- predict(lr_fit, data_test, type='class')
table(lr_pred, test_credit)
mean(lr_pred==test_credit)  #맞을 확률
mean(lr_pred!=test_credit)  #틀릴 확률

# LDA
lda.fit<-lda(credit~car+reality+child_num+income_total+income_type
             +edu_type+family_type+house_type+DAYS_BIRTH+DAYS_EMPLOYED
             +work_phone+email+family_size+begin_month,data=data_train)
lda.fit

lda.pred=predict(lda.fit, data_test)$class
table(lda.pred, test_credit)
mean(lda.pred==test_credit)  #맞을 확률
mean(lda.pred!=test_credit)  #틀릴 확률

# QDA
qda.fit = qda(formula=credit~car+reality+child_num+income_total+income_type
              +edu_type+family_type+house_type+DAYS_BIRTH+DAYS_EMPLOYED
              +work_phone+email+family_size+begin_month, data=data_train)
qda.fit

qda.pred=predict(qda.fit, data_test)$class
table(qda.pred, test_credit)
mean(qda.pred==test_credit)  #맞을 확률
mean(qda.pred!=test_credit)  #틀릴 확률

# KNN
library(class)
train.X=cbind(data_train$car, data_train$reality, data_train$child_num, 
              data_train$income_total, data_train$income_type, data_train$email, 
              data_train$edu_type, data_train$family_type, data_train$house_type, 
              data_train$DAYS_BIRTH, data_train$DAYS_EMPLOYED, 
              data_train$work_phone, data_train$family_size, data_train$begin_month)
train.Y=data_train$credit
test.X=cbind(data_test$car, data_test$reality, data_test$child_num, 
             data_test$income_total, data_test$income_type, data_test$email, 
             data_test$edu_type, data_test$family_type, data_test$house_type, 
             data_test$DAYS_BIRTH, data_test$DAYS_EMPLOYED, 
             data_test$work_phone, data_test$family_size, data_test$begin_month)

knn.pred=knn(train.X, test.X, train.Y, k=20)
table(knn.pred, test_credit)
mean(knn.pred==test_credit)  #맞을 확률
mean(knn.pred!=test_credit)  #틀릴 확률

knn.pred=knn(train.X, test.X, train.Y, k=40)
table(knn.pred, test_credit)
mean(knn.pred==test_credit)  #맞을 확률
mean(knn.pred!=test_credit)  #틀릴 확률

knn.pred=knn(train.X, test.X, train.Y, k=80)
table(knn.pred, test_credit)
mean(knn.pred==test_credit)  #맞을 확률
mean(knn.pred!=test_credit)  #틀릴 확률

# TREE - random forest
library(randomForest)
set.seed(1)
rf.pred=randomForest(credit~car+reality+child_num+income_total+income_type
                     +edu_type+family_type+house_type+DAYS_BIRTH+DAYS_EMPLOYED
                     +work_phone+email+family_size+begin_month, data=data_train, 
                     mtry=(sqrt(14)), importance=TRUE)

rf.pred
plot(rf.pred)
importance(rf.pred)
varImpPlot(rf.pred)

yhat.rf=predict(rf.pred, newdata=data_test, type='class')
table(yhat.rf, test_credit)
mean(yhat.rf==test_credit)
mean(yhat.rf!=test_credit)


#K-fold-randomForest
t_index <- sample(1:nrow(data), size=nrow(data))
split_index <- split(t_index, 1:10)
class(split_index)
split_index[[1]]

accuracy_3 <- c()        # 데이터를 받을 빈 벡터
for(i in 1:10){
  test <- data[split_index[[i]],]  
  train <- data[-split_index[[i]],]  
  
  set.seed(1000)
  rf.pred <- randomForest(credit~car+reality+child_num+income_total+income_type
                          +edu_type+family_type+house_type+DAYS_BIRTH+DAYS_EMPLOYED
                          +work_phone+email+family_size+begin_month, 
                          data=train, mtry=(sqrt(14)), importance=TRUE)
  plot(rf.pred)
  importance(rf.pred)    
  varImpPlot(rf.pred)     
  
  yhat.rf <- predict(rf.pred, test)
  table <- table(real=test$credit, predict=yhat.rf)
  
  #정확도
  accuracy_3[i] <- sum(diag(table))/sum(table)
}
sum=0
for(i in 1:10) {
  sum = sum + accuracy_3[i] 
}
avg=sum/100
avg
#K-fold-KNN
library(caret)
k.fold <- 50
folds <- createFolds(data$credit, k = k.fold)
folds
kk.seq = seq(1, 201, by = 10)
names(kk.seq) <- sapply(kk.seq, function(kk){
  paste("k = ", kk)
})
kk.result <- sapply(kk.seq, function(kk){
  xval.result <- sapply(folds, function(idx){
    df.train.i <- data[-idx, ]
    df.test.i <- data[idx, ]
    model.i <- knn3(credit~car+reality+child_num+income_total+income_type
                    +edu_type+family_type+house_type+DAYS_BIRTH+DAYS_EMPLOYED
                    +work_phone+email+family_size+begin_month, data = df.train.i, k = kk)
    predict.i <- predict(model.i, df.test.i, type ="class")
    accuracy.i <- sum(predict.i == df.test.i$credit) / length(predict.i)
    return(accuracy.i)
  })
  print(xval.result)
  return(mean(xval.result))
})
print(kk.result)
plot.data = data.frame(k = kk.seq, Accuracy = kk.result)
plot(formula = Accuracy ~ k, data = plot.data, type="o", pch = 20, main  = "validation - optimal k")

with(plot.data, text(Accuracy ~ k, labels = plot.data$k, pos = 1, cex = 0.7))
min(plot.data[plot.data$Accuracy %in% max(plot.data$Accuracy), "k"])

