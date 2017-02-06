#START: Course Project

#Objectives:
#1. Describe how you built the model.

#2. Describe how you used cross validation
#I will split the training set into 5 training subsets with cross-validation set in my training control.

#3. What do I think the out-of-sample error will be?
# The out-of-sample error will be higher than my in-sample error rate due to the in-sample having only 6 participants.
# The random forests methodology will be 

library(caret)
library(ggplot2)
training<-read.csv(file="pml-training.csv")

#dimensions
dim(training)

#examining properties
str(training)
training.classes<-NULL
for(i in 1:dim(training)[2]){training.classes<-append(training.classes,class(training[,i]))}

#features that are factors:
training.factors<-names(training)[training.classes=="factor"]
sum(as.numeric(training.classes=="factor"))
#features that are numeric:
training.numerics<-names(training)[training.classes=="numeric"]
sum(as.numeric(training.classes=="numeric"))
#features that are integer:
training.integers<-names(training)[training.classes=="integer"]
sum(as.numeric(training.classes=="integer"))


#Are the numerics complete?
numericNAs<-NULL

for(i in 1:dim(training[,training.numerics])[2]){
  pct.comp<-NULL
  print(paste0("Numeric Feature: ",training.numerics[i]))
  pct.comp<-sum(as.numeric(complete.cases(training[,training.numerics[i]])))/length(training[,1])
  print(pct.comp)
  numericNAs<-append(numericNAs,pct.comp)
}
#Since only 2% or so is only available for (non-NA) for several numeric features, I will be discarding features non-complete data.
training.numerics.comp<-training.numerics[numericNAs==1]

#Are the factors complete?
factorsNAs<-NULL

for(i in 1:dim(training[,training.factors])[2]){
  pct.comp<-NULL
  print(paste0("Factor Feature: ",training.factors[i]))
  pct.comp<-sum(as.numeric(!training[,training.factors[i]]==""))/dim(training)[1]
  print(pct.comp)
  factorsNAs<-append(factorsNAs,pct.comp)
}
training.factors.comp<-training.factors[factorsNAs==1]

#Are the integers complete?
integersNAs<-NULL

for(i in 1:dim(training[,training.integers])[2]){
  pct.comp<-NULL
  print(paste0("Integer Feature: ",training.integers[i]))
  pct.comp<-sum(as.numeric(complete.cases(training[,training.integers[i]])))/dim(training)[1]
  print(pct.comp)
  integersNAs<-append(integersNAs,pct.comp)
}

#Since only 2% or so is only available for (non-NA) for several integer features, I will be discarding features non-complete data.
#It would not be reliable to attempt to impute from them.
training.integers.comp<-training.integers[integersNAs==1]


set.seed("5432")

#Concatenate the known good numeric/int
training.quant<-append(training.integers.comp, training.numerics.comp)

#concatenate the factors with the quant
training.comp<-append(training.quant,training.factors.comp)

#remove fields for timestamps and usernames
training.features<-training.comp[-grep("^.*user|^.*time|^.*window|^X$",training.comp)]


#splitting training into two sets: training, and testing.
set.seed(5432)
inTrain<-createDataPartition(training$classe,p=0.6,list=FALSE)
training.sub<-training[inTrain,]
training.sub<-training.sub[,c(training.features)]
testing.sub<-training[-inTrain,]
testing.sub<-testing.sub[,c(training.features)]

#Create a decision tree of training based on rpart.
#Need to shrink set by 1/2 to fit into memory
set.seed(5432)
training.sub.dtree<-training.sub[createDataPartition(training.sub$classe,p=0.5,list=FALSE),]
set.seed(5432)
dTree.modFit<-train(classe~.,method="rpart",data=training.sub.dtree)

library(rattle)

fancyRpartPlot(dTree.modFit$finalModel,shadow.offset=0)

dTree.predict<-predict(dTree.modFit,testing.sub)

#We can see that the decision tree is only around 48.97% accurate.
confusionMatrix(testing.sub$classe,dTree.predict)



#Developing random forests on the training subset. Using the training subset with a 4-fold cross-validation
#Using parallel rf and formatting predictors and outcomes.
library(mlbench)
x=training.sub[,-53]
y=training.sub[,53]
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) #leaves 1 core for OS
registerDoParallel(cluster)

#Train RF model
set.seed(5432)
train_control<-trainControl(method="cv",number=4,allowParallel=TRUE)
rf.modFit<-train(x,y,method="rf",trControl=train_control)

#Shut down parallel cluster
stopCluster(cluster)
registerDoSEQ()

#Getting the variances for each variable:
varImp(rf.modFit)

#Predict with random forests:
rf.predict<-predict(rf.modFit,testing.sub)
#we get a 99.08% accuracy with the random forests prediction model against out-of-sample test subset.
confusionMatrix(testing.sub$classe,rf.predict)


#Predict the scores with the test data set of 20 entries.

testData<-read.csv(file="pml-testing.csv")
testData.clean<-testData[,c(training.features[-53])]
rf.predict.td<-predict(rf.modFit,testData.clean)
rf.predict.td
