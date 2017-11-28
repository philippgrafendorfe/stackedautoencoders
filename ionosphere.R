library(caret)
library(nnet)
library(mlbench)
library(mxnet)

data(Ionosphere)
summary(Ionosphere)

df <- Ionosphere

table(df$Class)
#check for rare event
prop.table(table(df$Class))

# create data partition
set.seed(42)
a <- caret::createDataPartition(df$Class, p = 0.8, list = F)
training <- df[a,]
test <- df[-a,]

summary(training)
summary(test)


#### train with nnet package
formula <- as.formula(paste0("Class ~ ", paste0(names(df)[3:34], collapse = "+")))
set.seed(41)
nnet_model <- nnet(formula, training, size = 23, max.iterations = 100)
prediction <- predict(nnet_model, test, type = "class")

confusionMatrix(prediction, test$Class)

########################################################################################



##neural network with mxnet

df[,35] = as.numeric(df[,35])-1
train.ind = c(1:50, 100:150, 200:250, 300:325)
train.x = data.matrix(df[train.ind, 3:34])
train.y = df[train.ind, 35]
test.x = data.matrix(df[-train.ind, 3:34])
test.y = df[-train.ind, 35]

#We are going to use a multi-layer perceptron as our classifier. 
#In mxnet, we have a function called mx.mlp for building a general multi-layer 
#neural network to do classification or regression.
#mx.mlp requires the following parameters:
#Training data and label
#Number of hidden nodes in each hidden layer
#Number of nodes in the output layer
#Type of the activation
#Type of the output loss
#The device to train (GPU or CPU)
#Other parameters for mx.model.FeedForward.create

mx.set.seed(0)
model <- mx.mlp(train.x, train.y, hidden_node=10, out_node=2, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9,
                eval.metric=mx.metric.accuracy)

preds = predict(model, test.x)
pred.label = max.col(t(preds))-1
table(pred.label, test.y)
