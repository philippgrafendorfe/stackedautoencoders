library(caret)
library(nnet)
library(mlbench)
library(mxnet)
library(autoencoder)

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




###############################################################################################
#AUTOENCODER
###############################################################################################
df <- df[, 3:35]
train_matrix_nolabel <- as.matrix(df[,1:length(df) - 1])


#### with autoencoder package
nl = 3 ## number of layers (default is 3: input, hidden, output)
unit.type = "logistic" ## specify the network unit type, i.e., the unit's
## activation function ("logistic" or "tanh")
#Nx.patch=10 ## width of training image patches, in pixels 
#Ny.patch=10 ## height of training image patches, in pixels 
#N.input = Nx.patch*Ny.patch ## number of units (neurons) in the input layer (one unit per pixel) 
#N.hidden = 10*10 ## number of units in the hidden layer
N.input = 32 ## number of units (neurons) in the input layer (one unit per pixel)
N.hidden = 12 ## number of units in the hidden layer
lambda = 0.0002 ## weight decay parameter
beta = 6 ## weight of sparsity penalty term
rho = 0.01 ## desired sparsity parameter
epsilon <- 0.001 ## a small parameter for initialization of weights
## as small gaussian random numbers sampled from N(0,epsilon^2)
max.iterations = 2000 ## number of iterations in optimizer

autoencoder.object <- autoencode(X.train = train_matrix_nolabel
                                 ,nl = nl
                                 ,N.hidden = N.hidden
                                 ,unit.type = unit.type
                                 ,lambda = lambda
                                 ,beta = beta
                                 ,rho = rho
                                 ,epsilon = epsilon
                                 ,optim.method = "BFGS"
                                 ,max.iterations = max.iterations
                                 ,rescale.flag = TRUE
                                 ,rescaling.offset = 0.001)

## Extract weights W and biases b from autoencoder.object: 
W <- autoencoder.object$W 
b <- autoencoder.object$b 
## Visualize learned features of the autoencoder: 
#visualize.hidden.units(autoencoder.object,Nx.patch,Ny.patch)


## Report mean squared error for training and test sets: 
cat("autoencode(): mean squared error for training set: ", round(autoencoder.object$mean.error.training.set,3),"\n")


X.output <- predict(autoencoder.object, X.input=train_matrix_nolabel, hidden.output=FALSE)$X.output
