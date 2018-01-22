library(caret)
library(nnet)
library(mlbench)
library(mxnet)
library(autoencoder)
library(h2o)
library(dplyr)

data(Ionosphere)
summary(Ionosphere)

df <- Ionosphere

table(df$Class)
#check for rare event
prop.table(table(df$Class))

# create data partition
set.seed(45)
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



######################################## Autoencoder H20

local.h2o <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads=-1)
a <- caret::createDataPartition(Ionosphere$Class, p = 0.8, list = F)

iono <- as.h2o(Ionosphere)
iono <- iono[,3:34]
response <- "Class"
features <- setdiff(colnames(iono), response)

model_nn <- h2o.deeplearning(x = features,
                             training_frame = iono,
                             model_id = "model_nn",
                             autoencoder = TRUE,
                             reproducible = TRUE, #slow - turn off for real problems
                             ignore_const_cols = FALSE,
                             seed = 42,
                             hidden = c(10, 2, 10), 
                             epochs = 100,
                             activation = "Tanh")




train.iono <- as.data.frame(Ionosphere[a,3:ncol(Ionosphere)])
test.iono <- as.data.frame(Ionosphere[-a,3:ncol(Ionosphere)])

trData<-as.h2o(train.iono)
tsData<-as.h2o(test.iono)



test_autoenc <- h2o.predict(model_nn, tsData[,-ncol(tsData)])

summary(model_nn)


v <- c('V33', 'V29', 'V31', 'V16', 'V34', 'V11')

df_auto <- Ionosphere

df_auto$V33 <- NULL
df_auto$V29 <- NULL
df_auto$V31 <- NULL
df_auto$V16 <- NULL
df_auto$V34 <- NULL
df_auto$V11 <- NULL

######neuralnet

# create data partition
set.seed(35)
a <- caret::createDataPartition(df_auto$Class, p = 0.8, list = F)
training <- df_auto[a,]
test <- df_auto[-a,]

summary(training)
summary(test)


#### train with nnet package
formula <- as.formula(paste0("Class ~ ", paste0(names(df_auto)[3:(ncol(df_auto)-1)], collapse = "+")))
set.seed(31)
nnet_model <- nnet(formula, training, size = (ncol(df_auto)-3), max.iterations = 100)
prediction <- predict(nnet_model, test, type = "class")

confusionMatrix(prediction, test$Class)

########################################################################################



##neural network with mxnet

df_auto[,ncol(df_auto)] = as.numeric(df_auto[,ncol(df_auto)])-1
train.ind = c(1:50, 100:150, 200:250, 300:325)
train.x = data.matrix(df_auto[train.ind, 3:(ncol(df_auto)-1)])
train.y = df[train.ind, ncol(df_auto)]
test.x = data.matrix(df_auto[-train.ind, 3:(ncol(df_auto)-1)])
test.y = df[-train.ind, ncol(df_auto)]

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

mx.set.seed(2)
model <- mx.mlp(train.x, train.y, hidden_node=10, out_node=2, out_activation="softmax",
                num.round=20, array.batch.size=15, learning.rate=0.07, momentum=0.9,
                eval.metric=mx.metric.accuracy)

preds = predict(model, test.x)
pred.label = max.col(t(preds))-1
table(pred.label, test.y)




###############################################################################################



##########FEATURE SELECTION AUTOENCODER############################################################
local.h2o <- h2o.init(nthreads = -1)

df <- df[, 3:ncol(df)]

df$Class <- ifelse(df$Class=='good', 1, 0)

##neuronales netz ohne autoencoder#####################################################

a <- caret::createDataPartition(df$Class, p = 0.8, list = F)

train.df <- data.frame(df[a,1:ncol(df)])
test.df <- data.frame(df[-a,1:ncol(df)])

trData<-as.h2o(train.df)
tsData<-as.h2o(test.df)


res.dl <- h2o.deeplearning(x = 1:(ncol(trData)-1), y = ncol(trData), trData, activation = "Tanh", 
                           hidden=c(10, 2, 10),epochs = 20, nfold = 5)
summary(res.dl)

h2o.predict(res.dl, tsData) %>%
  as.data.frame() %>%
  mutate(actual = as.vector(tsData[, ncol(tsData)])) %>%
  group_by(actual, predict) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))


#####################################################################################################################
##############################################AUTOENCODER############################################################
#####################################################################################################################

# convert data to H2OFrame
Iono_hf <- as.h2o(df)


splits <- h2o.splitFrame(Iono_hf, 
                         ratios = c(0.4, 0.4), 
                         seed = 22)

train_unsupervised  <- splits[[1]]
train_supervised  <- splits[[2]]
test <- splits[[3]]

response <- "Class"
features <- setdiff(colnames(train_unsupervised), response)



model_nn <- h2o.deeplearning(x = features,
                             training_frame = train_unsupervised,
                             model_id = "model_nn",
                             autoencoder = TRUE,
                             reproducible = TRUE, #slow - turn off for real problems
                             ignore_const_cols = FALSE,
                             seed = 22,
                             hidden = c(10, 2, 10), 
                             epochs = 100,
                             activation = "Tanh")


model_nn

#Convert to autoencoded representation
test_autoenc <- h2o.predict(model_nn, test)

#But we could use the reduced dimensionality representation of one of the hidden layers as features for model training. 
#An example would be to use the 10 features from the first or third hidden layer:
library(dplyr)
# let's take the third hidden layer
train_features <- h2o.deepfeatures(model_nn, train_unsupervised, layer = 3) %>%
  as.data.frame() %>%
  mutate(Class = as.factor(as.vector(train_unsupervised[, ncol(train_unsupervised)]))) %>%
  as.h2o()

features_dim <- setdiff(colnames(train_features), response)

model_nn_dim <- h2o.deeplearning(y = response,
                                 x = features_dim,
                                 training_frame = train_features,
                                 reproducible = TRUE, #slow - turn off for real problems
                                 balance_classes = TRUE,
                                 ignore_const_cols = FALSE,
                                 seed = 22,
                                 hidden = c(10, 2, 10), 
                                 epochs = 100,
                                 activation = "Tanh")

model_nn_dim


#For measuring model performance on test data, we need to convert the test data to the same 
#reduced dimensions as the trainings data:
 
test_dim <- h2o.deepfeatures(model_nn, test, layer = 3)

h2o.predict(model_nn_dim, test_dim) %>%
  as.data.frame() %>%
  mutate(actual = as.vector(test[, ncol(test)])) %>%
  group_by(actual, predict) %>%
  summarise(n = n()) %>%
  mutate(freq = n / sum(n))



