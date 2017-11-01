# library(neuralnet)
library(SAENET)
library(autoencoder)
library(caret)
library(nnet)
library(deepnet)
library(dplyr)
library(Matrix)

#### packages to be tested: h2o with deep autoencoders

# data exploration
df <- data.table::fread(input =  '../Data/sensor_readings_24.csv', data.table = F, stringsAsFactors = T)
df <- plyr::rename(df, c('V25' = 'Direction'))
table(df$Direction)
prop.table(table(df$Direction))

train_matrix_nolabel <- as.matrix(df[,1:length(df) - 1])
train_df_labeled <- df

#### with autoencoder package
nl = 3 ## number of layers (default is 3: input, hidden, output)
unit.type = "logistic" ## specify the network unit type, i.e., the unit's
## activation function ("logistic" or "tanh")
N.input = 24 ## number of units (neurons) in the input layer (one unit per pixel)
N.hidden = 12 ## number of units in the hidden layer
lambda = 0.0002 ## weight decay parameter
beta = 6 ## weight of sparsity penalty term
rho = 0.01 ## desired sparsity parameter
epsilon <- 0.001 ## a small parameter for initialization of weights
## as small gaussian random numbers sampled from N(0,epsilon^2)
max.iterations = 2000 ## number of iterations in optimizer

autoencoder.object <- autoencode(X.train = train
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

# create data partition
set.seed(42)
a <- caret::createDataPartition(df$Direction, p = 0.8, list = F)
training <- df[a,]
test <- df[-a,]


#### train with neuralnet package
#### ist im handling so schlecht, dass es für mich hier rausfällt.
# train_dummy_matrix <- training
# train_dummy_matrix['MoveForward'] <- ifelse(training$Direction == 'Move-Forward', 1, 0)
# train_dummy_matrix['SharpRightTurn'] <- ifelse(training$Direction == 'Sharp-Right-Turn', 1, 0)
# train_dummy_matrix['SlightLeftTurn'] <- ifelse(training$Direction == 'Slight-Left-Turn', 1, 0)
# train_dummy_matrix['SlightRightTurn'] <- ifelse(training$Direction == 'Slight-Right-Turn', 1, 0)
# train_dummy_matrix$Direction <- NULL
# # names(train_dummy_matrix) <- make.names(train_dummy_matrix)
# formula <- as.formula(paste0("MoveForward + SharpRightTurn + SlightLeftTurn + SlightRightTurn ~ ", paste0(names(df)[1:24], collapse = "+")))
# 
# neuralnet_model <- neuralnet(formula = formula, data = train_dummy_matrix)
# 
# neuralnet_prediction <- prediction(neuralnet_model, test[-length(test)])

#### train with nnet package
formula <- as.formula(paste0("Direction ~ ", paste0(names(df)[1:24], collapse = "+")))
set.seed(41)
nnet_model <- nnet(formula, training, size = 23, max.iterations = 100)
prediction <- predict(nnet_model, test, type = "class")

confusionMatrix(prediction, test$Direction)

# deepnet approach
train_data <- training[-c(length(training))]
train_classes <- training$Direction

test_data <- test[-c(length(test))]
test_classes <- test$Direction

preProcValues <- preProcess(train_data, method = "range")

trainTransformed <- predict(preProcValues, train_data)
testTransformed <- predict(preProcValues, test_data)

dnngrid <- expand.grid(layer1 = 12:18
                       ,layer2 = 12:18
                       ,layer3 = 12:18
                       ,hidden_dropout = c(0, 0.1)
                       ,visible_dropout = 0)

fitControl <- trainControl(
  method = "cv"
  ,number = 5
  # ,search = "random"
  ,returnResamp = "all"
  )

dnnfit <- train(x = train_data
                ,y = train_classes
                ,method = "dnn"
                ,trControl = fitControl
                ,tuneGrid = dnngrid
                # ,tuneLength = 30
                ,preProcess = "range")

dnn_predict <- predict(dnnfit, test_data)
confusionMatrix(dnn_predict, test_classes)

train_classes_probe <- plyr::mapvalues(train_classes
                                       ,from = c("Move-Forward", "Slight-Right-Turn", "Sharp-Right-Turn", "Slight-Left-Turn")
                                       ,to = c(1,2,3,4)) %>% as.numeric()

sae_dnn_fit <- sae.dnn.train(as.matrix(trainTransformed)
                             ,y = train_classes_probe
                             ,hidden = c(18)
                             ,activationfun = "sigm"
                             ,learningrate = 0.8
                             ,momentum = 0.5
                             ,learningrate_scale = 1
                             ,output = "sigm"
                             ,sae_output = "linear"
                             ,numepochs = 3
                             ,batchsize = 100
                             ,hidden_dropout = 0
                             ,visible_dropout = 0)

test_classes_probe <- plyr::mapvalues(test_classes
                                       ,from = c("Move-Forward", "Slight-Right-Turn", "Sharp-Right-Turn", "Slight-Left-Turn")
                                       ,to = c(1,2,3,4)) %>% as.numeric()

sae_dnn_pred <- nn.predict(sae_dnn_fit, as.matrix(testTransformed))


# Exploring SAENET package

# you will find out how autoencoders can be "stacked" in a greedy layerwise fashion for pretraining (initializing) the weights of a deep network.

# what we do: we compare the classification results of a neural net with no weight initiliazitation to the result of a neural nets which has weight initilialization with sae.

# Sparse Autoencoder for Automatic Learning of Representative Features from Unlabeled Data

# Build a stacked Autoencoder
output <- SAENET.train(
  as.matrix(iris[1:100,1:4])
  ,n.nodes = c(3)
  ,lambda = 1e-5
  ,beta = 1e-5
  ,rho = 0.01
  ,epsilon = .01
  )

class(output)
# Obtain the compressed representation of new data for specified layers
# from a stacked autoencoder
pred <- SAENET.predict(
  output
  ,as.matrix(iris[101:150,1:4])
  ,layers = c(3)
  )
class(pred)
print(pred)

prediction <- pred[[1]]$X.output
print(prediction)
