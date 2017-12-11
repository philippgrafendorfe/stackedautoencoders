
library(mxnet)
library(RCurl)
library(caret)
library(e1071)

# specify the URL
urlfile <-"http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data"
# download the file
downloaded <- getURL(urlfile, ssl.verifypeer=FALSE)
# treat the text data as a steam so we can read from it
connection <- textConnection(downloaded)
# parse the downloaded data as CSV
semeion <- read.csv(connection, header=FALSE, sep= " ")
# preview the first 5 rows
head(semeion)

semeion$digit <- semeion[,ncol(semeion)]

semeion$digit[which(as.logical(semeion$V266), arr.ind=TRUE)] = 9
semeion$digit[which(as.logical(semeion$V265), arr.ind=TRUE)] = 8
semeion$digit[which(as.logical(semeion$V264), arr.ind=TRUE)] = 7
semeion$digit[which(as.logical(semeion$V263), arr.ind=TRUE)] = 6
semeion$digit[which(as.logical(semeion$V262), arr.ind=TRUE)] = 5
semeion$digit[which(as.logical(semeion$V261), arr.ind=TRUE)] = 4
semeion$digit[which(as.logical(semeion$V260), arr.ind=TRUE)] = 3
semeion$digit[which(as.logical(semeion$V259), arr.ind=TRUE)] = 2
semeion$digit[which(as.logical(semeion$V258), arr.ind=TRUE)] = 1
semeion$digit[which(as.logical(semeion$V257), arr.ind=TRUE)] = 0
semeion <- semeion[,c(0:256,268)]
table(semeion[,257])

semeion[,ncol(semeion)]<-as.factor(semeion[,ncol(semeion)])

# c <- c(0,1,2,3,0,1,2,3)
# which(as.logical(c,arr.ind=T))

#################################################### MxNetR ###############################################

a <- caret::createDataPartition(semeion$digit, p = 0.8, list = F)

train.semeion <- data.matrix(semeion[a,1:ncol(semeion)-1])
train.label.semeion <- semeion[a,ncol(semeion)]

test.semeion <- data.matrix(semeion[-a,1:ncol(semeion)-1])
test.label.semeion <- data.matrix(semeion[-a,ncol(semeion)])
table(train.label.semeion)
table(test.label.semeion)


data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="sigm1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="sigm2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc2, name="softmax")

mx.set.seed(0)

model <- mx.model.FeedForward.create(softmax, X=train.semeion, y=train.label.semeion,
                                     num.round=10, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9,  eval.metric=mx.metric.accuracy)
preds <- 0
preds <- predict(model, test.semeion)
pred.label <- max.col(t(preds)) - 1
table(pred.label)
table(test.label.semeion)
confusionMatrix(pred.label,test.label.semeion)
table(pred.label, test.label.semeion)
sum(diag(table(pred.label, test.label.semeion)))
#### 292 von 314

#################################################### H2o ###########################################
library(h2o)

local.h2o <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads=-1)


train.semeion <- data.frame(semeion[a,1:ncol(semeion)])
test.semeion <- data.frame(semeion[-a,1:ncol(semeion)])

trData<-as.h2o(train.semeion)
tsData<-as.h2o(test.semeion)


## Step 4: Train the model
## Next is to train the model. For this experiment, 5 layers of 160 nodes each are used. 
## The rectifier used is Tanh and number of epochs is 20
## nfold = 5 5 fold cross validation 
## Optionally, the user can save the cross-validation predicted
## values (generated during cross-validation) by setting keep cross validation predictions
## parameter to true.
## c("Tanh", "TanhWithDropout", "Rectifier","RectifierWithDropout", "Maxout", "MaxoutWithDropout")
res.dl <- h2o.deeplearning(x = 1:(ncol(trData)-1), y = ncol(trData), trData, activation = "Tanh", 
                           hidden=rep(160,5),epochs = 20, nfold = 5)
summary(res.dl)

#use model to predict testing dataset
pred.dl<-h2o.predict(object=res.dl, newdata=tsData[,-ncol(tsData)])
pred.dl.df<-as.data.frame(pred.dl)
summary(pred.dl)
test_labels<-data.frame(test.semeion[,ncol(test.semeion)])
#calculate number of correct prediction
table(test_labels[,1],pred.dl.df[,1])
sum(diag(table(test_labels[,1],pred.dl.df[,1])))

#### 289 von 314 richtig -> 92.04 % act. Tanh
#### nfold = 5, 292 richtig von 314 -> 92.99 % act. Tanh
#### 291 von 314 richtig -> 92,68 %  act. Maxout
#### 287 von 312 richtig ->         act. Maxout nfold = 5


####################################### Plotting digits

par(mfrow=c(2,1))
rotate <- function(x) t(apply(x, 2, rev))
m = matrix(unlist(semeion[100,-ncol(semeion)]), nrow = 16, byrow = TRUE)
d<-rotate(matrix(unlist(semeion[100,-ncol(semeion)]),nrow = 16, byrow = TRUE))
# Plot that matrix
image(d,col=c("black","white"))


# Plot some of images
# par(mfrow=c(2,3))
# lapply(1:6, 
#        function(x) image(
#          rotate(matrix(unlist(train[x,-1]),nrow = 28, byrow = TRUE)),
#          col=grey.colors(255),
#          xlab=train[x,1]
#        )
# )
# 
# par(mfrow=c(1,1))
#######################################################################################
library(autoencoder)

df <- semeion[, 1:ncol(semeion)]
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

X.output <- as.data.frame(X.output)

rotate <- function(x) t(apply(x, 2, rev))
m = matrix(unlist(X.output[100,]), nrow = 16, byrow = TRUE)
d<-rotate(matrix(unlist(X.output[100,]),nrow = 16, byrow = TRUE))
# Plot that matrix
image(d,col=c("black","white"))

######################################## Autoencoder H20
Semeion2 <- as.h2o(semeion)
Semeion2 <- Semeion2[,-ncol(Semeion2)]
response <- "Class"
features <- setdiff(colnames(Semeion2), response)

model_nn <- h2o.deeplearning(x = features,
                             training_frame = Semeion2,
                             model_id = "model_nn",
                             autoencoder = TRUE,
                             reproducible = TRUE, #slow - turn off for real problems
                             ignore_const_cols = FALSE,
                             seed = 42,
                             hidden = c(10, 2, 10), 
                             epochs = 100,
                             activation = "Tanh")

test_autoenc <- h2o.predict(model_nn, tsData[,-ncol(tsData)])
rotate <- function(x) t(apply(x, 2, rev))
m = matrix(unlist(tsData[1,-ncol(tsData)]), nrow = 16, byrow = TRUE)
d<-rotate(matrix(unlist(tsData[1,-ncol(tsData)]),nrow = 16, byrow = TRUE))
# Plot that matrix
image(d,col=c("black","white"))


rotate <- function(x) t(apply(x, 2, rev))
m = matrix(unlist(test_autoenc[1,]), nrow = 16, byrow = TRUE)
d<-rotate(matrix(unlist(test_autoenc[1,]),nrow = 16, byrow = TRUE))
# Plot that matrix
image(d,col=c("black","white"))

