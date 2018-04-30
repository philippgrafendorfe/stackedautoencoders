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