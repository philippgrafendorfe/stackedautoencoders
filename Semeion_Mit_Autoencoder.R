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


###################################################### Neuronales Netz ###################################
library(h2o)

local.h2o <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, nthreads=-1)
a <- caret::createDataPartition(semeion$digit, p = 0.8, list = F)

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
nrow(tsData)
############################################################# Autoencoder 




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
                             hidden = c(128, 64,128), 
                             epochs = 1000,
                             activation = "Tanh")
summary(model_nn)$variable_importance
ggplot(data=summary(model_nn),aes(x=variable, y=relative_importance)) + geom_point()

test_autoenc <- h2o.predict(model_nn, Semeion2[,-ncol(Semeion2)])
hilfe <- as.data.frame(test_autoenc)
num <- data.frame()
num2 <- data.frame()
index <- which(semeion$digit==0)
num <- rbind(num,semeion[index[1],])
num2 <- rbind(num2,hilfe[index[1],])
index <- which(semeion$digit==1)
num <- rbind(num,semeion[index[1],])
num2 <- rbind(num2,hilfe[index[1],])
index <- which(semeion$digit==2)
num <- rbind(num,semeion[index[1],])
num2 <- rbind(num2,hilfe[index[1],])
index <- which(semeion$digit==3)
num <- rbind(num,semeion[index[1],])
num2 <- rbind(num2,hilfe[index[1],])
index <- which(semeion$digit==4)
num <- rbind(num,semeion[index[1],])
num2 <- rbind(num2,hilfe[index[1],])
index <- which(semeion$digit==5)
num <- rbind(num,semeion[index[1],])
num2 <- rbind(num2,hilfe[index[1],])
index <- which(semeion$digit==6)
num <- rbind(num,semeion[index[1],])
num2 <- rbind(num2,hilfe[index[1],])
index <- which(semeion$digit==7)
num <- rbind(num,semeion[index[1],])
num2 <- rbind(num2,hilfe[index[1],])
index <- which(semeion$digit==8)
num <- rbind(num,semeion[index[1],])
num2 <- rbind(num2,hilfe[index[1],])
index <- which(semeion$digit==9)
num <- rbind(num,semeion[index[1],])
num2 <- rbind(num2,hilfe[index[1],])
num2$digit

# reverses (rotates the matrix)
rotate <- function(x) t(apply(x, 2, rev))

# Plot some of images
par(mfrow=c(4,3))
lapply(1:10, 
       function(x) image(
         rotate(matrix(unlist((num[x,-ncol(num)])),nrow = 16, byrow = TRUE)),
         col=grey.colors(255),
         xlab=(num[x,ncol(num)])
       )
)

# Plot some of images
par(mfrow=c(4,3))
lapply(1:10, 
       function(x) image(
         rotate(matrix(unlist(num2[x,]),nrow = 16, byrow = TRUE)),
         col=grey.colors(255),
         xlab=(num[x,ncol(num)])
       )
)

#############################################################
hilfe <- as.data.frame(test_autoenc)
df <- cbind(hilfe,semeion[,ncol(semeion)])
df <- as.data.frame(df)
a <- caret::createDataPartition(as.vector(df[,ncol(df)]), p = 0.8, list = F)

train.semeion <- data.frame(df[a,1:ncol(df)])
test.semeion <- data.frame(df[-a,1:ncol(df)])

trData<-as.h2o(train.semeion)
tsData<-as.h2o(test.semeion)

res.dl <- h2o.deeplearning(x = 1:(ncol(trData)-1), y = ncol(trData), trData, activation = "Tanh", 
                           hidden=rep(160,5),epochs = 20, nfold = 5)
summary(res.dl)


pred.dl<-h2o.predict(object=res.dl, newdata=tsData[,-ncol(tsData)])
pred.dl.df<-as.data.frame(pred.dl)
summary(pred.dl)
test_labels<-data.frame(test.semeion[,ncol(test.semeion)])
#calculate number of correct prediction
table(test_labels[,1],pred.dl.df[,1])
sum(diag(table(test_labels[,1],pred.dl.df[,1])))
nrow(tsData)

