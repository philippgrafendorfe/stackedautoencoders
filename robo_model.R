library(neuralnet)
library(SAENET)
library(autoencoder)
library(caret)

# data exploration
df <- data.table::fread(input =  '../Data/sensor_readings_24.csv', data.table = F, stringsAsFactors = T)
df <- plyr::rename(df, c('V25' = 'Direction'))
table(df$Direction)
prop.table(table(df$Direction))

train_matrix_nolabel <- as.matrix(df[,1:length(df) - 1])
train_df_labeled <- df

nl=3 ## number of layers (default is 3: input, hidden, output)
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
a <- caret::createDataPartition(df$Direction, p = 0.8, list = F)
training <- df[a,]
test <- df[-a,]

formula <- as.formula(paste0("Direction ~ ", paste0(names(df)[1:24], collapse = "+")))
net <- neuralnet(formula = formula, data = training)

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
