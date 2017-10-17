library(neuralnet)
library(SAENET)
library(autoencoder)
library(caret)

# Exploring SAENET package

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
