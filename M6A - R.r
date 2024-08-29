# Install necessary packages if not already installed
if (!requireNamespace("keras", quietly = TRUE)) {
  install.packages("keras")
}
library(keras)

# Load the Fashion MNIST dataset
fashion_mnist <- dataset_fashion_mnist()
c(x_train, y_train) %<-% fashion_mnist$train
c(x_test, y_test) %<-% fashion_mnist$test

# Preprocess the data
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

# Build the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile the model
model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = 'accuracy'
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 10, batch_size = 64, validation_split = 0.2
)

# Evaluate the model
score <- model %>% evaluate(x_test, y_test)
cat('Test accuracy:', score$acc)

# Save the model
save_model_hdf5(model, "fashion_mnist_cnn.h5")


# Make predictions on test data
predictions <- model %>% predict(x_test)

# Function to plot the images along with predictions
plot_image <- function(predictions_array, true_label, img){
  plot(as.raster(img, max = 1), main = paste("Predicted:", which.max(predictions_array), "Actual:", true_label), col.main = ifelse(which.max(predictions_array) == true_label, "blue", "red"))
}

# Plot the first two test images and their predicted labels
par(mfrow=c(1,2))
for (i in 1:2) {
  plot_image(predictions[i,], which.max(y_test[i,]), x_test[i,,,drop=FALSE])
}
