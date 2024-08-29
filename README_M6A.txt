                                  Fashion MNIST Classification Using Convolutional Neural Network

Overview:

This project implements a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset. The purpose of this Project is to train a model that can accurately predict various categories of clothing such as: shoes, T-shirts, etc.

The implementation is provided in both Python and R, using the Keras library.

Requirements:

1) Python 3.x.
2) R 3.x or higher.
3) Keras library for Python & R.
4) TensorFlow.
5) NumPy.
6) Matplotlib.

Setup:

For Jupyter Notebook in PyCharm Professional:
1. Install the required packages:
   ```bash
   pip install keras tensorflow numpy matplotlib

For R:
Install R from R Project.
Install the required packages
    install.packages("keras")
    library(keras)

Running The Code
1) For Jupyter Notebook:

- Open the Jupyter notebook "module_6_assignment_fashion_mnist" in PyCharm Professional and run the code.
- Execute all cells to train the CNN and make predictions.

2) For R:
- Open the R script .
- Execute all lines to train the CNN and make predictions.

Training:
- The model is compiled with "Adam Optimizer" and trained using the "Categorical_Crossentropy" loss function. Training runs for 10 Epochs. Test accuracy is 91%.


Output:
The predictions for the first two images from the test set will be displayed alongside the true labels.

Note:
Ensure that the Keras library is correctly configured to use the appropriate backend (TensorFlow).
The dataset is automatically downloaded when running the scripts for the first time.

This README file provides instructions for setting up and running the code, as well as interpreting the results.