# allows import from different folder
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from keras.datasets import mnist
import keras.utils
from neural_network.dense_layer import Dense_layer
from neural_network.activation_functions import Tanh
from neural_network.loss_functions import mse, mse_derivative
from neural_network.network import train, save 

# load training data
# x_train: image of 28 x 28 pixels, y_train: output of the number (0-9)
(x_train, y_train), (_, _) = mnist.load_data()

input_length = 784
training_data_size = 10000
testing_data_size = 20

def adjust_data(X: np.ndarray, Y: np.ndarray, data_size: int) -> tuple:
    """
    Transforms input matrix to input vector and normalize input.
    Transforms output to one hot.

    X (numpy.ndarray): Input matrix
    Y (numpy.array): Output vector
    data_size (int): Limits data size
    """
    X = X.astype("float32") / 255                       # normalize pixel values (0-255) to values between 0 and 1
    X = X.reshape(X.shape[0], input_length, 1)          # adjust input matrix (28 x 28) to vector (784)
    
    Y = keras.utils.to_categorical(Y)                   # transform regression output to one hot output
    Y = Y.reshape(Y.shape[0], 10, 1)                    # adjust vector

    return X[:data_size], Y[:data_size]                 # limit data size

if __name__ == "__main__":
    x_train, y_train = adjust_data(x_train, y_train, training_data_size)

    # create neural network
    network = [
        Dense_layer(input_length, 50),
        Tanh(),
        Dense_layer(50, 10),
        Tanh()
    ]

    # train and save the model
    train(x_train, y_train, network, mse, mse_derivative, epochs = 100, log_error = True, log_duration = True)
    save("mnist.pkl", network)

