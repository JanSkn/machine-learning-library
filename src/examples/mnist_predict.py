# allows import from parent folder
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from keras.datasets import mnist
from network import predict, load
from mnist_train import adjust_data

network = load("mnist.pkl")

(_, _), (x_test, y_test) = mnist.load_data()

x_test, y_test = adjust_data(x_test, y_test, 20)

for x, y in zip(x_test, y_test):
    output = predict(x, network)
    # argmax returns index of the highest value, thus returns the number with the highest prediction
    print(f"Prediction: {np.argmax(output)}, Real: {np.argmax(y)}")

