# allows import from parent folder
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
from keras.datasets import mnist
from network import predict, load, evaluate
from mnist_train import adjust_data

network = load("mnist.pkl")

(_, _), (x_test, y_test) = mnist.load_data()

x_test, y_test = adjust_data(x_test, y_test, 20)

x_data = [] 
y_data = []

for x, y in zip(x_test, y_test):
    output = predict(x, network)
    x_data.append(np.argmax(output))
    y_data.append(np.argmax(y))
    # argmax returns index of the highest value, thus returns the number with the highest prediction
    print(f"Prediction: {np.argmax(output)}, Real: {np.argmax(y)}")

evaluation_data = (np.array(x_data).T, np.array(y_data).T)
print(evaluate(evaluation_data[0], evaluation_data[1]))