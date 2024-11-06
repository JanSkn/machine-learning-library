# Example for neural network

# allows import from different folder
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
from examples.mnist_train import adjust_data, testing_data_size
from pylearn import NeuralNetwork, load, accuracy, precision, recall, f1_score

network = load("mnist.pkl")

(_, _), (x_test, y_test) = load("data/mnist.npy")

x_test, y_test = adjust_data(x_test, y_test, testing_data_size)

x_data = [] 
y_data = []

print(f"Prediction of the first {testing_data_size} numbers:\n")
for x, y in zip(x_test, y_test):
    output = NeuralNetwork.predict(x, network)
    x_data.append(np.argmax(output))
    y_data.append(np.argmax(y))
    # argmax returns index of the highest value, thus returns the number with the highest prediction
    print(f"Prediction: {np.argmax(output)}, Real: {np.argmax(y)}")
print()

index = int(input(f"Enter an index (0-{testing_data_size - 1}) of the test data to get a visualized prediction: "))
if index >= 0 and index < testing_data_size:
    img = x_test[index]
    plt.imshow(img.reshape(28, 28), cmap = "Greys")
    plt.title(f"Prediction: {np.argmax(NeuralNetwork.predict(x_test[index], network))}, Real: {np.argmax(y_test[index])}")
    plt.show()
else:
    print("Invalid index.")

print()
print("Accuracy:", accuracy(y_data, x_data))
print("Precision:", precision(y_data, x_data))
print("Recall:", recall(y_data, x_data))
print("F1 Score:", f1_score(y_data, x_data))