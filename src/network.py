import numpy as np
import dill
from typing import BinaryIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def train(X: np.array, Y: np.array, network: list, loss_function: classmethod, loss_derivative: classmethod, epochs = 1000, learning_rate = 0.1, log_error = False, log_details = True) -> None:
    """
    Trains the neural network by performing forward pass 
    and back propagation and also computes the error for every epoch.

    To log the parameters of the network: log_details
    To log the error of the network: log_error

    X (numpy.array): Training input
    Y (numpy.array): Training output
    network (list[Layer]): List of network layers
    loss_function (classmethod): Loss function to compute the error
    loss_derivative (classmethod): Derivative of the loss function to perform backpropagation
    epochs (int, optional): Hyperparameter, number of learning iterations, default: 1000
    learning_rate (int, optional): Hyperparameter, default: 0.01
    log_error (bool, optional): Logs the error of each iteration, default: False
    log_details (bool, optional): Logs the parameters of each layer, default: True
    """
    for epoch in range(epochs):
        error = 0
        for x, y in zip(X, Y):
            output = x

            # Forward Pass
            for layer in network:
                output = layer.forward_pass(output)
        
            # Error for all data
            error += loss_function(y, output)

            # Backpropagation
            gradient = loss_derivative(y, output)       
            for layer in reversed(network):
                gradient = layer.backpropagation(gradient, learning_rate)

        error /= len(X)

        if log_error:
            print(f"Epoch {epoch + 1}/{epochs}, Error: {error}")

    print()

    if log_details:
        i = 0
        print(f"Error: {error}\n")
        for layer in network:
            if hasattr(layer, "weights") and hasattr(layer, "bias"):
                print(f"Weights of Layer {i}:\n {layer.weights}\n")
                print(f"Bias of Layer {i}:\n {layer.bias}\n")
                i += 1

def predict(X: np.array, network: list) -> np.array:
    """
    The neural network computes the output of a given input X.

    X (numpy.array): Training input
    network (list[Layer]): List of network layers
    """
    output = X
    for layer in network:
        output = layer.forward_pass(output) 

    return output

def save(file_name: str, network: list) -> None:
    """
    Saves the trained network to the storage.
    Recommended file ending: .pkl

    file_name (str): Name of the file
    network (list[Layer]): The network to save
    """
    with open(file_name, 'wb') as file:     # wb: writing in binary mode
        dill.dump(network, file)

def load(file_name: str) -> BinaryIO:
    """
    Loads the trained network from the storage.
    Recommended file ending: .pkl

    file_name (str): Name of the file
    """
    with open(file_name, 'rb') as file:     # rb: reading in binary mode
        return dill.load(file)

def plot(network, density = 25):
    """
    Takes points and plots their prediction of the neural network on a 3D graph.

    network (list[Layer]): List of network layers
    density (int, optional): Number of points per axis, default: 25
    """
    # TODO adjust to more than 2 data points
    if network[0].input_size > 2:
        print("TODO: Allow more than 2 data points as input")
        return

    points = []
    for x in np.linspace(0, 1, density):
        for y in np.linspace(0, 1, density):
            z = [[x], [y]]      
            for layer in network:
                z = layer.forward_pass(z)
            
            points.append([x, y, z[0,0]])

    points = np.array(points)

    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
    axes.set_title("Decision Boundary")
    
    figure.canvas.manager.full_screen_toggle()
    plt.show()