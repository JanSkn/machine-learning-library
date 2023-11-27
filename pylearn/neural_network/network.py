import numpy as np
import time
import matplotlib.pyplot as plt

def train(X: np.ndarray, Y: np.ndarray, network: list, loss_function: classmethod, loss_derivative: classmethod, epochs=1000, learning_rate=0.1, log_error=False, log_details=False, log_duration=True) -> None:
    """
    Trains the neural network by performing forward pass 
    and back propagation and also computes the error for every epoch.

    To log: log_details, log_error
    
    Parameters:
        :X (numpy.array): Training input
        :Y (numpy.array): Training output
        :network (list[Layer]): List of network layers
        :loss_function (classmethod): Loss function to compute the error
        :loss_derivative (classmethod): Derivative of the loss function to perform backpropagation
        :epochs (int, optional): Hyperparameter, number of learning iterations, default: 1000
        :learning_rate (int, optional): Hyperparameter, default: 0.01
        :log_error (bool, optional): Logs the error of each iteration, default: False
        :log_details (bool, optional): Logs the parameters of each layer, default: False
        :log_duration (bool, optional): Logs the duration of the training, default: True
    
    Returns:
        None
    """
    start = time.time()

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

    end = time.time()

    print()

    if log_details:
        i = 0
        print(f"Error: {error}\n")
        for layer in network:
            if hasattr(layer, "weights") and hasattr(layer, "bias"):
                print(f"Weights of Layer {i}:\n {layer.weights}\n")
                print(f"Bias of Layer {i}:\n {layer.bias}\n")
                i += 1

    if log_duration:
        print(f"Duration of training: {end - start}\n")

def predict(X: np.ndarray, network: list) -> np.ndarray:
    """
    The neural network computes the output of a given input X.

    Parameters:
        :X (numpy.array): Testing input
        :network (list[Layer]): List of network layers

    Returns:
        The predicted output as an array
    """
    output = X
    for layer in network:
        output = layer.forward_pass(output) 

    return output

def evaluate(X: np.ndarray, Y: np.ndarray) -> list[tuple]:
    """
    Evaluates the neural network by calculating 
    Accuracy, Precision, Recall and F1-Score.

    Parameters:
        :X (np.array): Prediction of the network
        :Y (np.array): Real output

    Returns:
        List of tuples of the parameters.
        First tuple index: Label, second tuple index: Its measures.
    """
    labels = set()
    result = []

    for label in Y: 
        if label not in labels:     # don't consider duplicates
            labels.add(label)
            tp = tn = fp = fn = accuracy = precision = recall = f1_score = 0

            for i in range(len(Y)):
                if Y[i] == label:
                    if Y[i] == X[i]:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if X[i] == label:
                        fn += 1
                    else:
                        tn += 1

            if tp + tn + fp + fn > 0:
                accuracy = (tp + tn) / (tp + tn + fp + fn)
            if tp + fp > 0:
                precision = tp / (tp + fp)
            if tp + fn > 0:
                recall = tp / (tp + fn)
            if precision + recall > 0:
                f1_score = (2 * precision * recall) / (precision + recall)
            
            result.append((label, {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1_score}))
    
    return result

def plot(network: list, density=25) -> None:
    """
    Takes points and plots their prediction of the neural network on a 3D graph.

    Parameters:
        :network (list[Layer]): List of network layers
        :density (int, optional): Number of points per axis, default: 25

    Returns:
        None
    """
    # TODO adjust to arbitrary dimensions
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