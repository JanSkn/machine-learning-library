import numpy as np
import time
import warnings

class NeuralNetwork:
    @staticmethod
    def fit(X: np.ndarray, Y: np.ndarray, network: list, loss_function: classmethod, loss_derivative: classmethod, epochs=1000, learning_rate=0.1, log_error=False, log_details=False, log_duration=True) -> None:
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
        warnings.warn(f"Parameters X and Y are of shape {X.shape} and {Y.shape}. This will be deprecated in the next version.", DeprecationWarning, stacklevel=2)

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

    @staticmethod
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