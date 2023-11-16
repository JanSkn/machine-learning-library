import numpy as np
from pylearn.neural_network.layer import Layer

class Dense_layer(Layer):
    """
    Defines the dense layer structure (fully connected layer)
    to use the forward pass and backpropagation.
    
    Attributes:
        :weights (numpy.ndarray): Weight matrix of the layer
        :bias (numpy.array): Bias vector of the layer
    """
    def  __init__(self, input_size: int, output_size: int) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Receives its input (x) from the previous layer
        and provides an output for the next layer.

        Parameters:
            :x (numpy.ndrray): Input vector of the previous layer

        Returns:
            Result as array
        """
        self.x = x                                          # store input to use it in backpropagation           
        return np.dot(self.weights, self.x) + self.bias     # y = Wx + b
    
    def backpropagation(self, output_gradient: np.ndarray, learning_rate=0.1) -> np.ndarray:
        """
        Updates the parameters depending on the derivatives.

        Parameters:
            :output_gradient (numpy.array): input of the next layer (output = ∂E/∂Y)
            :learning_rate (numpy.array, optional): hyperparameter to determine the step size of updates, default: 0.01

        Returns:
            Derivative of the function as array
        """
        self.weights -= learning_rate * np.dot(output_gradient, self.x.T)       # W := W - LR * ∂E/∂Y * x^T
        self.bias -= learning_rate * output_gradient                            # b := b - LR * ∂E/∂Y
        return np.dot(self.weights.T, output_gradient)                          # ∂E/∂x = W^T * ∂E/∂Y
    
