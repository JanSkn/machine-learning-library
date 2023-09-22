import numpy as np
from layer import Layer

class Dense_layer(Layer):
    """
    Defines the dense layer structure (fully connected layer)
    to use the forward pass and backpropagation.
    
    Attributes:
    weights (numpy.ndarray): Weight matrix of the layer
    bias (numpy.array): Bias vector of the layer

    Methods:
    forward_pass(x)
    backpropagation(output_gradient, learning_rate)
    """
    def  __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward_pass(self, x: np.array) -> np.dot:
        """
        Receives its input (x) from the previous layer
        and provides an output for the next layer.

        x (numpy.array): Input vector of the previous layer
        """
        self.x = x                                          # store input to use it in backpropagation           
        return np.dot(self.weights, self.x) + self.bias     # y = Wx + b
    
    def backpropagation(self, output_gradient: np.array, learning_rate = 0.1) -> np.array:
        """
        Updates the parameters depending on the derivatives.

        output_gradient (numpy.array): input of the next layer (output = ∂E/∂Y)
        learning_rate (numpy.array, optional): hyperparameter to determine the step size of updates, default: 0.01
        """
        self.weights -= learning_rate * np.dot(output_gradient, self.x.T)       # W := W - LR * ∂E/∂Y * x^T
        self.bias -= learning_rate * output_gradient                            # b := b - LR * ∂E/∂Y
        return np.dot(self.weights.T, output_gradient)                          # ∂E/∂x = W^T * ∂E/∂Y
    
