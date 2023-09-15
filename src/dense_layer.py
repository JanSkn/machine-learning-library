import numpy as np
from layer import Layer

class Dense_layer(Layer):
    def  __init__(self, input_size: int, output_size: int):
        self.weights = np.random.rand(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)

    def forward_pass(self, x: np.array):
        """
        Receives its input (x) from the previous layer
        and provides an output for the next layer.
        """
        self.x = x                                          # store input to use it in backpropagation           
        return np.dot(self.weights, self.x) + self.bias     # y = Wx + b
    
    def backpropagation(self, output_gradient: np.array, learning_rate=0.1):
        """
        Updates the parameters depending on the derivatives.

        output_gradient: input from the next layer (output = ∂E/∂Y)
        learning_rate: hyperparameter to determine the step size of updates

        returns derived output for the predecessor
        """
        self.weights -= learning_rate * np.dot(output_gradient, self.x.T)       # W := W - LR * ∂E/∂Y * x^T
        self.bias -= learning_rate * output_gradient                            # b := b - LR * ∂E/∂Y
        return np.dot(self.weights.T, output_gradient)                          # ∂E/∂x = W^T * ∂E/∂Y
    
