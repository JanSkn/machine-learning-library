import numpy as np
from pylearn.neural_network.layer import Layer

class Activation(Layer):
    """
    Defines the activation functions for a layer.
    Inherits from Layer to compute forward pass and
    backpropagation with the activation function.
    
    Attributes:
        :function (function): Activation function
        :derivative (function): Derivative of the activation function
    """
    def __init__(self, function: classmethod, derivative: classmethod) -> None:
        self.function = function
        self.derivative = derivative

    def forward_pass(self, x: np.ndarray) -> np.ndarray:
        """
        Takes the input (x) of the layer and outputs f(x).

        Parameters:
            :x (numpy.ndarray): Input vector of the previous layer

        Returns:
            Result as array
        """
        self.x = x                          # store input to use it in backpropagation                  
        return self.function(x)             # y = f(x)
    
    def backpropagation(self, output_gradient: np.ndarray, learning_rate: int) -> np.ndarray:
        """
        Derived output for the predecessor after activation.

        Parameters:
            :output_gradient (numpy.ndarray): input of the next layer (output = ∂E/∂Y)

        Returns:
            Derivative of the function as array
        """
        return output_gradient * self.derivative(self.x)            # ∂E/∂X = ∂E/∂Y * f'(x) 