import numpy as np
from neural_network.layer import Layer

class Activation(Layer):
    """
    Defines the activation functions for a layer.
    Inherits from Layer to compute forward pass and
    backpropagation with the activation function.
    
    Attributes:
    function (function): Activation function
    derivative (function): Derivative of the activation function

    Methods:
    forward_pass(x)
    backpropagation(output_gradient, learning_rate)
    """
    def __init__(self, function: classmethod, derivative: classmethod) -> None:
        self.function = function
        self.derivative = derivative

    def forward_pass(self, x: np.array):
        """
        Takes the input (x) of the layer and outputs f(x).

        x (numpy.array): Input vector of the previous layer
        """
        self.x = x                          # store input to use it in backpropagation                  
        return self.function(x)             # y = f(x)
    
    def backpropagation(self, output_gradient: np.array, learning_rate: int) -> np.array:
        """
        Returns derived output for the predecessor after activation.

        output_gradient (numpy.array): input of the next layer (output = ∂E/∂Y)
        """
        return output_gradient * self.derivative(self.x)            # ∂E/∂X = ∂E/∂Y * f'(x) 