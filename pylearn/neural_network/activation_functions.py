import numpy as np
from pylearn.neural_network.activation import Activation   

class Sigmoid(Activation):
    """
    Defines the Sigmoid activation function for a layer.
    Inherits from Activation and passes the function and derivative.
    """
    def __init__(self) -> None:
        function = lambda x: 1 / (1 + np.exp(-x))
        derivative = lambda x: function(x) * (1 - function(x))
        super().__init__(function, derivative)
    
class ReLU(Activation):
    """
    Defines the ReLU activation function for a layer.
    Inherits from Activation and passes the function and derivative.
    """
    def __init__(self) -> None:
        function = lambda x: np.maximum(0, x)           # change every element less than 0 to 0
        derivative = lambda x: np.where(x > 0, 1, 0)    # change every element greater than 0 to 1
        super().__init__(function, derivative)

class Tanh(Activation):
    """
    Defines the Tanh activation function for a layer.
    Inherits from Activation and passes the function and derivative.
    """
    def __init__(self) -> None:
        function = lambda x: np.tanh(x)
        derivative = lambda x: 1 - np.tanh(x)**2
        super().__init__(function, derivative)

class Softmax(Activation):
    """
    Defines the Softmax activation function for a layer.
    Inherits from Activation and passes the function and derivative.
    """
    def __init__(self) -> None:
        function = lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))      # stable softmax (x = x - np.max(x)): prevent overflow/underflow (float limits)
        derivative = lambda x: function(x) * (1 - function(x))
        super().__init__(function, derivative)





