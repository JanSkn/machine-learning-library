from activation import Activation
import numpy as np

class Sigmoid(Activation):
    def __init__(self):
        function = lambda x: 1 / (1 + np.exp(-x))
        derivative = lambda x: np.exp(-x) / (1 + np.exp(-x))**2
        super().__init__(function, derivative)
    
class ReLU(Activation):
    def __init__(self):
        function = lambda x: np.max(x)
        derivative = lambda x: 1 if x > 0 else 0
        super().__init__(function, derivative)

class Tanh(Activation):
    def __init__(self):
        function = lambda x: np.tanh(x)
        derivative = lambda x: 1 - np.tanh(x)**2
        super().__init__(function, derivative)

class Softmax(Activation):
    def __init__(self):
        function = lambda x: np.exp(x) / np.sum(np.exp(x)) 
        derivative = lambda x: function(x) * (1 - function(x))
        super().__init__(function, derivative)





