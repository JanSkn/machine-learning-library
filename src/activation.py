from layer import Layer

class Activation(Layer):
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    def forward_pass(self, x):
        """
        Takes the input (x) of the layer and outputs f(x).
        """
        self.x = x                          # store input to use it in backpropagation                  
        return self.function(x)             # y = f(x)
    
    def backpropagation(self, output_gradient, learning_rate):
        """
        Returns derived output for the predecessor after activation.
        """
        return output_gradient * self.derivative(self.x)            # ∂E/∂X = ∂E/∂Y * f'(x) 