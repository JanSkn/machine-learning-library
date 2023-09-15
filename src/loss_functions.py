import numpy as np

def mse(y_true, y_pred):
    """
    Calculates the Mean Squared Error 
    between the estimated values and the actual value.
    """
    return np.mean((y_pred - y_true)**2)                # 1/n * ∑ (y_i - y_i*)^2

def mse_derivative(y_true, y_pred):
    """
    Calculates the derivative (∂E/∂Y) of the Mean Squared Error 
    between the estimated values and the actual value.
    """
    return 2 * (y_true - y_pred) / np.size(y_true)      # 2/n * (Y - Y*)

def cross_entropy_loss(y_true, y_pred):
    """
    Calculates the Cross Entropy Loss 
    between the estimated values and the actual value.
    """
    return -np.sum(y_true * np.log(y_pred))             # −∑ y_i * log(y_i*)

def cross_entropy_loss_derivative(y_true, y_pred):
    return -np.log(y_pred)                               # ??? müsste stimmen