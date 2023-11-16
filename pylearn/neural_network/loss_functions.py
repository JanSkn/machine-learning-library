import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error 
    between the estimated values and the actual value.

    Parameters:
        y_true (numpy.ndarray): Correct output
        y_pred (numpy.ndarray): Predicted output

    Returns:
        Error as float
    """
    return np.mean((y_pred - y_true)**2)                # 1/n * ∑ (y_i* - y_i)^2

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculates the derivative (∂E/∂Y) of the Mean Squared Error (MSE)
    between the estimated values and the actual value.

    Parameters:
        :y_true (numpy.ndarray): Correct output
        :y_pred (numpy.ndarray): Predicted output

    Returns:
        Derivative of MSE as array
    """
    return 2 * (y_pred - y_true) / np.size(y_true)      # 2/n * (Y* - Y)

def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Cross Entropy Loss 
    between the estimated values and the actual value.

    Parameters:
        :y_true (numpy.ndarray): Correct output
        :y_pred (numpy.ndarray): Predicted output

    Returns:
        Error as float
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)      # stable cross entropy: clip predicted values to avoid values close to 0

    return -np.sum(y_true * np.log(y_pred))             # −∑ y_i * log(y_i*)

def cross_entropy_loss_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculates the derivative (∂E/∂Y) of the Cross Entropy Loss
    between the estimated values and the actual value.

    Parameters:
        :y_true (numpy.ndarray): Correct output
        :y_pred (numpy.ndarray): Predicted output

    Returns:
        Derivative of CE as array
    """
    return -np.log(y_pred)        
                  
