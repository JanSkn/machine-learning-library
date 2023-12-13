import numpy as np
import pandas as pd
import dill
from typing import BinaryIO

def accuracy(y_true: np.ndarray, y_pred: np.ndarray, average=False) -> float | list:
    """
    Evaluates the model by calculating Accuracy.

    Parameters:
        :y_true (numpy.ndarray): Real output
        :y_pred (numpy.ndarray): Prediction of the model
        :average (bool, optional): Return average of all class scores, default: False
        
    Returns:
        List of score of the classes.
    """
    unique_labels = sorted(np.unique(y_true))
    accuracy_scores = []

    for label in unique_labels:
        tp = sum((y_pred == label) & (y_true == label))
        tn = sum((y_pred != label) & (y_true != label))
        fp = sum((y_pred == label) & (y_true != label))
        fn = sum((y_pred != label) & (y_true == label))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        accuracy_scores.append(accuracy)

    if average:
        return sum(accuracy_scores) / len(accuracy_scores) 

    return accuracy_scores

def precision(y_true: np.ndarray, y_pred: np.ndarray, average=False) -> float | list:
    """
    Evaluates the model by calculating Precision.

    Parameters:
        :y_true (numpy.ndarray): Real output
        :y_pred (numpy.ndarray): Prediction of the model
        :average (bool, optional): Return average of all class scores, default: False
        
    Returns:
        List of score of the classes.
    """
    unique_labels = sorted(np.unique(y_true))
    precision_scores = []

    for label in unique_labels:
        tp = sum((y_pred == label) & (y_true == label))
        fp = sum((y_pred == label) & (y_true != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_scores.append(precision)

    if average:
        return sum(precision_scores) / len(precision_scores)

    return precision_scores

def recall(y_true: np.ndarray, y_pred: np.ndarray, average=False) -> float | list:
    """
    Evaluates the model by calculating Recall.

    Parameters:
        :y_true (numpy.ndarray): Real output
        :y_pred (numpy.ndarray): Prediction of the model
        :average (bool, optional): Return average of all class scores, default: False
        
    Returns:
        List of score of the classes.
    """
    unique_labels = sorted(np.unique(y_true))
    recall_scores = []

    for label in unique_labels:
        tp = sum((y_pred == label) & (y_true == label))
        fn = sum((y_pred != label) & (y_true == label))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_scores.append(recall)

    if average:
        return sum(recall_scores) / len(recall_scores)

    return recall_scores

def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average=False) -> float | list:
    """
    Evaluates the model by calculating F1 score.

    Parameters:
        :y_true (numpy.ndarray): Real output
        :y_pred (numpy.ndarray): Prediction of the model
        :average (bool, optional): Return average of all class scores, default: False
        
    Returns:
        List of score of the classes.
    """
    precision_scores = precision(y_true, y_pred)
    recall_scores = recall(y_true, y_pred)
    f1_scores = []

    for i in range(len(precision_scores)):
        prec = precision_scores[i]
        rec = recall_scores[i]
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        f1_scores.append(f1)

    if average:
        return sum(f1_scores) / len(f1_scores)

    return f1_scores

def min_max_normalization(data: np.ndarray, axis=0) -> np.ndarray:
    """
    Normalizes Data in a range from 0 to 1 using min-max value of every feature. 
    Perform feature scaling to prevent features from dominating the calculations.

    Parameters:
        :data (numpy.ndarray): Data to normalize
        :axis (int, optional): Axis to perform normalization, default: 0

    Returns:
        Normalized data
    """
    # each row is a data point, each column symbolizes a feature
    min = np.amin(data, axis=axis)
    max = np.amax(data, axis=axis)
    normalized_data = np.ndarray(shape=data.shape)

    for (data_point, feature), value in np.ndenumerate(data):
        normalized_data[data_point, feature] = (value-min[feature]) / (max[feature]-min[feature])

    return normalized_data

def z_normalization(data: np.ndarray, axis=0) -> np.ndarray:
    """
    Normalizes Data in a range from 0 to 1 using Z-Scores.
    Perform feature scaling to prevent features from dominating the calculations.

    Parameters:
        :data (numpy.ndarray): Data to normalize
        :axis (int, optional): Axis to perform normalization, default: 0

    Returns:
        Normalized data
    """
    # each row is a data point, each column symbolizes a feature
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis)
    normalized_data = np.ndarray(shape=data.shape)

    for (data_point, feature), value in np.ndenumerate(data):
        normalized_data[data_point, feature] = (value-mean[feature]) / (std[feature])

    return normalized_data

def to_one_hot(y: np.ndarray, num_of_classes=10):
    """
    Converts array of numbers to array of One Hots.

    Parameters:
        :y (numpy.ndarray): Array of numbers
        :num_of_classes (int, optional): Number of classes, default: 10

    Returns:
        One Hot representation of the numbers as numpy array
    """
    one_hot = np.zeros((len(y), num_of_classes))
    
    for i, label in enumerate(y):
        one_hot[i, label] = 1
    
    return one_hot

def save(file_name: str, model: object) -> None:
    """
    Saves the trained model to the storage.
    Recommended file ending: .pkl

    Parameters:
        :file_name (str): Name of the file
        :model (object): The model to save

    Returns:
        None
    """
    with open(file_name, 'wb') as file:     # wb: writing in binary mode
        dill.dump(model, file)

def load(file_name: str) -> BinaryIO:
    """
    Loads the trained model from the storage.
    Recommended file ending: .pkl

    Parameters:
        :file_name (str): Name of the file

    Returns:
        The file
    """
    with open(file_name, 'rb') as file:     # rb: reading in binary mode
        return dill.load(file)