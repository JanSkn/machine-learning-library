import numpy as np
import pandas as pd
import dill
from typing import BinaryIO

def evaluate(y_pred: np.ndarray, y_true: np.ndarray ) -> list[tuple]:
        """
        Evaluates the model by calculating 
        Accuracy, Precision, Recall and F1-Score.

        Parameters:
            :y_pred (numpy.ndarray): Prediction of the model
            :y_true (numpy.ndarray): Real output

        Returns:
            List of tuples of the parameters.
            First tuple index: Label, second tuple index: Its measures.
        """
        labels = set()
        result = []
        
        for label in y_true: 
            if label not in labels:     # don't consider duplicates
                labels.add(label)
                tp = tn = fp = fn = accuracy = precision = recall = f1_score = 0

                for i in range(len(y_true)):
                    if y_true[i] == label:
                        if y_true[i] == y_pred[i]:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        if y_pred[i] == label:
                            fn += 1
                        else:
                            tn += 1

                if tp + tn + fp + fn > 0:
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                if tp + fn > 0:
                    recall = tp / (tp + fn)
                if precision + recall > 0:
                    f1_score = (2 * precision * recall) / (precision + recall)
                
                result.append((label, {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-Score": f1_score}))
        
        return result

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
    
def min_max_normalization(data: np.ndarray) -> np.ndarray:
    """
    Normalizes Data in a range from 0 to 1 using minx-max Value of every Feature. 
    Real-world datasets often contain features that are 
    varying in degrees of magnitude, range, and units. 
    Therefore, perform feature scaling, otherwise some features dominate the calculations.

    Parameters:
        :data (numpy.npdarray): Input for machine learning algorithm

    Returns:
        Normalized data
    """
    #each row is a data point, each column symbolizes a feature
    #we need to find min and max for every feature
    min = np.amin(data, axis=0)
    max = np.amax(data, axis=0)
    normalized_data = np.ndarray(shape=data.shape)

    for (data_point, feature), value in np.ndenumerate(data):
        normalized_data[data_point, feature] = (value-min[feature])/(max[feature]-min[feature])

    return normalized_data

def z_normalization(data: np.ndarray) -> np.ndarray:
    """
    Normalizes Data in a range from 0 to 1 using Z-Scores.
    Real-world datasets often contain features that are 
    varying in degrees of magnitude, range, and units. 
    Therefore, perform feature scaling, otherwise some features dominate the calculations.

    Parameters:
        :data (numpy.npdarray): Input for machine learning algorithm

    Returns:
        Normalized data
    """
    #each row is a data point, each column symbolizes a feature
    #we need to find mean and standart derivation for every feature
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    normalized_data = np.ndarray(shape=data.shape)

    for (data_point, feature), value in np.ndenumerate(data):
        normalized_data[data_point, feature] = (value-mean[feature])/(std[feature])

    return normalized_data