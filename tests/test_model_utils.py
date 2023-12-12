import numpy as np
from pylearn import accuracy, precision, recall, f1_score, min_max_normalization, z_normalization

def test_accuracy():
    y_true = np.array([1, 2, 1, 2])
    y_pred = np.array([1, 1, 1, 2])

    accuracies = accuracy(y_true, y_pred)
    expected_accuracies = [0.75, 0.75]  
    assert accuracies == expected_accuracies

    avg_accuracy = accuracy(y_true, y_pred, average=True)
    expected_avg_accuracy = 0.75  
    assert avg_accuracy == expected_avg_accuracy

def test_precision():
    y_true = np.array([1, 2, 1, 2])
    y_pred = np.array([1, 1, 1, 2])

    precisions = precision(y_true, y_pred)
    expected_precisions = [0.67, 1.0]  
    np.testing.assert_array_almost_equal(precisions, expected_precisions, decimal=2)

    avg_precision = precision(y_true, y_pred, average=True)
    expected_avg_precision = 0.83  
    assert round(avg_precision, 2) == expected_avg_precision

def test_recall():
    y_true = np.array([1, 2, 1, 2])
    y_pred = np.array([1, 1, 1, 2])

    recalls = recall(y_true, y_pred)
    expected_recalls = [1.0, 0.5]  
    np.testing.assert_array_almost_equal(recalls, expected_recalls, decimal=2)

    avg_recall = recall(y_true, y_pred, average=True)
    expected_avg_recall = 0.75  
    assert round(avg_recall, 2) == expected_avg_recall
    
def test_f1_score():
    y_true = np.array([1, 2, 1, 2])
    y_pred = np.array([1, 1, 1, 2])

    f1_scores = f1_score(y_true, y_pred)
    expected_f1_scores = [0.8, 0.67] 
    np.testing.assert_array_almost_equal(f1_scores, expected_f1_scores, decimal=2)

    avg_f1_score = f1_score(y_true, y_pred, average=True)
    expected_avg_f1_score = 0.73  
    assert round(avg_f1_score, 2) == expected_avg_f1_score

def test_min_max_normalization():
    data = np.array([[1, 2], [3, 4]])
    normalized_data = min_max_normalization(data)

    expected_normalized_data = np.array([[0, 0], [1, 1]])
    np.testing.assert_array_almost_equal(normalized_data, expected_normalized_data, decimal=2)

def test_z_normalization():
    data = np.array([[1, 2], [3, 4]])
    normalized_data = z_normalization(data)

    expected_normalized_data = np.array([[-1, -1], [1, 1]])
    np.testing.assert_array_almost_equal(normalized_data, expected_normalized_data, decimal=2)