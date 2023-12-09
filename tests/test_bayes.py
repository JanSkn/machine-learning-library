import pytest
import numpy as np
import pandas as pd
from pylearn import GaussianNaiveBayes

def test_fit():
    X = np.array([[1, 2], [2, 3], [3, 4]])
    Y = np.array([0, 1, 0])
    model = GaussianNaiveBayes()
    model.fit(X, Y)

    assert len(model.classes) == 2  
    assert model.mean is not None  
    assert model.variance is not None  
    assert model.prior is not None  

def test_predict():
    X_train = np.array([[1, 2], [2, 3], [3, 4]])
    Y_train = np.array([0, 1, 0])
    model = GaussianNaiveBayes()
    model.fit(X_train, Y_train)

    X_test = np.array([1, 2])
    predictions = model.predict(X_test)
    assert predictions is not None  
    assert len(predictions) == 2  

def test__predict():
    model = GaussianNaiveBayes()
    model.classes = [0, 1]
    model.prior = pd.DataFrame([0.5, 0.5])
    model.mean = pd.DataFrame([[1, 2], [2, 3]])
    model.variance = pd.DataFrame([[1, 1], [1, 1]])

    prediction = model._predict((1, 2))
    # Check whether the prediction corresponds to a known class
    assert prediction in model.classes  

def test__gaussian_distribution():
    model = GaussianNaiveBayes()

    model.classes = [0, 1]
    model.mean = pd.DataFrame([[1, 2], [3, 4]])
    print(model.mean)
    model.variance = pd.DataFrame([[1, 2], [2, 2]])

    index = 0  # class 0
    x = (1.5, 2.5)  

    gaussian_dist = model._gaussian_distribution(index, x)
    # x = 1.5:
    #   mean = 1, standard deviation = 1
    # x = 2.5:
    #   mean = 2, standard deviation = sqrt(2)
    expected_values = [0.3520653267642995, 0.26500353234402857]

    # error if the objects are not equal up to desired tolerance rtol
    np.testing.assert_allclose(gaussian_dist, expected_values, rtol=1e-5)

def test_invalid_input():
    model = GaussianNaiveBayes()
    X_train = np.array([[1, 2], [2, 3], [3, 4]])
    with pytest.raises(AttributeError):             # different error when running directly?
        model.fit(None, None)   

def test_model_fitting():
    # check difference of train and test accuracy to see if the model is overfitting or underfitting

    np.random.seed(0)   # set seed to reproduce values
    X_train = np.random.rand(100, 2)  # 100 samples with 2 features each
    Y_train = np.random.randint(0, 2, 100)  # 100 labels, 0 or 1

    np.random.seed(1)   # set seed to reproduce values
    X_test = np.random.rand(100, 2) 
    Y_test = np.random.randint(0, 2, 100) 

    def accuracy_score(true_labels, predicted_labels):
        correct_predictions = sum(true_labels == predicted_labels)
        total_predictions = len(true_labels)
        return correct_predictions / total_predictions

    model = GaussianNaiveBayes()
    model.fit(X_train, Y_train)
    train_accuracy = accuracy_score(Y_train, np.array(model.predict(X_train)).T[0])      
    test_accuracy = accuracy_score(Y_test, np.array(model.predict(X_test)).T[0])

    assert abs(train_accuracy - test_accuracy) < 0.15

def test_model_performance():
    np.random.seed(0)   # set seed to reproduce values
    X_train = np.random.rand(100, 2)  # 100 samples with 2 features each
    Y_train = np.random.randint(0, 2, 100)  # 100 labels, 0 or 1

    np.random.seed(1)   # set seed to reproduce values
    X_test = np.random.rand(100, 2) 
    Y_test = np.random.randint(0, 2, 100) 

    model = GaussianNaiveBayes()
    model.fit(X_train, Y_train)

    predictions = model.predict(X_test)

    precision = precision_score(Y_test, predictions)                        # TODO
    recall = recall_score(Y_test, predictions)
    f1 = f1_score(Y_test, predictions)

    assert precision > 0.7
    assert recall > 0.7
    assert f1 > 0.7
