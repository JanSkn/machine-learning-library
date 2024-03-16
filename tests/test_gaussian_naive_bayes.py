import pytest
import numpy as np
import pandas as pd
from pylearn import GaussianNaiveBayes, accuracy, precision, recall, f1_score

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

def test_model_fitting():
    # check difference of train and test accuracy to see if the model is overfitting or underfitting

    np.random.seed(0)   # set seed to reproduce values
    X_train = np.random.rand(100, 2)  # 100 samples with 2 features each
    Y_train = np.random.randint(0, 2, 100)  # 100 labels, 0 or 1

    np.random.seed(1)   # set seed to reproduce values
    X_test = np.random.rand(100, 2) 
    Y_test = np.random.randint(0, 2, 100) 

    model = GaussianNaiveBayes()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    Y_pred = np.array(Y_pred).T[0]
    train_accuracy = accuracy(Y_train, Y_pred, average=True)      
    test_accuracy = accuracy(Y_test, Y_pred, average=True)

    assert abs(train_accuracy - test_accuracy) < 0.15

def test_model_performance():
    def generate_test_data(seed: int):
        np.random.seed(seed)  # set seed to reproduce values
        X = np.random.rand(100, 2)

        # 1 if sum of both features is 1, else 0
        Y = (X[:, 0] + X[:, 1] > 1).astype(int)
        data = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
        data['Label'] = Y

        return data

    training = generate_test_data(0)
    X_train = training[["Feature1", "Feature2"]]
    Y_train = training["Label"]

    testing = generate_test_data(1)
    X_test = testing[["Feature1", "Feature2"]]
    Y_test = testing["Label"]

    model = GaussianNaiveBayes()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    # average scores
    prec = precision(Y_test, Y_pred, average=True)                     
    rec = recall(Y_test, Y_pred, average=True)
    f1 = f1_score(Y_test, Y_pred, average=True)

    assert prec > 0.85
    assert rec > 0.85
    assert f1 > 0.85