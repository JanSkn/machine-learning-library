import numpy as np
from pylearn import Dense_layer, Tanh, mse, mse_derivative, NeuralNetwork  

def test_fit():
    # XOR as example
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    X = X.reshape(X.shape[0], 2, 1) 
    Y = np.array([[0], [1], [1], [0]])
    Y = Y.reshape(Y.shape[0], 1, 1)  

    network = [
        Dense_layer(2, 3),
        Tanh(),
        Dense_layer(3, 1),
        Tanh()
    ]

    NeuralNetwork.fit(X, Y, network, mse, mse_derivative, epochs=1000, log_duration=False)
    prediction = NeuralNetwork.predict([[0], [1]], network)
    real = 1
    assert np.where(prediction >= 0.5, 1, 0) == real    # round prediction to 0 or 1