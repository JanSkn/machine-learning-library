=====================================
Usage
=====================================

|

Installation
============

PyLearn can be installed via pip:

.. code-block:: bash

    pip install pylearn-ml

.. note::

    For installation: `pylearn-ml`
    
    For import: `pylearn`

|


Importing PyLearn
-----------------

To use PyLearn, import it as follows:

.. code-block:: python

    import pylearn as pl

|
|

Usage
============

|

Save and Load Models
--------------------

To save and load models, you can use the following functions.

This avoids training your model again with every program execution.

.. code-block:: python

    pl.save(model, filename)
    pl.load(filename)

|

**Example**

.. code-block:: python
    
    network = [
      ...
    ]
    
    # train model
    pl.NeuralNetwork.fit(...)
    pl.save(network, "models/network.pkl")

    ...

    # test model
    network = pl.load("models/network.pkl")
    pl.NeuralNetwork.predict(...)

|
|

Normalization
-------------

Functions for normalization:

.. code-block:: python

    pl.min_max_normalization(data)
    pl.z_normalization(data)

|

**Example**

.. code-block:: python

    # Data to normalize
    data = [2, 5, 10, 15, 20]

    min_max_norm = pl.min_max_normalization(data)
    z_norm = pl.z_normalization(data)

|
|

Evaluation Metrics
------------------

To evaluate classification models, you can use:

.. code-block:: python

    pl.accuracy(true_labels, predicted_labels)
    pl.precision(true_labels, predicted_labels)
    pl.recall(true_labels, predicted_labels)
    pl.f1_score(true_labels, predicted_labels)

|

**Example**

.. code-block:: python

    # train any model
    model = pl.Model()
    model.fit(features, output)

    # predict
    predictions = model.predict(features)

    # evaluate
    accuracy = pl.accuracy(output, predictions)
    precision = pl.precision(output, predictions)
    recall = pl.recall(output, predictions)
    f1 = pl.f1_score(output, predictions)

|
|

Mean Squared Error
------------------

For regression tasks, you can calculate mean squared error:

.. code-block:: python

    pl.mean_squared_error(true_values, predicted_values)

|

**Example**

.. code-block:: python

    # true values and predicted values
    true_values = [2, 4, 6, 8, 10]
    predicted_values = [3, 5, 7, 9, 11]

    # calculate MSE
    mse = pl.mean_squared_error(true_values, predicted_values)

|
|

Sum of Squared Errors
----------------------

Additionally, you can compute the sum of squared errors:

.. code-block:: python

    pl.sum_of_squared_errors(true_values, predicted_values)

|

**Example**

.. code-block:: python

    # true values and predicted values
    true_values = [2, 4, 6, 8, 10]
    predicted_values = [3, 5, 7, 9, 11]

    # calculate MSE
    sse = pl.sum_of_squared_errors(true_values, predicted_values)

|
|

One-Hot Encoding
----------------

pylearn provides a function for one-hot encoding:

.. code-block:: python

    pl.to_one_hot(labels)

|

**Example**

.. code-block:: python

    # labels to encode
    labels = [0, 1, 2, 3, 4]

    # perform One-Hot Encoding
    encoded_labels = pl.to_one_hot(labels)

|
|
|
|

Classification
--------------

For classification tasks, PyLearn offers:

|

Gaussian Naive Bayes
~~~~~~~~~~~~~~~~~~~~

Gaussian Naive Bayes is a probabilistic classifier that assumes that the features are independent and follows a normal distribution.

|

Usage Example:

.. code-block:: python

    gnb = pl.GaussianNaiveBayes()
    gnb.fit(features, output)
    predictions = gnb.predict(features)

|

Multinomial Naive Bayes
~~~~~~~~~~~~~~~~~~~~~~~~

Multinomial Naive Bayes is suitable for classification with discrete features (e.g., word counts).

|

Usage Example:

.. code-block:: python

    mnb = pl.MultinomialNaiveBayes()
    mnb.fit(features, output)
    predictions = mnb.predict(features)

|
|

Clustering
----------

For clustering tasks, pylearn offers:

|

K-Means
~~~~~~~

K-Means is a popular clustering algorithm that partitions data into K clusters based on similarity.

|

Usage Example:

.. code-block:: python

    kmeans = pl.KMeans()
    kmeans.fit(points)
    clusters = kmeans.assigned_clusters()

|

K-Medoids
~~~~~~~~~

K-Medoids is similar to K-Means but uses actual data points (medoids) as cluster centers.

|

Usage Example:

.. code-block:: python

    kmedoids = pl.KMedoids()
    kmedoids.fit(points)
    clusters = kmedoids.assigned_clusters()
    kmedoids.rename(old_cluster_id, new_cluster_id)

|

Gaussian Mixture Model
~~~~~~~~~

GMM is similar to K-Means but uses Gaussian distribution.

|

Usage Example:

.. code-block:: python

    gmm = pl.GaussianMixture()
    gmm.fit(points)

|
|

Neural Network
--------------

For neural network implementations, you can define a network architecture and train it using:

.. code-block:: python

    network = [
        pl.Dense_layer(input_length, output_length),
        pl.Tanh(),
        pl.Dense_layer(input_length, output_length),
        pl.Tanh()
    ]

    pl.NeuralNetwork.fit(x_train, y_train, network, loss, loss_derivative, epochs, log_error, log_duration)
    predictions = pl.NeuralNetwork.predict(x, network)

|

Usage Example:

.. code-block:: python

    # load data
    features, output = load_data()

    # train Gaussian Naive Bayes classifier
    gnb = pl.GaussianNaiveBayes()
    gnb.fit(features, output)

    # predict
    predictions = gnb.predict(features)

    # evaluate
    accuracy = pl.accuracy(output, predictions)
    precision = pl.precision(output, predictions)
    recall = pl.recall(output, predictions)
    f1 = pl.f1_score(output, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)