Usage
=====

PyLearn provides different features from supervised and unsupervised learning. 

To use them, just import pylearn:

.. code-block:: python

   import pylearn as pl

|

Each model can be saved to storage to prevent training your model again. Just import it:

.. code-block:: python

   import pl.save, pl.load

|

If you want to normalize your input data, simply import:

.. code-block:: python

   import pl.min_max_normalization, pl.z_normalization

|

You can evaluate every model with accuracy, precision, recall and F1 score:

.. code-block:: python

   import pl.evaluate

|

Change numbers into a one hot representation:

.. code-block:: python

   import pl.to_one_hot

|

The major features are:

|
Classification
~~~~~~~~~~~~~~

You can use Gaussian Naive Bayes.

Gaussian works for continuous data.
Multinomial Naive Bayes works perfect for text classification and will come in version 1.1.0.

The usage is quite simple:

.. code-block:: python

   gnb = pl.GaussianNaiveBayes()

|
Now, train the model by using the fit function:

.. code-block:: python

   gnb.fit(features, output)

|
Let the model predict your input:

.. code-block:: python

   gnb.predict(features)

|
|
Clustering
~~~~~~~~~~

You can choose between K-Means and K-Medoids as clustering models.

The usage of both is quite similar:

.. code-block:: python

   kmeans = pl.KMeans()
   kmedoids = pl.KMedoids()

|
Now, train the model by using the fit function, we will use kmeans to continue:

.. code-block:: python

   kmeans.fit(points)

|
This returns a list of the to the data points assigned clusters.
You could visualize the result with matplotlib.

|
If you want to customize the result, the following functions may help you:

.. code-block:: python

   kmeans.assigned_clusters(any_cluster)
   kmeans.rename(old, new)

|
|
Neural Network
~~~~~~~~~~~~~~

The neural network comes with different activation functions and loss functions.

First, you need to create a network, for example:

.. code-block:: python

   network = [
        pl.Dense_layer(input_length, output_length),
        pl.Tanh(),
        plpDense_layer(input_length, output_length),
        pl.Tanh()
    ]

|
Now, train the model:

.. code-block:: python

    pl.NeuralNetwork.fit(x_train, y_train, network, loss, loss_derivative, epochs, log_error, log_duration)

|

Let the model predict your input:

.. code-block:: python

   pl.NeuralNetwork.predict(x, network)