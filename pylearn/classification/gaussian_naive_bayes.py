import numpy as np
import pandas as pd
import time

class GaussianNaiveBayes:
    """
    Computes continuous classification problems by applying the Bayes theorem
    with a gaussian distribution.

    Attributes:
        :classes (list): A list of all classes
        :mean (numpy.ndarray | pandas.DataFrame): Mean of all features
        :variance (numpy.ndarray | pandas.DataFrame): Variance of all features
        :prior (numpy.ndarray | pandas.DataFrame): Prior of Bayes theorem
    """
    # loc selects rows by index label (can also be numeric, but numeric index can differ from real index), 
    # iloc selects row by actual index
    def fit(self, X: np.ndarray | pd.DataFrame | pd.Series, Y: np.ndarray | pd.DataFrame | pd.Series, log_duration=True) -> None:
        """
        Trains the algorithm. Input can be a numpy or pandas object.

        Parameters:
            :X (numpy.ndarray | pandas.DataFrame | pd.Series): Training input
            :Y (numpy.ndarray | pandas.DataFrame | pd.Series): Training output
            :log_duration (bool, optional): Logs the duration of the training, default: True

        Returns:
            None
        """
        start = time.time()

        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)
        self.classes = sorted(list(Y.iloc[:, 0].unique()))
        num_of_samples, num_of_features = X.shape 
        self.mean = pd.DataFrame(data=0, index=range(len(self.classes)), columns=range(num_of_features)).astype(float)
        self.variance = pd.DataFrame(data=0, index=range(len(self.classes)), columns=range(num_of_features)).astype(float)
        self.prior = pd.DataFrame(data=0, index=range(len(self.classes)), columns=range(1)).astype(float)

        for index, c in enumerate(self.classes):
            class_df = X[Y.iloc[:, 0] == c]
            self.mean.iloc[index] = class_df.mean().values        # operator returns column names and values --> .values
            self.variance.iloc[index] = class_df.var().values     # NaN for one value due to 0 divison
            self.prior.iloc[index] = float(len(class_df)) / float(num_of_samples)

        end = time.time()

        if log_duration:
            print(f"Duration of training: {end - start}\n")

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Computes the output of a given X.

        Parameters:
            :X (numpy.ndarray | pandas.DataFrame): Testing input

        Returns:
            Predicted classes as array
        """
        X = pd.DataFrame(X)
        y_pred = [self._predict(x[1:len(x)]) for x in X.itertuples()]           # x is itertuple object --> x[1:len(x)] removes index
        return np.array(y_pred)

    def _predict(self, x: tuple) -> int | str:
        """
        Helper function for predict.
        """
        posteriors = []
        # P(y) * ∏ P(x_i|y) --> ln P(y) + ∑ ln P(x_i|y) to prevent underflow
        for index, c in enumerate(self.classes):
            prior = np.log(self.prior.iloc[index])      # get prior at index
            posterior = np.sum(np.log(self._gaussian_distribution(index, x)))
            posterior += prior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _gaussian_distribution(self, index: int, x: tuple) -> pd.Series:
        """
        The gaussian or normal distribution of a feature x_i (Probability Density Function).

        Parameters:
            :index (int): Index of the current class
            :x (tuple): Current row

        Returns:
            Gaussian distribution of each feature as pandas series
        """
        mean, variance = self.mean.iloc[index], self.variance.iloc[index]       # mean, variance for each class (row)
        # P(x_i|y) = N(μ, σ) with x = x_i, μ = μ_y, σ = σ_y
        return np.exp(-((x - mean) ** 2)/(2 * variance)) / np.sqrt(2 * np.pi * variance)      