import numpy as np
import pandas as pd

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
    # loc vs. iloc: 
    # loc selects rows by index label (can also be numeric, but numeric index can differ from real index), 
    # iloc selects row by actual index
    def fit(self, X: np.ndarray | pd.DataFrame, Y: np.ndarray | pd.DataFrame) -> None:
        """
        Trains the algorithm. Input can be a numpy or pandas object.

        Parameters:
            :X (numpy.ndarray | pandas.DataFrame): Training input
            :Y (numpy.ndarray | pandas.DataFrame): Training output

        Returns:
            None
        """
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)

        self.classes = sorted(list(Y.iloc[:, 0].unique()))
        num_of_samples, num_of_features = X.shape 
        self.mean = pd.DataFrame(np.zeros((len(self.classes), num_of_features)))
        self.variance = pd.DataFrame(np.zeros((len(self.classes), num_of_features)))
        self.prior = pd.DataFrame(np.zeros(len(self.classes)))

        for index, c in enumerate(self.classes):
            class_df = X[Y.iloc[:, 0] == c]
            self.mean.iloc[index] = class_df.mean().values        # operator returns column names and values --> .values
            self.variance.iloc[index] = class_df.var().values     # NaN for one value due to 0 divison
            self.prior.loc[index] = float(len(class_df)) / float(num_of_samples)

    def predict(self, X: np.ndarray | pd.DataFrame) -> pd.DataFrame:
        """
        Computes the output of a given X.

        Parameters:
            :X (np.ndarray | pd.DataFrame): Testing input

        Returns:
            Predicted classes as pandas dataframe
        """
        X = pd.DataFrame(X)
        y_pred = [self._predict(x[1:len(x)]) for x in X.itertuples()]           # x is itertuple object --> x[1:len(x)] removes index
        return pd.DataFrame(y_pred)

    def _predict(self, x: tuple) -> int | str:
        """
        Helper function for predict.
        """
        posteriors = []
        # P(y) * ∏ P(x_i|y) --> ln P(y) + ∑ ln P(x_i|y) to prevent underflow
        for index, c in enumerate(self.classes):
            prior = np.log(self.prior.iloc[index])      # get prior at index
            posterior = np.sum(np.log(self.gaussian_distribution(index, x)))
            posterior += prior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def gaussian_distribution(self, index: int, x: tuple) -> pd.Series:
        """
        The gaussian or normal distribution of a feature x_i.

        Parameters:
            :index (int): Index of the current class
            :x (tuple): Current row

        Returns:
            Gaussian distribution of each feature as pandas series
        """
        mean, variance = self.mean.iloc[index], self.variance.iloc[index]       # mean, variance for each class (row)
        # P(x_i|y) = N(μ, σ^2) with x = x_i, μ = μ_y, σ = σ_y
        return np.exp(-((x - mean) ** 2)/(2 * variance)) / np.sqrt(2 * np.pi * variance)      

