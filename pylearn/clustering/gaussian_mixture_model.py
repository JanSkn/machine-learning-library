import numpy as np

class GaussianMixture:
    """      
    Gaussian Mixture algorithm computes clusters by using the Gaussian distribution.
    
    Attributes:
        :g (int, optional): Number of clusters
        :centroids (numpy.ndarray): Matrix of centroids of all g clusters
        :data_points_to_cluster (list): List of each data point's assigned cluster
        :posteriors (numpy.ndarray): Matrix of the posterior probabilities
        :clusters (numpy.ndarray): List of all k clusters
    """
    def __init__(self, g=3) -> None:
        self.g = g
        self.centroids = None
        self.data_points_to_cluster = None
        self.posteriors = None
        self.clusters = None

    def fit(self, X: np.ndarray, max_iterations=100, threshold=0.001) -> tuple[np.ndarray, np.ndarray]:
        """
        Assigns each data point the best cluster by calculating the distances.

        Parameters:
            :X (numpy.ndarray): Matrix of data points (each row is one data point) 
            :max_iterations (int, optional): Number of iterations to update the centroids, default: 100
            :threshold (float, optional): Stopping criterion to interrupt the update iterations, default: 0.001

        Returns:
            The to data points assigned clusters and the posterior probabilities
        """
        num_of_samples = X.shape[0]
        # TODO initial centroids from K Means
        self.centroids = X[np.random.choice(num_of_samples, self.g, replace=False), :]
        cov = np.repeat(np.cov(X, rowvar=False)[:, :, np.newaxis], self.g, axis=-1)
        mix = np.ones(self.g) / self.g

        for _ in range(max_iterations):
            prev_centroids = self.centroids
            
            # E-Step
            posteriors = self._posterior_prob(X, cov, mix)
            data_points_to_cluster = np.argmax(posteriors, axis=1)

            # M-Step
            mix, cov, self.centroids = self._update_params(
                X, posteriors, mix, cov)
                    
            if np.sum(np.abs(self.centroids - prev_centroids)) < threshold:
                break

        data_points_to_cluster = np.argmax(self._posterior_prob(X, cov, mix), axis=1)
        self.clusters = list(set(data_points_to_cluster))  
        self.data_points_to_cluster = data_points_to_cluster
        self.posteriors = posteriors

        return data_points_to_cluster, posteriors

    # E-Step
    @staticmethod
    def _multivariate_gaussian_density(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
        """
        Helper function for fit.
        Applies the E-Step of the GMM algorithm. Calculating the multivariate Gaussian density of the data.
        
        Parameters:
            :X (numpy.ndarray): Matrix of data points (each row is one data point) 
            :mu (numpy.ndarray): Mean location of the Gaussian component 
            :cov (numpy.ndarray): Covariance matrix of the Gaussian component 

        Returns:
            Gaussian density
        """
        num_of_samples = X.shape[0]
        
        determinant = np.linalg.det(cov)
        normalisation_factor = 1.0 / ((2 * np.pi) * (num_of_samples / 2) * determinant ** (1.0 / 2))
        mean_centred_data = X - mu
        inverse_covariance = np.linalg.inv(cov)
        result = np.e**(-(1.0 / 2) * (mean_centred_data @ inverse_covariance @ mean_centred_data.T))
        result = normalisation_factor * result

        return result

    def _posterior_prob(self, X: np.ndarray, cov: np.ndarray, mix: np.ndarray) -> np.ndarray:
        """
        Helper function for fit.
        Calculates posterior probability for each Guassian component.
        
        Parameters:
            :X (numpy.ndarray): Matrix of data points (each row is one data point) 
            :mu (numpy.ndarray): Mean location of the Gaussian component 
            :cov (numpy.ndarray): Covariance matrix of the Gaussian component 

        Returns:
            A matrix with posterior probability that each sample belongs to a Gaussian component
        """
        
        num_of_samples = X.shape[0]
        num_of_components = self.centroids.shape[0]
        p = np.zeros([num_of_samples, num_of_components])
        p_total = np.zeros(num_of_samples)
        posteriors = np.zeros([num_of_samples, num_of_components])

        for i_sample in range(num_of_samples):
            for i_component in range(num_of_components):
                p[i_sample, i_component] = self._multivariate_gaussian_density(
                    X[i_sample, :].T, self.centroids[i_component, :], cov[:, :, i_component])
                p_total[i_sample] = p_total[i_sample] + p[i_sample, i_component]

            for i_component in range(num_of_components):
                posteriors[i_sample, i_component] = (
                    p[i_sample, i_component] * mix[i_component]) / (p_total[i_sample] * mix[i_component])

        return posteriors


    # M-Step
    def _update_params(self, X: np.ndarray, posteriors: np.ndarray, mix: np.ndarray, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply the M-Step of GMM algorithm. Updating the component parameters and assignments.
        
        Parameters:
            :X (numpy.ndarray): Matrix of data points (each row is one data point) 
            :posteriors (numpy.ndarray): Matrix of posterior probabilities
            :mu (numpy.ndarray): Mean location of the Gaussian component 
            :cov (numpy.ndarray): Covariance matrix of the Gaussian component 
        
        Returns:
            Updated mixing coefficients, covariance matrix, centroids
        """
        
        num_of_samples = X.shape[0]
        num_of_components = self.centroids.shape[0]
        
        # transpose due to numpy's handling of dimensions
        X = X.T
        
        cluster_weight = np.sum(posteriors, axis=0)
        new_mix = (1 / X.shape[0]) * cluster_weight

        new_centroids = np.zeros(self.centroids.shape)
        for i_component in range(num_of_components):
            new_centroids[i_component, :] = np.sum(posteriors[:, i_component] * X, axis=1) / cluster_weight[i_component]

        new_cov = np.zeros(cov.shape)
        for i_component in range(num_of_components):
            mu_centred_data = X - np.expand_dims(new_centroids[i_component, :], axis=-1)
            for i_sample in range(num_of_samples):
                cov = mu_centred_data[:, i_sample:i_sample+1] @ mu_centred_data[:, i_sample:i_sample+1].T
                scaled_cov = posteriors[i_sample, i_component] * cov
                new_cov[:, :, i_component] += scaled_cov
                
            new_cov[:, :, i_component] /= cluster_weight[i_component]

        return new_mix, new_cov, new_centroids