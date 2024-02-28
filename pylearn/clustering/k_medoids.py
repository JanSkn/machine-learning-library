import numpy as np
from pylearn.clustering.clustering import Clustering

class KMedoids(Clustering):
    """
    K Medoids algorithm computes clusters by calculating the median of the cluter points.
    Centroid must be a data point itself.

    Attributes:
        :k (int): Number of clusters
        :centroids (numpy.ndarray): Matrix of centroids of all k clusters
        :data_points (numpy.ndarray): Matrix of all data points
        :data_points_to_cluster (list): List of each data point's assigned cluster
        :clusters (list): List of all k clusters
    """
    

    def fit(self, X: np.ndarray, max_iterations=500, threshold=0.001) -> np.ndarray:
        """
        Parameters:
            :X (numpy.ndarray): Matrix of data points (each row is one data point) 
            :max_iterations (int, optional): Number of iterations to update the centroids, default: 500
            :threshold (float, optional): Stopping criterion to interrupt the update iterations, default: 0.001

        Returns:
            An array of the to data points assigned clusters
        """
        # axis 0: rows, axis 1: columns
        # Centroids as k x len(X) matrix with one centroid each row
        indices = np.random.choice(X.shape[0], self.k)      # initialise centroids randomly from the existing data points
        self.centroids = X[indices]
        self.data_points = X
        for _ in range(max_iterations):
            data_points_to_cluster = []                      # stores cluster number each data point belongs to

            for data_point in X:
                distances = KMedoids.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)          
                data_points_to_cluster.append(cluster_num)
            
            data_points_to_cluster = np.array(data_points_to_cluster)

            cluster_indices = []            # array of arrays 

            for i in range(self.k):
                # argwhere returns array of indices where condition true
                # each cluster has an array of indices of its associated data points
                cluster_indices.append(np.argwhere(data_points_to_cluster == i))     

            cluster_centers = []            # to recalculate the new cluster centers

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:   # if a centroid has no data point
                    cluster_centers.append(self.centroids[i])
                else:
                    # each row of the X matrix is a data point
                    # calculate mean for each coordinate (axis 0) for each selected data point (indices)
                    # returned array only has one element, but is an array of array, therefore [0]
                    cluster_centers.append(KMedoids.median(X[indices]))

            if np.max(self.centroids - np.array(cluster_centers)) < threshold:      # stopping criterion to prevent stagnation
                break
            else:
                self.centroids = np.array(cluster_centers)                          # update centroids
        
        self.clusters = list(set(data_points_to_cluster))
        self.data_points_to_cluster = list(data_points_to_cluster)
        return np.array(data_points_to_cluster)
