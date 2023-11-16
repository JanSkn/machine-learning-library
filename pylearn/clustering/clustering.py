import numpy as np

class Clustering:
    """
    An abstract base class for Clustering Algorithms.
    Defines the basic properties and methods that can be used.

    Attributes:
        :k (int): Number of clusters
        :centroids (numpy.ndarray): Matrix of centroids of all k clusters
        :data_points (numpy.ndarray): Matrix of all data points
        :data_points_to_cluster (list): List of each data point's assigned cluster
        :clusters (list): List of all k clusters
    """
    def __init__(self, k=3) -> None:
        self.k = k
        self.centroids = None
        self.data_points = None
        self.data_points_to_cluster = None
        self.clusters = None

    @staticmethod
    def euclidean_distance(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Calculates distance of a data point x 
        to all k centroids.

        Parameters:
            :x (numpy.ndarray): Data point (vector)
            :centroids (numpy.ndarray): Centroids in a matrix (each row is one centroid) 

        Returns:
            Array of the distances
        """
        # change axis to 1 because every centroid is stored in one row
        return np.sqrt(np.sum((x - centroids)**2, axis = 1))
    
    @staticmethod
    def median(x: np.ndarray) -> np.ndarray:
        """
        Determines the point with the median smallest distance to
        all other data points in the cluster.
        The median point must be one of the data points.

        Parameters:
            x (numpy.ndarray): Matrix of all data points in one cluster

        Returns:
            Data point as one-element array
        """  
        min_distance = float('inf') 
    
        for point in x:
            distance = 0
        
            for other_point in x:
                distance += Clustering.euclidean_distance(point, other_point)  
            
            if distance < min_distance:
                min_distance = distance
                centroid = point

        # returned array only has one element, but is an array of array, therefore [0]
        return centroid[0]

    def assigned_clusters(self, clusters: list | str | int) -> list[tuple]:
        """
        All to the clusters assigned data points.

        Parameters:
            :clusters (list | str | int): Cluster name(s) 

        Returns:
            List of the data points
        """
        mapping = zip(self.data_points, self.data_points_to_cluster)
        result = []
        for data_point, cluster in mapping:
            if (isinstance(clusters, list) and cluster in clusters) or (not isinstance(clusters, list) and cluster == clusters):    # allow clusters as list, str or int
                result.append((list(data_point), cluster))
        return result

    def rename(self, old_clusters: list, new_clusters: list) -> list: 
        """
        Renames the clusters.  

        Parameters:
            :old_clusters (list): List of all old clusters to get renamed
            :new_clusters (list): List of the renamed clusters

        Returns:
            A list of the data points
        """
        # TODO add parameter limit (int) to limit the output of each cluster data points 
        mapping = zip(old_clusters, new_clusters)
        mapping = {old:new for old, new in mapping}
        renamed_clusters = list(self.data_points_to_cluster)
        
        for i, data_point in enumerate(self.data_points_to_cluster):
            renamed_clusters[i] = mapping[data_point]
            self.data_points_to_cluster[i] = mapping[data_point]

        self.clusters = list(set(renamed_clusters))
        return renamed_clusters