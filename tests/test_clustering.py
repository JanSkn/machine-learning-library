import numpy as np
from pylearn import Clustering, KMeans

def test_euclidean_distance():
    x = np.array([1, 2])
    centroids = np.array([[1, 2], [4, 5]])
    distances = Clustering.euclidean_distance(x, centroids)
    assert distances.shape == (2,)
    assert distances[0] == 0
    assert distances[1] > 0

def test_median():
    x = np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
    median = Clustering.median(x)
    assert len(median) == 2

def test_assigned_clusters():
    kmeans = KMeans(2)
    kmeans.data_points = np.array([[1, 2], [3, 4]])
    kmeans.data_points_to_cluster = [0, 1]
    assigned = kmeans.assigned_clusters(0)
    assert len(assigned) == 1

def test_rename():
    kmeans = KMeans(2)
    kmeans.data_points_to_cluster = [0, 1]
    kmeans.rename([0, 1], ['A', 'B'])
    assert kmeans.data_points_to_cluster == ['A', 'B']