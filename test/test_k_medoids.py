import numpy as np
from pylearn import KMedoids

def test_fit():
    kmedoids = KMedoids(3)
    np.random.seed(0)                   # set seed to reproduce values
    X = np.random.rand(100, 2)          # 100 samples with 2 features each
    clusters = kmedoids.fit(X)
    assert len(clusters) == 100
    assert len(np.unique(clusters)) == 3
    assert kmedoids.centroids.shape == (3, 2)