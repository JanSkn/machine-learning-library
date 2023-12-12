import numpy as np
import pandas as pd
from pylearn import KMeans, accuracy, precision, recall, f1_score

def test_fit():
    kmeans = KMeans(3)
    np.random.seed(0)                   # set seed to reproduce values
    X = np.random.rand(100, 2)          # 100 samples with 2 features each
    clusters = kmeans.fit(X)
    assert len(clusters) == 100
    assert len(np.unique(clusters)) == 3
    assert kmeans.centroids.shape == (3, 2)