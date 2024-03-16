import numpy as np
from pylearn import GaussianMixture

def test_fit():
    gm = GaussianMixture()
    X = np.random.rand(100, 2)  # Sample data
    data_points_to_cluster, posteriors = gm.fit(X)
    
    assert len(data_points_to_cluster) == 100
    assert posteriors.shape == (100, 3)

def test_posterior_probabilities():
    gm = GaussianMixture()
    np.random.seed(0)   # set seed to reproduce values
    X = np.random.rand(100, 2)  
    gm.centroids = np.array([[0, 0], [1, 1], [2, 2]])
    cov = np.tile(np.eye(2)[:, :, np.newaxis], (1, 1, 3))  
    mix = np.ones(3) / 3
    posteriors = gm._posterior_prob(X, cov, mix)
    assert posteriors.shape == (100, 3)
    np.testing.assert_allclose(np.sum(posteriors, axis=1), np.ones(100))

def test_convergence():
    np.random.seed(0)   # set seed to reproduce values
    gm = GaussianMixture()
    X = np.random.rand(100, 2)  
    data_points_to_cluster, posteriors = gm.fit(X, max_iterations=1)
    
    assert len(data_points_to_cluster) == 100
    assert posteriors.shape == (100, 3)

def test_parameter_update():
    gm = GaussianMixture()
    np.random.seed(0)   # set seed to reproduce values
    X = np.random.rand(100, 2)  
    posteriors = np.random.rand(100, 3)
    mix = np.random.rand(3)
    cov = np.random.rand(2, 2, 3)
    gm.centroids = np.random.rand(3, 2) 
    gm.data_points_to_cluster = np.random.randint(0, 3, size=100)
    gm.posteriors = posteriors  
    new_mix, new_cov, new_centroids = gm._update_params(X, posteriors, mix, cov)
    
    assert new_mix.shape == (3,)
    assert new_cov.shape == (2, 2, 3)
    assert new_centroids.shape == (3, 2)


