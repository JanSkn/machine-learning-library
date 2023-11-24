# Example for clustering

# allows import from different folder
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import matplotlib.pyplot as plt
from pylearn import KMeans, KMedoids

# Due to the short training duration, this example doesn't store and load the trained model and retraines every execution

random_points = np.random.randint(0, 100, (100, 2))
max_number_of_points = 30
print(f"Data Points ({max_number_of_points} of {len(random_points)}):")
print(random_points[:max_number_of_points])
print("...")
print()

kmeans = KMeans(5)
labels = kmeans.fit(random_points, max_iterations = 100)
print("Assigned Clusters:")
print(labels)
print()
plt.scatter(random_points[:, 0], random_points[:, 1], c = labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c = range(len(kmeans.centroids)), marker = "x", s = 200)
plt.title("K Means")

kmeans.rename([0, 1, 2, 3, 4], ["a", "b", "c", "d", "e"])
print("To 'a' assigned data points:")
print(kmeans.assigned_clusters("a"))
plt.show()

print()

kmedoids = KMedoids(5)
labels = kmedoids.fit(random_points, max_iterations = 100)
print("Assigned Clusters:")
print(labels)
print()
plt.scatter(random_points[:, 0], random_points[:, 1], c = labels)
plt.scatter(kmedoids.centroids[:, 0], kmedoids.centroids[:, 1], c = range(len(kmedoids.centroids)), marker = "x", s = 200)
plt.title("K Medoids")

kmedoids.rename([0, 1, 2, 3, 4], ["f", "g", "h", "i", "j"])
print("To 'f' and 'g' assigned data points:")
print(kmedoids.assigned_clusters(["f", "g"]))
plt.show()




