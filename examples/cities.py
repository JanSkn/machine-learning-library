# Example for clustering

# allows import from different folder
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylearn import KMeans, KMedoids

# Due to the short training duration, this example doesn't store and load the trained model and does training again every execution

cities = pd.read_csv("examples/data/city_coordinates.csv")
city_coordinates = cities[["longitude", "latitude"]].values

plt.title("Cities before training")
plt.scatter(city_coordinates[:, 0], city_coordinates[:, 1])
for i, city in enumerate(cities["city"]):
    plt.text(city_coordinates[i, 0], city_coordinates[i, 1], city, fontsize=8)
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()

kmeans = KMeans()
labels = kmeans.fit(city_coordinates, max_iterations=100)

plt.scatter(city_coordinates[:, 0], city_coordinates[:, 1], c = labels)
for i, city in enumerate(cities["city"]):
    plt.text(city_coordinates[i, 0], city_coordinates[i, 1], city, fontsize=8)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c = range(len(kmeans.centroids)), marker = "x", s=200)
plt.title("Cities assigned to continents after K Means")
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()

print("Coordinates of continents:")
print(kmeans.centroids)
print()
# you can now rename the centroids to North America, Europe, Asia with kmeans.rename

kmedoids = KMedoids()
labels = kmedoids.fit(city_coordinates, max_iterations = 100)

plt.scatter(city_coordinates[:, 0], city_coordinates[:, 1], c=labels)
for i, city in enumerate(cities["city"]):
    plt.text(city_coordinates[i, 0], city_coordinates[i, 1], city, fontsize=8)
plt.scatter(kmedoids.centroids[:, 0], kmedoids.centroids[:, 1], c = range(len(kmedoids.centroids)), marker = "x", s=200)
plt.title("Cities assigned to continents after K Medoids")
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
plt.show()

print("Coordinates of continents:")
print(kmedoids.centroids)
# you can now rename the centroids to North America, Europe, Asia with kmedoids.rename