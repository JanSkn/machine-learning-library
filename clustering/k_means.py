import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k = 3):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(x, centroids):
        return np.sqrt(np.sum((x - centroids)**2, axis = 1))      

    def fit(self, X, max_iterations = 500, threshold = 0.001):
        # axis 0: rows, axis 1: columns
        self.centroids = np.random.uniform(np.amin(X, axis = 0), np.amax(X, axis = 0), size = (self.k, X.shape[1]))     
        
        for _ in range(max_iterations):
            data_point_to_cluster = []      # stores cluster number each data point belongs to

            for data_point in X:
                distances = KMeans.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                data_point_to_cluster.append(cluster_num)
            
            data_point_to_cluster = np.array(data_point_to_cluster)
            #print(data_point_to_cluster)
            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(data_point_to_cluster == i))
            #print(cluster_indices)
            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                #print("asdas",indices)
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis = 0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < threshold:
                break
            else:
                self.centroids = np.array(cluster_centers)
                #print(cluster_centers)
                
        return data_point_to_cluster
    

random_points = np.random.randint(0, 100, (100, 2))
kmeans = KMeans(k = 3)
labels = kmeans.fit(random_points)

plt.scatter(random_points[:,0], random_points[:,1],c=labels)
plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:,1],c=range(len(kmeans.centroids)), marker="*",s=200)
plt.show()