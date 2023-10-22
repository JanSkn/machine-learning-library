import numpy as np

class Scaling:
    @staticmethod
    def min_max_normalization(data: np.ndarray) -> np.ndarray:
        """
        Normalizes Data in a range from 0 to 1 using minx-max Value of every Feature. 
        Real-world datasets often contain features that are 
        varying in degrees of magnitude, range, and units. 
        Therefore, perform feature scaling, otherwise some features dominate the calculations.
        """
        #each row is a data point, each column symbolizes a feature
        #we need to find min and max for every feature
        min = np.amin(data, axis=0)
        max = np.amax(data, axis=0)
        normalized_data = np.ndarray(shape=data.shape)

        for (data_point, feature), value in np.ndenumerate(data):
            normalized_data[data_point, feature] = (value-min[feature])/(max[feature]-min[feature])

        return normalized_data
    
    @staticmethod
    def z_normalization(data: np.ndarray) -> np.ndarray:
        """
        Normalizes Data in a range from 0 to 1 using Z-Scores.
        Real-world datasets often contain features that are 
        varying in degrees of magnitude, range, and units. 
        Therefore, perform feature scaling, otherwise some features dominate the calculations.
        """
        #each row is a data point, each column symbolizes a feature
        #we need to find mean and standart derivation for every feature
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = np.ndarray(shape=data.shape)

        for (data_point, feature), value in np.ndenumerate(data):
            normalized_data[data_point, feature] = (value-mean[feature])/(std[feature])

        return normalized_data

