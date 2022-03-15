# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_neighbors=5, weights='uniform', metric='l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        # Fit function just returns the X and y dataset
        self._X = X
        self._y = y

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        result = []
        for i in X:

            dist = []
            # We calculate the distance based on the parameter selected

            if self._distance == euclidean_distance:
                for j, k in enumerate(self._X):
                    dist.append((euclidean_distance(i, k), j))

            else:
                for j, k in enumerate(self._X):
                    dist.append((manhattan_distance(i, k), j))

            sorted_dist = sorted(dist, key=lambda x: int(x[0]))

            # We select the lowest n distances from sorted list, n being the number of neighbours specified.
            ns = sorted_dist[0:self.n_neighbors]

            temp = {}
            # We calculate the vote based on type of weight specified.

            if self.weights == "uniform":
                for i in ns:
                    if self._y[i[1]] in temp.keys():
                        temp[self._y[i[1]]] = temp[self._y[i[1]]] + 1
                    else:
                        temp[self._y[i[1]]] = 1

            elif self.weights == "distance":
                for i in ns:
                    if self._y[i[1]] in temp.keys():
                        if i[0] != 0:
                            temp[self._y[i[1]]] = temp[self._y[i[1]]] + 1 / i[0]
                        else:
                            temp[self._y[i[1]]] = temp[self._y[i[1]]]
                    else:
                        if i[0] != 0:
                            temp[self._y[i[1]]] = 1 / i[0]
                        else:
                            temp[self._y[i[1]]] = 0

            # Start of code from stack overflow https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
            result.append(max(temp, key=temp.get))
            # End of code from stack overflow

        return result
