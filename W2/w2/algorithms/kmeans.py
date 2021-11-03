import numpy as np
from numpy import arange
from numpy.random import random
from pandas import DataFrame
from scipy.spatial.distance import euclidean, minkowski


class Kmeans:
    def __init__(self, k=8, init='k-means++', metric='euclidean', max_iter=300, n_init=10):
        """
        K-means clustering algorithm.

        :param metric:
        :param k: The number of clusters to form.
        :param init: Method for initialization. Available methods are: 'k-means++' and 'random'.
        :param metric: Metric to compute the distances. Available metrics are: 'euclidean', and 'l1'.
        :param max_iter: Maximum number of iterations of the K-means algorithm for a single run.
        :param n_init: Number of times the K-means algorithm will be run with different centroid seeds.
        """
        self.k = k
        self.init = init
        self.centroids = []
        self.metric = metric
        self.max_iter = max_iter
        self.n_init = n_init
        self.labels = []
        self.square_error = None

    def _initialize_centroids(self, data):
        """
        Initializes the centroids using either K-means++ or random initialization.
        :param data: Data from wich to select the random centroids.
        :return: The array of centroids.
        """
        if self.init == 'k-means++':
            return self._kmeans_plusplus(data, self.k)
        else:
            return data[np.random.choice(data.shape[0], self.k, replace=False), :]

    def _compute_distance(self, a, b):
        """
        Compute the distance between two n-dimensional arrays.
        :param a: n-dimensional array.
        :param b: n-dimensional array.
        :return: Distance between a and b
        """
        if self.metric == 'euclidean':
            return euclidean(a, b) ** 2
        elif self.metric == 'l1':
            return minkowski(a, b, p=1)

    def fit(self, data: DataFrame):
        """
        Perform K-means clustering over the data.
        :param data: Data to use for the K-means.
        """
        x = data.to_numpy(dtype='float64')
        best_sse = np.inf
        best_centroids = []
        best_labels = []
        for _ in range(self.n_init):
            labels, centroids, sse = self.run_kmeans_once(x, self.max_iter)
            if sse <= best_sse:
                best_sse = sse
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels = best_labels
        self.square_error = best_sse

    def run_kmeans_once(self, x, max_iter):
        """
        Run the K-means algorithm once.
        :param x: Data to cluster using K-means.
        :param max_iter: Maximum number of iterations for updating the centers.
        :return: Tuple containing the labels of the data, the clusters centroids and the SSE.
        """

        # Initialize the centroids using K random samples of the data
        self.centroids = self._initialize_centroids(x)

        old_centroids = self.centroids.copy()
        new_centroids = []
        labels = []

        for i in range(max_iter):
            # Assign each sample to the nearest cluster and update the centroids
            new_centroids, labels = self._fit(x, old_centroids)
            if (old_centroids == new_centroids).all():
                break
            else:
                old_centroids = new_centroids.copy()

        centroids = np.array(new_centroids)
        sse = self._compute_sse(x, labels, centroids)

        return labels, centroids, sse

    def _fit(self, x, centroids):
        """
        Assing each sample to the nearest cluster and update the centroids.
        :param x: Array of samples.
        :param centroids: List of clusters centroids.
        :return: The new centroids and the labels for each sample.
        """

        labels = np.ndarray(x.shape[0])

        # Get the nearest centroid for every element in the data
        for i, elem in enumerate(x):
            min_distance = np.inf
            label = -1
            for idx, centroid in enumerate(centroids):
                distance = self._compute_distance(centroid, elem)
                if distance < min_distance:
                    min_distance = distance
                    label = idx

            labels[i] = label

        new_centroids = self._update_centroids(x, centroids, labels)

        return new_centroids, np.array(labels)

    def _update_centroids(self, x, old_centroids, labels):
        """
        Updates the centroids of each cluster.
        :param x: N-dimensional array of samples
        :param old_centroids: Current list of centroids.
        :param labels: List of labels for each sample.
        :return: The list of new centroids.
        """
        aux = np.ndarray((x.shape[0], x.shape[1] + 1))
        aux[:, :-1] = x
        aux[:, -1] = labels
        new_centers = []
        for i, c in enumerate(old_centroids):
            cluster = x[labels == i]
            if cluster.size == 0:
                new_centers.append(c)
            else:
                new_centers.append(np.mean(cluster, axis=0, dtype=x.dtype))

        return np.array(new_centers)

    def _compute_sse(self, x, labels, centroids):
        """
        Computes the Sum of Squared Error (SSE) of the clustering.
        :param x: Data samples.
        :param labels: Labels for each data sample.
        :param centroids: Centroids of each cluster.
        :return: The total SSE of the clustering.
        """
        sum_squared_error = 0
        for i, centroid in enumerate(centroids):
            indices = np.where(labels == i)
            sum_squared_error += np.sum([self._compute_distance(centroid, point) for point in x[indices]])

        return sum_squared_error

    def _kmeans_plusplus(self, x, k):
        """
        Initialize the clusters centroids by applying the K-means++ algorithm.
        :param x: Data samples.
        :param k: Desired number of clusters.
        :return: The list of initial centroids.
        """
        # Number of local seeding trials based on what
        # Arthur, David & Vassilvitskii, Sergei said in
        # K-Means++: The Advantages of Careful Seeding
        # 10.1145/1283383.1283494
        n_local_trials = 2 + int(np.log(k))

        n_samples, n_features = x.shape
        centroids = np.empty((k, n_features), dtype=x.dtype)

        centroid_idx = np.random.randint(n_samples)
        centroids[0] = x[centroid_idx]

        closest_distances = self._closest_distances(centroids[0], x)
        current_pot = closest_distances.sum()

        for centroid in range(1, k):
            weights = closest_distances / current_pot
            candidates_idx = np.random.choice(arange(closest_distances.size), n_local_trials, replace=False, p=weights)

            dist_to_candidates = self._closest_distances(x[candidates_idx], x)

            np.minimum(closest_distances, dist_to_candidates, out=dist_to_candidates)
            candidates_pot = dist_to_candidates.sum(axis=1)

            best_candidate = np.argmin(candidates_pot)
            closest_distances = dist_to_candidates[best_candidate]
            current_pot = candidates_pot[best_candidate]

            centroids[centroid] = x[best_candidate]

        return centroids

    def _closest_distances(self, x, y):
        """
        Compute the distances from y to each point in x.
        :param x: Array of points.
        :param y: Array of points.
        :return: A matrix of distances.
        """
        if x.ndim == 1:
            distances = np.array([self._compute_distance(x, point) for point in y])
        else:
            distances = np.empty((x.shape[0], y.shape[0]), dtype=x.dtype)
            for i, elem in enumerate(x):
                distances[i] = [self._compute_distance(elem, point) for point in y]
        return distances
