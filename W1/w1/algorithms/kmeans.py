import numpy as np
from numpy.random import random
from pandas import DataFrame
from scipy.spatial.distance import euclidean


class Kmeans:
    def __init__(self, k=8, init='k-means++', max_iter=300, n_init=10):
        """
        K-means clustering algorithm.

        :param k: The number of clusters to form.
        :param init: Method for initialization. Avaliable methods are: 'k-means++' and 'random'.
        :param max_iter: Maximum number of iterations of the K-means algorithm for a single run.
        :param n_init: Number of times the K-means algorithm will be run with different centroid seeds.
        """
        self.k = k
        self.init = init
        self.centroids = []
        self.max_iter = max_iter
        self.n_init = n_init
        self.labels = []
        self.square_root_error = None

    def _initialize_centroids(self, data):
        if self.init == 'k-means++':
            # TODO: Implement K-means++ initialization
            raise NotImplementedError("K-means++ initialization not implemented yet")
        else:
            return data[np.random.choice(data.shape[0], self.k, replace=False), :]

    # noinspection SpellCheckingInspection
    def fit(self, data: DataFrame):
        x = data.to_numpy()
        best_sse = np.inf
        best_centroids = []
        best_labels = []
        for _ in range(self.n_init):
            labels, centroids, sse = self.run_kmeans_once(x, self.max_iter)
            if sse <= best_sse:
                best_sse = best_sse
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels = best_labels
        self.square_root_error = best_sse

    def run_kmeans_once(self, x, max_iter):
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
        labels = np.ndarray(x.shape[0])

        # Get the nearest centroid for every element in the data
        for i, elem in enumerate(x):
            min_distance = np.inf
            label = -1
            for idx, centroid in enumerate(centroids):
                distance = euclidean(centroid, elem)
                if distance < min_distance:
                    min_distance = distance
                    label = idx

            labels[i] = label

        new_centroids = self._update_centroids(x, labels)

        return new_centroids, np.array(labels)

    def _update_centroids(self, x, labels):
        aux = np.ndarray((x.shape[0], x.shape[1] + 1))
        aux[:, :-1] = x
        aux[:, -1] = labels
        aux = aux[aux[:, -1].argsort()]
        cluster_data = np.split(aux[:, :-1], np.unique(aux[:, -1], return_index=True)[1][1:])
        # cluster_data = [[x[i] for i in range(x.shape[0]) if labels[i] == j] for j, _ in enumerate(self.centroids)]
        new_centers = []
        for cluster in cluster_data:
            new_centers.append(np.mean(cluster, axis=0, dtype=np.float32))

        return np.array(new_centers)

    def _compute_sse(self, x, labels, centroids):
        square_root_error = 0
        for i, centroid in enumerate(centroids):
            indices = np.where(labels == i)
            square_root_error += np.sum(np.square([euclidean(centroid, point) for point in x[indices]]))

        return square_root_error
