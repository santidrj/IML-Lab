import numpy as np
from numpy.random import random
from pandas import DataFrame
from scipy.spatial.distance import euclidean


class Kmeans:
    def __init__(self, k=5, max_iter=300, n_init=100):
        self.k = k
        self.centroids = []
        self.max_iter = max_iter
        self.n_init = n_init
        self.labels = []

    def _initialize_centroids(self, data):
        return data[np.random.choice(data.shape[0], self.k, replace=False), :]

    # noinspection SpellCheckingInspection
    def kmeans(self, data: DataFrame, k=None):
        if k is not None:
            self.k = k

        x = data.to_numpy()
        for _ in range(self.n_init):
            self.run_kmeans(x, self.max_iter)

    def run_kmeans(self, x, max_iter):
        # Initialize the centroids using K random samples of the data
        self.centroids = self._initialize_centroids(x)

        centroids = []
        labels = []
        for _ in range(max_iter):
            # Assign each sample to the nearest cluster
            centroids, labels = self._fit(x)

        self.labels = labels
        self.centroids = centroids

    def _fit(self, x):
        labels = np.ndarray(x.shape[0])

        # Get the nearest centroid for every element in the data
        for i, elem in enumerate(x):
            min_distance = np.inf
            label = -1
            for idx, centroid in enumerate(self.centroids):
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
