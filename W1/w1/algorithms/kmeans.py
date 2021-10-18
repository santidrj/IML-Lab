import numpy as np
from numpy.random import random
from pandas import DataFrame
from scipy.spatial.distance import euclidean


class Kmeans:
    def __init__(self, k=5, max_iter=300):
        self.k = k
        self.centroids = []
        self.max_iter = max_iter
        self.labels = []

    def initialize_centroids(self, data):
        for _ in range(self.k):
            self.centroids.append(random(data.shape[1]))

    # noinspection SpellCheckingInspection
    def kmeans(self, x: DataFrame, k=None):
        if k is not None:
            self.k = k

        self.initialize_centroids(x)
        for _ in range(self.max_iter):
            self.labels = self._fit(x)
            self._update_centroids(x, self.labels)

        return self.centroids, self.labels

    def _fit(self, x):
        labels = []
        for i in x:
            min_distance = np.inf
            label = -1
            for idx, centroid in enumerate(self.centroids):
                distance = euclidean(centroid, i)
                if distance < min_distance:
                    min_distance = distance
                    label = idx

            labels.append(label)

        return labels

    def _update_centroids(self, x, labels):
        cluster_data = [[x[i] for i in range(x.shape[0]) if labels[i] == j] for j, _ in enumerate(self.centroids)]
        new_centers = []
        for cluster in cluster_data:
            new_centers.append(np.mean(cluster, axis=0, dtype=np.float32))

        self.centroids = new_centers
        return new_centers
