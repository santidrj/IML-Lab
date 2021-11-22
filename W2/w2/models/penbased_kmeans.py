"""
import sys

sys.path.append("..")

import os

import numpy as np
from matplotlib import pyplot as plt

import utils
from algorithms.kmeans import Kmeans

data_root_path = os.path.join('..', '..', 'datasets')
df = utils.load_arff(os.path.join(data_root_path, 'datasets', 'pen-based.arff'))

df_gs = df[df.columns[-1]]
df = df.drop(columns=df.columns[-1])

K = 10
init_method = 'k-means++'
metric = 'euclidean'
n_iter = 300
init = 10
print("Starting K-means in Pen-Based")
kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans.fit(df)
print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(12, 12))
ax = plt.subplot(111)

colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']
labels = kmeans.labels
centers = np.array(kmeans.centroids)
for Class, colour in zip(range(len(centers)), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)
    ax.scatter(centers[Class, 0], centers[Class, 1], marker="x", color=colour, s=80)

ax.plot(df.iloc[labels == -1, 0],
        df.iloc[labels == -1, 1],
        'k+', alpha=0.1)

# ax.scatter(centers[:, 0], centers[:, 1], marker="x", color="k", s=10)
ax.set_title(f'K-means clustering\nwith K={K}, init={init_method} and metric={metric}', fontsize=22)

plt.savefig(os.path.join('..', '..', 'figures', 'pen-based', f'pen-based_k-means-{K}-{init_method}-{metric}.png'))
plt.show()

true_labels = df_gs.to_numpy(dtype='int32')

file_path = os.path.join('..', '..', 'validation', 'pen-based_k-means_val.txt')
with open(file_path, 'a') as f:
    f.write(f'\n \nK-means: K = {K}, init = {init_method}, metric = {metric}, max_inter = {n_iter}, n_init = {init}')

utils.print_metrics(df, true_labels, labels, file_path)
print("Finished K-means in Pen-Based")
"""

import time
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.io import arff
from pandas import DataFrame
from numpy import arange
from numpy.random import random
from scipy.spatial.distance import euclidean, minkowski
#!pip install umap-learn

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})



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
class our_PCA:
    """
    PCA finds a new set of dimensions which are orthogonal hence linearly independent
     and ranked wrt the variability of it. that is the most important principal axis is first.

      steps:
      1. Calculate the covariance matrix of X.
      2. Calculate the eigen vectors and eigen values.
      3. sort the eigen vectors wrt to the eigen values.
      4. pick the first k eigen vectors. these will be the new dimensions
      5. transform the n dimensional data into k dimensions.

      Params:
        X: the data to be fed, must be normalized
        k: desire number of dimensions
          MUST k <= d
        cat: True when dealing with encoded categorical data

      Returns:
        X': the transformed X input dataset
    """

    def __init__(self, dataset, k, cat=False):
        self.cat = cat
        self.dataset = dataset.to_numpy()
        self.k = k  # k desired dimensions.
        self.dimensions = self.dataset.shape  # (rows, column)
        self._data = None  # center dataset.
        self._covariance_matrix = None
        self._eigen_values = None
        self._eigen_vectors = None
        self._eigen_values_srt = None
        self._eigen_vectors_srt = None
        self._data_transformed = None
        self.df = None
        self.recon_dataset = None
        if self.k > self.dimensions[1]:
            print("k needs to be smaller to the dimensions of the dataset in order to make sense")
        elif self.k == 0:
            print("k must be bigger than 0, there must be a dimension")
        else:
            self._pca()

    def print_info(self):
        print('--CUSTOM PCA--')
        print('\nCovariance matrix:')
        print(self.get_covariance_matrix())
        print('\nEigenvectors:')
        print(self.get_eigen_vectors())
        print('\nEigenvalues')
        print(self.get_eigenvalues())
        print('\nSorted eigenvectors:')
        print(self.get_sorted_eigen_vectors())
        print('\nSorted eigenvalues')
        print(self.get_sorted_eigenvalues())

    def get_center_data(self):
        return self._data

    def get_covariance_matrix(self):
        return self._covariance_matrix

    def get_eigenvalues(self):
        return self._eigen_values

    def get_eigen_vectors(self):
        return self._eigen_vectors

    def get_sorted_eigenvalues(self):
        return self._eigen_values_srt

    def get_sorted_eigen_vectors(self):
        return self._eigen_vectors_srt

    def get_data_transformed(self):
        return self._data_transformed

    def get_df_transformed(self):
        return self.df

    def get_reconstructed_dataset(self):
        return self.recon_dataset

    def explained_variance_ratio(self):
        eigenvalues = self.get_sorted_eigenvalues()
        return eigenvalues[:self.k]/eigenvalues.sum()

    def _pca(self):
        # step 1
        dataset_mean = np.mean(self.dataset, axis=0)
        self._data = self.dataset - dataset_mean
        self._covariance_matrix = (1 / self.dimensions[0]) * np.dot(np.transpose(self._data), self._data)
        # step 2
        self._eigen_values, self._eigen_vectors = np.linalg.eigh(self._covariance_matrix)
        # step 3
        i = np.argsort(self._eigen_values)[::-1]
        self._eigen_values_srt = self._eigen_values[i]
        self._eigen_vectors_srt = self._eigen_vectors[:, i]
        # step 4
        self._data_transformed = np.transpose(np.dot(np.transpose(self._eigen_vectors_srt[:, :self.k]), np.transpose(self._data)))
        # optional convert numpy to df
        self.df = pd.DataFrame(data=self._data_transformed, index=[i for i in range(self._data_transformed.shape[0])],
                               columns=['PCA' + str(i) for i in range(self._data_transformed.shape[1])])
        # step 5
        # reconstruction of the dataset
        self.recon_dataset = np.transpose(np.dot(self._eigen_vectors_srt[:, :self.k], np.transpose(self._data_transformed))) + dataset_mean
        # transform decimals into integers to reconstruct the encoded categorical dataset
        if self.cat:
            self.recon_dataset = np.round(self.recon_dataset)
            # number of mismatches wrt the original dataset
            mismat = self.recon_dataset.size - (self.dataset == self.recon_dataset).sum()
            perc_mismat = np.round(mismat*100/self.recon_dataset.size).astype(np.int)
            print(f"Mismatches in the reconstructed categorical dataset: {perc_mismat}% [{mismat}/{self.recon_dataset.size}]")

K = 10
init_method = 'k-means++'
metric = 'euclidean'
n_iter = 300
init = 10

#LOCATION OF PENBASED DATASET
dataset, meta = arff.loadarff('LOCATION OF PENBASED ARFF FILE')

#from meta we will get the names and types of the columns
print(meta)
# converting into pandas dataframe
df = pd.DataFrame(dataset)

#dropping the last column because is the output label value for supervised learning.
df_gs = df['a17']
df = df.drop('a17', 1)
df_gs = [int(i) for i in df_gs]

custom_pca = our_PCA(df, 6)
#########################################
#     K MEANS WITHOUT DATA REDUCTION
#########################################
start_kmeans = time.time()
kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans.fit(df)
finish_kmeans = time.time()
print(f'Sum of squared error: {kmeans.square_error}')


#########################################
#     K MEANS WITH DATA REDUCTION
#########################################
start_kmeans_reduced = time.time()
kmeans_reduced = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans_reduced.fit(custom_pca.df)
finish_kmeans_reduced = time.time()
print(f'Sum of squared error: {kmeans_reduced.square_error}')

#########################################
#    PLOT RESULTS USING UMAP
#########################################
original_reducer = umap.UMAP()
pca_reducer = umap.UMAP()

original_embedding = reducer.fit_transform(df)
pca_embedding = reducer.fit_transform(custom_pca.df)

# UMAP reduces down to 2D.
print("Time diferences:")
print("Without data reduction total time of execution: ", finish_kmeans - start_kmeans, "seconds")
print("With data reduction total time of execution: ", finish_kmeans_reduced - start_kmeans_reduced, "seconds")

fig, ax = plt.subplots(figsize = (10,10))
data = pd.DataFrame({"orginal_1": original_embedding[:, 0], "original_2": original_embedding[:, 1], "Category":  kmeans.labels})
groups = data.groupby("Category")
for name, group in groups:
  ax.plot(group["orginal_1"], group["original_2"], marker="o", linestyle="", label=name)
ax.set_title('UMAP projection of the k means original-Pen-based dataset', fontsize=24)
ax.legend()

fig, ax = plt.subplots(figsize = (10,10))
data = pd.DataFrame({"orginal_1": pca_embedding[:, 0], "original_2": pca_embedding[:, 1], "Category":  kmeans_reduced.labels})
groups = data.groupby("Category")
for name, group in groups:
  ax.plot(group["orginal_1"], group["original_2"], marker="o", linestyle="", label=name)
ax.set_title('UMAP projection of the k means original-Pen-based dataset', fontsize=24)
ax.legend()


# without data reduction
#internals
ch_score = metrics.calinski_harabasz_score(df, kmeans.labels)
db_score = metrics.davies_bouldin_score(df, kmeans.labels)

#external
adj_mutual_info_sc = metrics.adjusted_mutual_info_score(df_gs, kmeans.labels)
fm_score = metrics.fowlkes_mallows_score(df_gs, kmeans.labels)

#with data reduction
#internals
rch_score = metrics.calinski_harabasz_score(custom_pca.df, kmeans_reduced.labels)
rdb_score = metrics.davies_bouldin_score(custom_pca.df, kmeans_reduced.labels)
#external
radj_mutual_info_sc = metrics.adjusted_mutual_info_score(df_gs, kmeans_reduced.labels)
rfm_score = metrics.fowlkes_mallows_score(df_gs, kmeans_reduced.labels)


print("Without data reduction")
print("Internal Validations: ")
print("\t1) Calinski-Harabasz score: ", ch_score)
print("\t2) Davies-Bouldin score: ", db_score)
print("\nExternal Validations: ")
print("\tAdjusted Mutual Information score (from 0 to 1): ", adj_mutual_info_sc)
print("\tFowlkes-Mallows score (from 0 to 1):", fm_score)

print("\nWith data reduction")
print("Internal Validations: ")
print("\t1) Calinski-Harabasz score: ", rch_score)
print("\t2) Davies-Bouldin score: ", rdb_score)
print("\nExternal Validations: ")
print("\tAdjusted Mutual Information score (from 0 to 1): ", radj_mutual_info_sc)
print("\tFowlkes-Mallows score (from 0 to 1):", rfm_score)