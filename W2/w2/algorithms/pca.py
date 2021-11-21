import numpy as np
import pandas as pd


# Implementation of the Principal Component Analysis PCA algorithms

# VAR: is a measure of variability, how spread the dataset is.
# COV: is a measure of the extent to which corresponding elements of 2 sets of ordered data move in the same direction.


class PCA:
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

