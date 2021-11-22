"""
import os
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_pen-based.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'pen-based_gs.pkl'))

df = df.drop(columns=df.columns[-1])


Model parameter selection
    decomposition.PCA:
    n_components = number of component to keep.
        the transformed dimensions of the data.

pca_model = PCA(n_components='mle').fit_transform(df)
ipca_model = IncrementalPCA(n_components=pca_model.shape[1]).fit_transform(df)
custom_pca = pca.PCA(df, 15, )
custom_pca.print_info()

save_path = os.path.join(data_root_path, 'processed')
pd.to_pickle(pd.DataFrame(pca_model), os.path.join(save_path, 'penbased_pca.pkl'))
pd.to_pickle(pd.DataFrame(ipca_model), os.path.join(save_path, 'penbased_ipca.pkl'))
pd.to_pickle(custom_pca, os.path.join(save_path, 'penbased_custom_pca.pkl'))

# Percentage of variance explained for each components
print(
    "explained variance ratio (for all components): %s"
    % str(pca.explained_variance_ratio_)
)


figs_folder_path = os.path.join('..', '..', 'figures')
sns.set(style='white', context='paper', rc={'figure.figsize': (14, 10)})
ax = plt.subplot()
ax.scatter(pca_model[:, 0], pca_model[:, 1], c=[sns.color_palette()[x] for x in df_gs])
plt.gca().set_aspect('equal', 'datalim')
ax.set_title('PCA of the Pen-based dataset', fontsize=24)
plt.savefig(os.path.join(figs_folder_path, 'penbased_pca.png'))
plt.show()

ax = plt.subplot()
ax.scatter(ipca_model[:, 0], ipca_model[:, 1], c=[sns.color_palette()[x] for x in df_gs])
ax.set_title('Incremental PCA of the Pen-based dataset', fontsize=24)
plt.savefig(os.path.join(figs_folder_path, 'penbased_ipca.png'))
plt.show()

ax = plt.subplot()
ax.scatter(custom_pca.df['PCA0'], custom_pca.df['PCA1'],
           c=[sns.color_palette()[x] for x in df_gs])
ax.set_title('Custom PCA of the Pen-based dataset', fontsize=24)

plt.savefig(os.path.join(figs_folder_path, 'penbased_custom_pca.png'))
plt.show()
"""

from scipy.io import arff
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA

import matplotlib.pyplot as plt
import seaborn as sns

#from google.colab import drive
#drive.mount('/content/drive')

dataset, meta = arff.loadarff('LOCATION OF THE DATASET OF THE PENBASED')

#from meta we will get the names and types of the columns
print(meta)
# converting into pandas dataframe
df = pd.DataFrame(dataset)

#dropping the last column because is the output label value for supervised learning.
df_gs = df['a17']
df = df.drop('a17', 1)

print("total number of missing values: ", df.isnull().sum().sum())

#visualize the characteristic of each column using describe
df.describe()

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


pca_implement = our_PCA(df, df.shape[1])
pca_implement.print_info()

df_gs = [int(i) for i in df_gs]

fig, ax = plt.subplots(3, 1, figsize=(20, 20))
axes = ax.ravel()

data = pd.DataFrame({"Ourpca_1": pca_implement.df['PCA0'], "Ourpca_2": pca_implement.df['PCA1'], "Category": df_gs})
groups = data.groupby("Category")
for name, group in groups:
  axes[0].plot(group["Ourpca_1"], group["Ourpca_2"], marker="o", linestyle="", label=name)
axes[0].set_title('Pen-based transformed using our PCA', fontsize=24)
axes[0].legend()

data = pd.DataFrame({"orginal_1": df['a1'], "original_2": df['a2'], "Category": df_gs})
groups = data.groupby("Category")
for name, group in groups:
  axes[1].plot(group["orginal_1"], group["original_2"], marker="o", linestyle="", label=name)
axes[1].set_title('Pen-based original Dataset', fontsize=24)
axes[1].legend()

recovery = pca_implement.get_reconstructed_dataset()
data = pd.DataFrame({"recovered_data_1": recovery[:,0], "recovered_data_2": recovery[:,1], "Category": df_gs})
groups = data.groupby("Category")
for name, group in groups:
  axes[2].plot(group["recovered_data_1"], group["recovered_data_2"], marker="o", linestyle="", label=name)
axes[2].set_title('Pen-based recovered Dataset', fontsize=24)
axes[2].legend()

"""
To find the variance explained by each component you should divide each component’s eigenvalue by the sum of all eigenvalues
"""
total_ev = pca_implement.get_sorted_eigenvalues().sum()
x = [i for i in range(len(pca_implement.get_sorted_eigenvalues()))]
y = [i/total_ev for i in pca_implement.get_sorted_eigenvalues()]

fig, ax = plt.subplots(figsize = (10,7))
ax.plot(x,y)
ax.set(xlabel='# number of eigen vectors', ylabel = 'percentage of variance',title="each variance percentage of the new dimensions")

# the scree method states that the number of dimensions to be selected finish when the curve starts to leveling off. in this case 6
# obviously the most signicant components is the first one, then the second and so one.

N = 6

pca_model = PCA(n_components=N).fit_transform(df)
ipca_model = IncrementalPCA(n_components=N).fit_transform(df)
custom_pca = our_PCA(df, N)

df_gs = [int(i) for i in df_gs]

fig, ax = plt.subplots(3, 1, figsize=(20,20))
axes = ax.ravel()

data = pd.DataFrame({"pca_1": pca_model[:, 0], "pca_2": pca_model[:, 1], "Category": df_gs})
groups = data.groupby("Category")
for name, group in groups:
  axes[0].plot(group["pca_1"], group["pca_2"], marker="o", linestyle="", label=name)
axes[0].set_title('Pen-based reduced using sklearn PCA', fontsize=24)
axes[0].legend()

data = pd.DataFrame({"IncrementalPca_1": ipca_model[:, 0], "IncrementalPca_2": ipca_model[:, 1], "Category": df_gs})
groups = data.groupby("Category")
for name, group in groups:
  axes[1].plot(group["IncrementalPca_1"], group["IncrementalPca_2"], marker="o", linestyle="", label=name)
axes[1].set_title('Pen-based reduced using sklearn Incremental PCA', fontsize=24)
axes[1].legend()
data = pd.DataFrame({"Ourpca_1": custom_pca.df['PCA0'], "Ourpca_2": custom_pca.df['PCA1'], "Category": df_gs})
groups = data.groupby("Category")
for name, group in groups:
  axes[2].plot(group["Ourpca_1"], group["Ourpca_2"], marker="o", linestyle="", label=name)
axes[2].set_title('Pen-based reduced using our PCA', fontsize=24)
axes[2].legend()

