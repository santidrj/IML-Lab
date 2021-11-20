import os
import sys
import pickle

sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import umap
from sklearn import preprocessing

from algorithms.pca import PCA as CustomPCA

path_data = os.path.join('..', '..', 'datasets', 'processed')
path_models = os.path.join('..', '..', 'models_results', 'connect-4')
path_figs = os.path.join('..', '..', 'figures', 'connect-4')

df = pd.read_pickle(os.path.join(path_data, 'connect4_encoded.pkl'))
true_class = df['class']
df = df.drop(columns=df.columns[-1])

nc = 16
n_comp = 16
n_nb = 30

color = ['r', 'g', 'b']

##### ORIGINAL DATASET #####
features = df[['a6', 'a7']].to_numpy()
unique, counts = np.unique(features, axis=0, return_counts=True)

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(df['a6'], df['a7'])
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_ylim(0-0.3, 2+0.3)
ax.set_xlim(0-0.3, 2+0.3)
for i in range(unique.shape[0]):
    ax.text(unique[i][0]+0.05, unique[i][1]+0.05, counts[i])
ax.set_title('Connect-4 Dataset')
ax.set_xlabel('a6')
ax.set_ylabel('a7')
plt.legend([r'$^{number \; of \; points}$'], loc = (0.68,0.7), fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(path_figs, 'original.png'))
plt.show()


##### SKLEARN PCA #####
pca_data = PCA(nc).fit_transform(df)

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.9, s=0.008)
ax.set_title(f'PCA of the Connect-4 Dataset with nc = {nc}')
ax.set_xlabel('first component')
ax.set_ylabel('second component')
plt.tight_layout()
plt.savefig(os.path.join(path_figs, f'pca-{nc}.png'))
plt.show()


##### SKLEARN IPCA #####
ipca_data = IncrementalPCA(n_components=nc).fit_transform(df)

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(ipca_data[:, 0], ipca_data[:, 1], alpha=0.9, s=0.008)
ax.set_title(f'Incremental PCA of the Connect-4 Dataset')
ax.set_xlabel('first component')
ax.set_ylabel('second component')
plt.tight_layout()
plt.savefig(os.path.join(path_figs, f'ipca-{nc}.png'))
plt.show()


##### CUSTOM PCA #####
custom_pca_model = CustomPCA(df, nc, cat=True)

with open(os.path.join(path_data, f'connect4_custom-pca-{nc}.pkl'), 'wb') as f:
    pickle.dump(custom_pca_model.df, f)

print(f'Number of components: {nc}')
var_ratio = custom_pca_model.explained_variance_ratio()
print(f'Variance explained by {nc} components: {round(var_ratio.sum() * 100)}%')
print(f'Cumulative variance: {np.round(np.cumsum(var_ratio), 4)}')
custom_pca_model.print_info()

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(custom_pca_model.df['PCA0'], custom_pca_model.df['PCA1'], alpha=0.9, s=0.008)
ax.set_title(f'Custom PCA of the Connect-4 dataset')
ax.set_xlabel('first component')
ax.set_ylabel('second component')
plt.tight_layout()
plt.savefig(os.path.join(path_figs, f'custom-pca-{nc}.png'))
plt.show()

##### UMAP #####
umap_data = umap.UMAP(n_neighbors=n_nb, min_dist=.0, n_components=n_comp).fit_transform(df)

with open(os.path.join(path_data, f'connect4_umap-{n_nb}-{n_comp}.pkl'), 'wb') as f:
    pickle.dump(umap_data, f)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(umap_data[:, 0], umap_data[:, 1], alpha=0.9, s=0.008)
ax.set_title(f'UMAP of the Connect-4 Dataset with \n n_neighbours = {n_nb}, n_components = {n_comp}, min_dist = 0')
ax.set_xlabel('first component')
ax.set_ylabel('second component')
plt.tight_layout()
plt.savefig(os.path.join(path_figs, f'umap-{n_nb}-{n_comp}.png'))
plt.show()