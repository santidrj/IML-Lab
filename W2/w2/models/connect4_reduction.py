import os
import sys
import pickle

sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA
import umap

from algorithms.pca import PCA as CustomPCA

path_data = os.path.join('..', '..', 'datasets', 'processed')
path_models = os.path.join('..', '..', 'models_results', 'connect-4')
path_figs = os.path.join('..', '..', 'figures', 'connect-4')

df = pd.read_pickle(os.path.join(path_data, 'connect4_encoded.pkl'))
df = df.drop(columns=df.columns[-1])
print(df.to_numpy().dtype)

nc = 4

##### ORIGINAL DATASET #####
features = df[['a6', 'a7']].to_numpy()
unique, counts = np.unique(features, axis=0, return_counts=True)
fig, ax = plt.subplots(figsize=(6,5))
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
plt.show()


##### SKLEARN PCA #####
pca_data = PCA(n_components=nc).fit_transform(df)
with open(os.path.join(path_data, f'connect4_pca-{nc}.pkl'), 'wb') as f:
    pickle.dump(pca_data, f)
with open(os.path.join(path_data, f'connect4_pca-{nc}.pkl'), 'rb') as f:
    pca_data = pickle.load(f)

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.9, s=0.008)
ax.set_title(f'sklearn PCA reduction with nc = {nc}')
ax.set_xlabel('first component')
ax.set_ylabel('second component')
plt.tight_layout()
plt.savefig(os.path.join(path_figs, f'pca-{nc}.png'))
plt.show()


##### SKLEARN IPCA #####
ipca_data = IncrementalPCA(n_components=nc).fit_transform(df)
with open(os.path.join(path_data, f'connect4_ipca-{nc}.pkl'), 'wb') as f:
    pickle.dump(ipca_data, f)
with open(os.path.join(path_data, f'connect4_ipca-{nc}.pkl'), 'rb') as f:
    ipca_data = pickle.load(f)

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(ipca_data[:, 0], ipca_data[:, 1], alpha=0.9, s=0.008)
ax.set_title(f'sklearn IPCA reduction with nc = {nc}')
ax.set_xlabel('first component')
ax.set_ylabel('second component')
plt.tight_layout()
plt.savefig(os.path.join(path_figs, f'ipca-{nc}.png'))
plt.show()


##### CUSTOM PCA #####
custom_pca_model = CustomPCA(df, nc)
with open(os.path.join(path_data, f'connect4_our-pca-{nc}.pkl'), 'wb') as f:
    pickle.dump(custom_pca_model.df, f)
with open(os.path.join(path_data, f'connect4_our-pca-{nc}.pkl'), 'rb') as f:
    custom_pca_df = pickle.load(f)

print(custom_pca_df.head())

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(custom_pca_df['PCA0'], custom_pca_df['PCA1'], alpha=0.9, s=0.008)
ax.set_title(f'Our PCA reduction with nc = {nc}')
ax.set_xlabel('first component')
ax.set_ylabel('second component')
plt.tight_layout()
plt.savefig(os.path.join(path_figs, f'our-pca-{nc}.png'))
plt.show()


##### UMAP #####
umap_model = umap.UMAP(n_components=nc).fit_transform(df)
with open(os.path.join(path_data, f'connect4_umap-{nc}.pkl'), 'wb') as f:
    pickle.dump(umap_model.embedding_, f)
with open(os.path.join(path_data, f'connect4_umap-{nc}.pkl'), 'rb') as f:
    umap_data = pickle.load(f)

fig, ax = plt.subplots(figsize=(6,5))
ax.scatter(umap_data[:, 0], umap_data[:, 1], alpha=0.9, s=0.008)
ax.set_title(f'UMAP reduction with nc = {nc}')
ax.set_xlabel('first component')
ax.set_ylabel('second component')
plt.tight_layout()
plt.savefig(os.path.join(path_figs, f'umap-{nc}.png'))
plt.show()




