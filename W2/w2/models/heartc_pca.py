import os
import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA

from algorithms import pca

N = 7

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_heart-c.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_gs.pkl'))

pca_model = PCA(n_components=N).fit_transform(df)
ipca_model = IncrementalPCA(n_components=N).fit_transform(df)
custom_pca = pca.PCA(df, N)
custom_pca.print_info()
print(f'\nPercentage of variance conserved after PCA: {custom_pca.explained_variance_ratio().sum()}')

save_path = os.path.join(data_root_path, 'processed')
pd.to_pickle(pd.DataFrame(pca_model), os.path.join(save_path, 'heart-c_pca.pkl'))
pd.to_pickle(pd.DataFrame(ipca_model), os.path.join(save_path, 'heart-c_ipca.pkl'))
pd.to_pickle(custom_pca.df, os.path.join(save_path, 'heart-c_custom_pca.pkl'))

figs_folder_path = os.path.join('..', '..', 'figures', 'heart-c')
sns.set(style='white', context='paper', rc={'figure.figsize': (14, 10)})
ax = plt.subplot()
ax.scatter(pca_model[:, 0], pca_model[:, 1], c=[sns.color_palette()[x] for x in df_gs.map({'<50': 0, '>50_1': 1})])
plt.gca().set_aspect('equal', 'datalim')
ax.set_title('Heart-C reduced using sklearn PCA', fontsize=24)
plt.savefig(os.path.join(figs_folder_path, 'heartc_pca.png'))
plt.show()

ax = plt.subplot()
ax.scatter(ipca_model[:, 0], ipca_model[:, 1], c=[sns.color_palette()[x] for x in df_gs.map({'<50': 0, '>50_1': 1})])
ax.set_title('Heart-C reduced using sklearn Incremental PCA', fontsize=24)
plt.savefig(os.path.join(figs_folder_path, 'heartc_ipca.png'))
plt.show()

ax = plt.subplot()
ax.scatter(custom_pca.df['PCA0'], custom_pca.df['PCA1'],
           c=[sns.color_palette()[x] for x in df_gs.map({'<50': 0, '>50_1': 1})])
ax.set_title('Heart-C reduced using our PCA', fontsize=24)
plt.savefig(os.path.join(figs_folder_path, 'heartc_custom_pca.png'))
plt.show()
