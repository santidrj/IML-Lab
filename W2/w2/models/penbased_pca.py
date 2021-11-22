import os
import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA

from algorithms.pca import PCA as our_PCA

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_pen-based.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'pen-based_gs.pkl'))

df = df.drop(columns=df.columns[-1])

pca_implement = our_PCA(df, df.shape[1])
pca_implement.print_info()

df_gs = [int(i) for i in df_gs]

figs_folder_path = os.path.join('..', '..', 'figures', 'pen-based')
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
data = pd.DataFrame({"recovered_data_1": recovery[:, 0], "recovered_data_2": recovery[:, 1], "Category": df_gs})
groups = data.groupby("Category")
for name, group in groups:
    axes[2].plot(group["recovered_data_1"], group["recovered_data_2"], marker="o", linestyle="", label=name)
axes[2].set_title('Pen-based recovered Dataset', fontsize=24)
axes[2].legend()
plt.savefig(os.path.join(figs_folder_path, 'penbased_custom_pca.png'))
plt.show()

"""
To find the variance explained by each component you should divide each component’s eigenvalue by the sum of all eigenvalues
"""
total_ev = pca_implement.get_sorted_eigenvalues().sum()
x = [i for i in range(len(pca_implement.get_sorted_eigenvalues()))]
y = [i / total_ev for i in pca_implement.get_sorted_eigenvalues()]

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(x, y)
ax.set(xlabel='# number of eigen vectors', ylabel='percentage of variance',
       title="Each variance percentage of the new dimensions")
plt.savefig(os.path.join(figs_folder_path, 'penbased_variance_explained.png'))
plt.show()

# the scree method states that the number of dimensions to be selected finish when the curve starts to leveling off. in this case 6
# obviously the most signicant components is the first one, then the second and so one.

N = 6

pca_model = PCA(n_components=N).fit_transform(df)
ipca_model = IncrementalPCA(n_components=N).fit_transform(df)
custom_pca = our_PCA(df, N)
save_path = os.path.join(data_root_path, 'processed')
pd.to_pickle(custom_pca.df, os.path.join(save_path, 'pen-based_custom_pca.pkl'))

df_gs = [int(i) for i in df_gs]

fig, ax = plt.subplots(3, 1, figsize=(20, 20))
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
plt.savefig(os.path.join(figs_folder_path, 'penbased_pca.png'))
plt.show()
