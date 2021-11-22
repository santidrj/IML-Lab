import os
import sys
import pickle
import time

sys.path.append("..")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils
from algorithms.kmeans import Kmeans

path_data = os.path.join('..', '..', 'datasets', 'processed')
path_models = os.path.join('..', '..', 'models_results', 'connect-4')
path_val = os.path.join('..', '..', 'validation')

K = 3
init_method = 'k-means++'
metric = 'euclidean'
n_iter = 300
init = 10
nc = 16
n_comp = 16
n_nb = 30

print("Starting K-Means in Connect-4")

df = pd.read_pickle(os.path.join(path_data, 'connect4_encoded.pkl'))
true_class = df[df.columns[-1]]
df = df.drop(columns=df.columns[-1])

with open(os.path.join(path_data, f'connect4_custom-pca-{nc}.pkl'), 'rb') as f:
    df_custom_pca = pd.DataFrame(pickle.load(f))

with open(os.path.join(path_data, f'connect4_umap-{n_nb}-{n_comp}.pkl'), 'rb') as f:
    df_umap = pd.DataFrame(pickle.load(f))

kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)

##### K-MEANS WITH ORIGINAL DATASET #####
start_kmeans = time.time()
kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans.fit(df)
end_kmeans = time.time()

with open(os.path.join(path_models, f'k-means.pkl'), 'wb') as f:
    pickle.dump(kmeans, f)

with open(os.path.join(path_models, f'k-means.pkl'), 'rb') as f:
    kmeans = pickle.load(f)


##### K-MEANS WITH CUSTOM PCA REDUCED DATASET #####
start_kmeans_pca = time.time()
kmeans_pca = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans_pca.fit(df_custom_pca)
end_kmeans_pca = time.time()

with open(os.path.join(path_models, f'k-means-pca.pkl'), 'wb') as f:
    pickle.dump(kmeans_pca, f)

with open(os.path.join(path_models, f'k-means-pca.pkl'), 'rb') as f:
    kmeans_pca = pickle.load(f)


##### K-MENAS WITH UMAP REDUCED DATASET #####

start_kmeans_umap = time.time()
kmeans_umap = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans_umap.fit(df_umap)
end_kmeans_umap = time.time()

with open(os.path.join(path_models, f'k-means-umap.pkl'), 'wb') as f:
    pickle.dump(kmeans_umap, f)

with open(os.path.join(path_models, f'k-means-umap.pkl'), 'rb') as f:
    kmeans_umap = pickle.load(f)

models = [kmeans, kmeans_pca, kmeans_umap]
models_names = ['k-means', 'k-means-pca', 'k-means-umap']
dfs = [df, df_custom_pca, df_umap]
models_times = [end_kmeans - start_kmeans, end_kmeans_pca - start_kmeans_pca, end_kmeans_umap - start_kmeans_umap]


for model, name, df, time in zip(models, models_names, dfs, models_times):

    labels = model.labels
    unique, counts = np.unique(labels, return_counts=True)
    counts = counts / len(labels)

    plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)
    colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']

    for Class, colour in zip(range(9), colors):
        Xk = df[labels == Class]
        if name=='k-means':
            ax.scatter(Xk.iloc[:, 5], Xk.iloc[:, 6], color=colour)
            ax.set_xlabel('a6')
            ax.set_ylabel('a7')
            ax.set_xticks([0, 1, 2])
            ax.set_yticks([0, 1, 2])
            ax.set_ylim(0 - 0.3, 2 + 0.3)
            ax.set_xlim(0 - 0.3, 2 + 0.3)
        else:
            ax.scatter(Xk.iloc[:, 0], Xk.iloc[:, 1], color=colour, alpha=0.9, s=0.008)
            ax.set_xlabel('first component')
            ax.set_ylabel('second component')

    centers = np.array(model.centroids)
    if name=='k-means':
        ax.scatter(centers[:, 5], centers[:, 6], marker="x", color="k")
    else:
        ax.scatter(centers[:, 0], centers[:, 1], marker="x", color="k")

    if name=='k-means':
        ax.set_title(f'Clustering of the Full dataset')
    if name == 'k-means-pca':
        ax.set_title(f'Clustering of the custom PCA reduction')
    if name == 'k-means-umap':
        ax.set_title(f'Clustering of the UMAP reduction')
    
    plt.savefig(os.path.join('..', '..', 'figures', 'connect-4', f'connect-4_{name}.png'))
    plt.show()

    with open(os.path.join(path_val, 'connect-4_k-means_val.txt'), 'a') as f:
        f.write(f'\n \n{name}: K = {K}, init = {init_method}, metric = {metric}, max_inter = {n_iter}, n_init = {init}')
        f.write(f'pca components = {nc}, n_neighbors = {n_nb}, n_components = {n_comp}')
        f.write(f'\nPercentage of samples in each cluster: {counts}')
        f.write(f'\nExecution time: {time} s')

    utils.print_metrics(df, true_class, labels, os.path.join(path_val, 'connect-4_k-means_val.txt'))

print("Finished K-Means in Connect-4")
