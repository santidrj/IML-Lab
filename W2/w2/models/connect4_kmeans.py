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

print("Starting K-Means in Connect-4")

df = pd.read_pickle(os.path.join(path_data, 'connect4_encoded.pkl'))
true_class = df[df.columns[-1]]
df = df.drop(columns=df.columns[-1])
"""
df_custom_pca = pd.read_pickle(os.path.join(path_data, 'connect4_custom-pca-4.pkl'))

#with open(os.path.join(path_data, f'connect4_umap-30-10.pkl'), 'rb') as f:
    df_umap = pd.DataFrame(pickle.load(f))

##### K-MEANS WITH ORIGINAL DATASET #####
start_kmeans = time.time()
kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans.fit(df)
end_kmeans = time.time()

with open(os.path.join(path_models, f'k-means.pkl'), 'wb') as f:
    pickle.dump(kmeans, f)

with open(os.path.join(path_models, f'k-means.pkl'), 'rb') as f:
    kmeans = pickle.load(f)

##### K-MENAS WITH CUSTOM PCA REDUCED DATASET #####
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

with open(os.path.join(path_models, f'k-means-umap-{K}-{init_method}-{metric}.pkl'), 'wb') as f:
    pickle.dump(kmeans_umap, f)

#with open(os.path.join(path_models, f'k-means-umap-{K}-{init_method}-{metric}.pkl'), 'rb') as f:
    kmeans_umap = pickle.load(f)

models = [kmeans, kmeans_pca, kmeans_umap]
models_names = ['k-means', 'k-means-pca', 'k-means-umap']
models_times = [end_kmeans-start_kmeans, end_kmeans_pca-start_kmeans_pca, end_kmeans_umap-start_kmeans_umap]

for model, name, time in zip(models, models_names, models_times):
    print(f'Sum of squared error: {model.square_error}')
    
    plt.figure(figsize=(6, 5))
    ax = plt.subplot(111)
    
    colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']
    labels = model.labels
    for Class, colour in zip(range(9), colors):
        Xk = df[labels == Class]
        ax.plot(Xk.iloc[:, 6], Xk.iloc[:, 7], 'o', color=colour, alpha=0.3)
    
    ax.plot(df.iloc[labels == -1, 6], df.iloc[labels == -1, 7], 'k+', alpha=0.1)
    
    centers = np.array(model.centroids)
    ax.scatter(centers[:, 6], centers[:, 7], marker="x", color="k")
    #ax.set_title(f'K-means Clustering with K={K}, init={init_method} and metric={metric}')
    
    plt.savefig(os.path.join('..', '..', 'figures', 'connect-4', f'connect-4_{name}-{K}-{init_method}-{metric}.png'))
    plt.show()

    with open(os.path.join(path_val, 'connect-4_k-means_val.txt'), 'a') as f:
        f.write(f'\n \n{name}: K = {K}, init = {init_method}, metric = {metric}, max_inter = {n_iter}, n_init = {init}')
        f.write(f'\nExecution time: {time} s')
    
    utils.print_metrics(df, true_class, labels, path_val)
"""

for n_comp in [10, 20, 30]:
    for n_nb in [30, 60, 90]:
        with open(os.path.join(path_data, f'connect4_umap-{n_nb}-{n_comp}.pkl'), 'rb') as f:
            df_umap = pd.DataFrame(pickle.load(f))

        kmeans_umap = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
        kmeans_umap.fit(df_umap)

        with open(os.path.join(path_models, f'k-means-umap-{n_nb}-{n_comp}.pkl'), 'wb') as f:
            pickle.dump(kmeans_umap, f)

        with open(os.path.join(path_val, 'connect-4_k-means_val.txt'), 'a') as f:
            f.write(
                f'\n \nUMAP: n_neighbors = {n_nb}, n_components = {n_comp}, min_dist = 0')

        utils.print_metrics(df, true_class, kmeans_umap.labels, os.path.join(path_val, 'connect-4_k-means_val.txt'))


print("Finished K-Means in Connect-4")
