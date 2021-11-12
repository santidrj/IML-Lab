import os
import sys
import pickle

sys.path.append("..")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils
from algorithms.kmeans import Kmeans

path_data = os.path.join('..', '..', 'datasets', 'processed')
path_models = os.path.join('..', '..', 'models_results', 'connect-4')

K = 3
init_method = 'k-means++'
metric = 'euclidean'
n_iter = 300
init = 10

print("Starting K-Means in Connect-4")
# K-means with complete dataset
df = pd.read_pickle(os.path.join(path_data, 'connect4_encoded.pkl'))
df_gs = df[df.columns[-1]]
df = df.drop(columns=df.columns[-1])

kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans.fit(df)

with open(os.path.join(path_models, f'k-means-{K}-{init_method}-{metric}.pkl'), 'wb') as f:
    pickle.dump(kmeans, f)

with open(os.path.join(path_models, f'k-means-{K}-{init_method}-{metric}.pkl'), 'rb') as f:
    kmeans = pickle.load(f)


"""
# K-means with PCA reduced dataset
#with open(os.path.join(path_data, 'connect4_pca-reduced-3.pkl'), 'rb') as f:
    df_pca_reduced = pickle.load(f)
    
kmeans_pca = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans_pca.fit(df_pca_reduced)
np.save(os.path.join('..', '..', 'models_results', 'connect-4', f'connect-4_k-means-pca-{K}-{init_method}-{metric}.npy'), kmeans_pca)
kmeans_pca = np.load(os.path.join('..', '..', 'models_results', 'connect-4', f'connect-4_k-means-pca-{K}-{init_method}-{metric}.npy'))

# K-means with UMAP reduced dataset

models = [kmeans, kmeans_pca]
models_names = ['k-means', 'k-means-pca']
"""

models = [kmeans]
models_names = ['k-means']

for model, name in zip(models, models_names):
    print(f'Sum of squared error: {model.square_error}')
    
    plt.figure(figsize=(12, 12))
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
    
    
    file_path = os.path.join('..', '..', 'validation', 'connect-4_k-means_val.txt')
    with open(file_path, 'a') as f:
        f.write(f'\n \n{name}: K = {K}, init = {init_method}, metric = {metric}, max_inter = {n_iter}, n_init = {init}')
    
    utils.print_metrics(df, df_gs, labels, file_path)
    print("Finished K-Means in Connect-4")
