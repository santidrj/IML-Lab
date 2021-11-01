import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils
from algorithms.kmeans import Kmeans

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_heart-c.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_gs.pkl'))

K = 2
init_method = 'k-means++'
metric = 'euclidean'
n_iter = 300
init = 10
kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans.fit(df)
print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)

colors = ['r', 'g']
labels = kmeans.labels
centers = np.array(kmeans.centroids)
for Class, colour in zip(range(len(centers)), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)
    ax.scatter(centers[Class, 0], centers[Class, 1], marker="x", color=colour, s=80)
    ax.set_xlabel(df.columns[0], fontsize=18)
    ax.set_ylabel(df.columns[1], fontsize=18)

ax.plot(df.iloc[labels == -1, 0],
        df.iloc[labels == -1, 1],
        'k+', alpha=0.1)

ax.set_title(f'K-means clustering\nwith K={K}, init={init_method} and metric={metric}', fontsize=22)

plt.savefig(os.path.join('..', '..', 'figures', 'heart-c', f'heart-c_k-means-{K}-{init_method}-{metric}.png'))
plt.show()


counts = np.bincount(labels.astype(int))
maj_class = counts.argmax()
min_class = counts.argmin()
df_gs.replace({'<50': maj_class, '>50_1': min_class}, inplace=True)

file_path = os.path.join('..', '..', 'validation', 'heart-c_k-means_val.txt')
with open(file_path, 'a') as f:
    f.write(f'\n \nK-means: K = {K}, init = {init_method}, metric = {metric} max_inter = {n_iter}, n_init = {init}')

utils.print_metrics(df, df_gs, labels, file_path)
