import sys

sys.path.append("..")

import os

import numpy as np
from matplotlib import pyplot as plt

import utils
from algorithms.kmeans import Kmeans

data_root_path = os.path.join('..', '..', 'datasets')
df = utils.load_arff(os.path.join(data_root_path, 'datasets', 'pen-based.arff'))

df_gs = df[df.columns[-1]]
df = df.drop(columns=df.columns[-1])

K = 10
init_method = 'k-means++'
metric = 'euclidean'
n_iter = 300
init = 10
print("Starting K-means in Pen-Based")
kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans.fit(df)
print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(12, 12))
ax = plt.subplot(111)

colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']
labels = kmeans.labels
centers = np.array(kmeans.centroids)
for Class, colour in zip(range(len(centers)), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)
    ax.scatter(centers[Class, 0], centers[Class, 1], marker="x", color=colour, s=80)

ax.plot(df.iloc[labels == -1, 0],
        df.iloc[labels == -1, 1],
        'k+', alpha=0.1)

# ax.scatter(centers[:, 0], centers[:, 1], marker="x", color="k", s=10)
ax.set_title(f'K-means clustering\nwith K={K}, init={init_method} and metric={metric}', fontsize=22)

plt.savefig(os.path.join('..', '..', 'figures', 'pen-based', f'pen-based_k-means-{K}-{init_method}-{metric}.png'))
plt.show()

true_labels = df_gs.to_numpy(dtype='int32')

file_path = os.path.join('..', '..', 'validation', 'pen-based_k-means_val.txt')
with open(file_path, 'a') as f:
    f.write(f'\n \nK-means: K = {K}, init = {init_method}, metric = {metric}, max_inter = {n_iter}, n_init = {init}')

utils.print_metrics(df, true_labels, labels, file_path)
print("Finished K-means in Pen-Based")
