import os
import sys

sys.path.append("..")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils
from algorithms.kmeans import Kmeans

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'encoded_connect4.pkl'))

df_gs = df[df.columns[-1]]
df = df.drop(columns=df.columns[-1])

K = 3
init_method = 'k-means++'
metric = 'euclidean'
n_iter = 300
init = 10

print("Starting K-Means in Connect-4")
kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans.fit(df)
print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(12, 12))
ax = plt.subplot(111)

colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']
labels = kmeans.labels
for Class, colour in zip(range(9), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)

ax.plot(df.iloc[labels == -1, 0],
        df.iloc[labels == -1, 1],
        'k+', alpha=0.1)

centers = np.array(kmeans.centroids)
ax.scatter(centers[:, 0], centers[:, 1], marker="x", color="k")
ax.set_title(f'K-means Clustering with K={K}, init={init_method} and metric={metric}')

plt.savefig(os.path.join('..', '..', 'figures', 'connect-4', f'connect-4_k-means-{K}-{init_method}-{metric}.png'))
plt.show()


file_path = os.path.join('..', '..', 'validation', 'connect-4_k-means_val.txt')
with open(file_path, 'a') as f:
    f.write(f'\n \nK-means: K = {K}, init = {init_method}, metric = {metric}, max_inter = {n_iter}, n_init = {init}')

utils.print_metrics(df, df_gs, labels, file_path)
print("Finished K-Means in Connect-4")
