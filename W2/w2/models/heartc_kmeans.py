import sys

sys.path.append("..")

import os.path

import numpy as np
import pandas as pd
import seaborn as sns
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

print("Starting K-means in Heart-C")
kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
kmeans.fit(df)
print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)

labels = kmeans.labels
centers = np.array(kmeans.centroids)
colors = sns.color_palette()[:len(centers)]
for Class, colour in zip(range(len(centers)), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)
    ax.scatter(centers[Class, 0], centers[Class, 1], marker="x", color=colour, s=80)
    ax.set_xlabel(df.columns[0], fontsize=18)
    ax.set_ylabel(df.columns[1], fontsize=18)

ax.plot(df.iloc[labels == -1, 0],
        df.iloc[labels == -1, 1],
        'k+', alpha=0.1)

ax.set_title(f'K-means clustering in full Heart-C', fontsize=22)

plt.savefig(os.path.join('..', '..', 'figures', 'heart-c', f'full-heart-c_k-means.png'))
plt.show()


# counts = np.bincount(labels.astype(int))
# maj_class = counts.argmax()
# min_class = counts.argmin()
# df_gs.replace({'<50': maj_class, '>50_1': min_class}, inplace=True)

file_path = os.path.join('..', '..', 'validation', 'heart-c_k-means_val.txt')
with open(file_path, 'a') as f:
    f.write(f'\n \nK-means: K = {K}, init = {init_method}, metric = {metric} max_inter = {n_iter}, n_init = {init}')

utils.print_metrics(df, df_gs, labels, file_path)
print("Finished K-means in Heart-C")

## Run K-means in reduced Heart-C

df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_custom_pca.pkl'))

print("Starting K-means in reduced Heart-C")
kmeans.fit(df)
print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)

labels = kmeans.labels.astype(np.int32)
centers = np.array(kmeans.centroids)
colors = sns.color_palette()[:len(centers)]
for Class, colour in zip(range(len(centers)), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)
    ax.scatter(centers[Class, 0], centers[Class, 1], marker="x", color=colour, s=80)
    ax.set_xlabel(df.columns[0], fontsize=18)
    ax.set_ylabel(df.columns[1], fontsize=18)

ax.plot(df.iloc[labels == -1, 0],
        df.iloc[labels == -1, 1],
        'k+', alpha=0.1)

ax.set_title('K-means clustering in reduced Heart-C', fontsize=22)
plt.savefig(os.path.join('..', '..', 'figures', 'heart-c', 'reduced-heart-c_k-means.png'))
plt.show()

file_path = os.path.join('..', '..', 'validation', 'reduced-heart-c_k-means_val.txt')
with open(file_path, 'a') as f:
    f.write(f'\n \nK-means: K = {K}, init = {init_method}, metric = {metric} max_inter = {n_iter}, n_init = {init}')

utils.print_metrics(df, df_gs, labels, file_path)
print("Finished K-means in reduced Heart-C")
