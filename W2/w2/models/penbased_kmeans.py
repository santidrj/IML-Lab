import sys
import time

import pandas as pd

sys.path.append("..")

import os

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import utils
from algorithms.kmeans import Kmeans

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_pen-based.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'pen-based_gs.pkl'))

K = 10
init_method = 'k-means++'
metric = 'euclidean'
n_iter = 300
init = 10
print("Starting K-means in Pen-Based")
kmeans = Kmeans(k=K, init=init_method, metric=metric, max_iter=n_iter, n_init=init)
start_kmeans = time.time()
kmeans.fit(df)
finish_kmeans = time.time()
print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(12, 12))
ax = plt.subplot(111)

colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']
labels = kmeans.labels
centers = np.array(kmeans.centroids)
for Class, colour in zip(range(len(centers)), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)
    ax.scatter(centers[Class, 0], centers[Class, 1], marker="x", color=colour, s=50)

ax.plot(df.iloc[labels == -1, 0],
        df.iloc[labels == -1, 1],
        'k+', alpha=0.1)

ax.set_title(f'K-means clustering in original Pen-Based', fontsize=22)

plt.savefig(os.path.join('..', '..', 'figures', 'pen-based', f'original-pen-based_k-means.png'))
plt.show()

true_labels = df_gs.to_numpy(dtype='int32')

file_path = os.path.join('..', '..', 'validation', 'pen-based_k-means_val.txt')
with open(file_path, 'a') as f:
    f.write(
        f'\n \nK-means: K = {K}, init = {init_method}, metric = {metric}, max_inter = {n_iter}, n_init = {init}, time = {start_kmeans - finish_kmeans}')

utils.print_metrics(df, true_labels, labels, file_path)
print("Finished K-means in Pen-Based")

#########################################
#     K MEANS WITH DATA REDUCTION
#########################################
print("Starting K-means in reduced Pen-Based with PCA")
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'pen-based_custom_pca.pkl'))

start_kmeans = time.time()
kmeans.fit(df)
finish_kmeans = time.time()

print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)

labels = kmeans.labels.astype(np.int32)
centers = np.array(kmeans.centroids)
colors = sns.color_palette()[:len(centers)]
for Class, colour in zip(range(len(centers)), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)
    ax.scatter(centers[Class, 0], centers[Class, 1], marker="x", color=colour, s=50)
    ax.set_xlabel(df.columns[0], fontsize=18)
    ax.set_ylabel(df.columns[1], fontsize=18)

ax.plot(df.iloc[labels == -1, 0],
        df.iloc[labels == -1, 1],
        'k+', alpha=0.1)

ax.set_title('K-means clustering in reduced Pen-based\nusing PCA', fontsize=22)
plt.savefig(os.path.join('..', '..', 'figures', 'pen-based', 'pca-reduced-pen-based_k-means.png'))
plt.show()

file_path = os.path.join('..', '..', 'validation', 'pen-based_k-means_val.txt')
with open(file_path, 'a') as f:
    f.write(
        f'\n \nK-means: K = {K}, init = {init_method}, metric = {metric} max_inter = {n_iter}, n_init = {init}, reduction = PCA({df.shape[1]}), time = {finish_kmeans - start_kmeans}')

utils.print_metrics(df, df_gs, labels, file_path)
print("Finished K-means in reduced Pen-Based using PCA")

df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'pen-based_umap.pkl'))

print("Starting K-means in reduced Pen-Based with UMAP")
start_kmeans = time.time()
kmeans.fit(df)
finish_kmeans = time.time()
print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)

labels = kmeans.labels.astype(np.int32)
centers = np.array(kmeans.centroids)
colors = sns.color_palette()[:len(centers)]
for Class, colour in zip(range(len(centers)), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)
    ax.scatter(centers[Class, 0], centers[Class, 1], marker="x", color=colour, s=50)
    ax.set_xlabel(df.columns[0], fontsize=18)
    ax.set_ylabel(df.columns[1], fontsize=18)

ax.plot(df.iloc[labels == -1, 0],
        df.iloc[labels == -1, 1],
        'k+', alpha=0.1)

ax.set_title('K-means clustering in reduced Pen-Based\nusing UMAP', fontsize=22)
plt.savefig(os.path.join('..', '..', 'figures', 'pen-based', 'umap-reduced-pen-based_k-means.png'))
plt.show()

file_path = os.path.join('..', '..', 'validation', 'pen-based_k-means_val.txt')
with open(file_path, 'a') as f:
    f.write(
        f'\n \nK-means: K = {K}, init = {init_method}, metric = {metric} max_inter = {n_iter}, n_init = {init}, reduction=UMAP({df.shape[1]}), time = {start_kmeans - finish_kmeans}')

utils.print_metrics(df, df_gs, labels, file_path)
print("Finished K-means in reduced Pen-based with UMAP")
