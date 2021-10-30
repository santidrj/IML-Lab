import os
import pickle
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec
from sklearn.cluster import OPTICS

import utils

## OPTICS
df_connect4 = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'processed_connect4.pkl'))
df_connect4_encoded = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'encoded_connect4.pkl'))
df_connect4_encoded_subset = pd.read_pickle(
    os.path.join('..', '..', 'datasets', 'processed', 'encoded_subset_connect4.pkl'))

min_pts= 50

metrics = ['l2', 'l1', 'chebyshev']
algorithms = ['kd_tree', 'brute']

"""
start = time.time()
optics_clusterings = []
optics_clusterings.append(
    OPTICS(min_samples=min_pts, metric='l2', algorithm='kd_tree').fit(df_connect4_encoded.iloc[:, :-1]))

optics_clusterings.append(
    OPTICS(min_samples=min_pts, metric='l2', algorithm='brute').fit(df_connect4_encoded_subset.iloc[:, :-1]))
optics_clusterings.append(
    OPTICS(min_samples=min_pts, metric='l1', algorithm='kd_tree').fit(df_connect4_encoded_subset.iloc[:, :-1]))
optics_clusterings.append(
    OPTICS(min_samples=min_pts, metric='l1', algorithm='brute').fit(df_connect4_encoded_subset.iloc[:, :-1]))
optics_clusterings.append(
    OPTICS(min_samples=min_pts, metric='chebyshev', algorithm='kd_tree').fit(df_connect4_encoded_subset.iloc[:, :-1]))
optics_clusterings.append(
    OPTICS(min_samples=min_pts, metric='chebyshev', algorithm='brute').fit(df_connect4_encoded_subset.iloc[:, :-1]))

end = time.time()

with open('connect4_optics', 'wb') as file:
    pickle.dump(optics_clusterings, file)

comp_time = end - start
print(f'OPTICS computation time: {comp_time / 60.} minutes')
with open('info.txt', 'w') as f:
    f.write('*OPTICS computation time: \n' + str(comp_time))
"""
with open('connect4_optics_metrics', 'rb') as file:
    optics_clusterings = pickle.load(file)


def optics_plots(df, models):
    labels = models.labels_
    print(labels)
    label_set = set(labels)
    colors = colors = ['c.', 'b.', 'r.', 'y.', 'g.']
    reachability = models.reachability_[models.ordering_]
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 10))
    g = gridspec.GridSpec(2, 1)
    ax1 = plt.subplot(g[0, :])
    ax2 = plt.subplot(g[1, :], projection='3d')

    for l, c in zip(label_set, colors):
        if l == -1:
            ax1.plot(reachability[labels == l], color='k', marker='.', ls='', alpha=0.3)
            x, y, z = df['a3'][labels == l], df['a4'][labels == l], df['a5'][labels == l]
            ax2.scatter(x, y, z, color='k', alpha=0.3)
            ax2.text(x * (1 + 0.01), y * (1 + 0.01), z * (1 + 0.01), counts[unique == l], fontsize=12)
        else:
            ax1.plot(reachability[labels == l], color=c, marker='.', ls='', alpha=0.3)
            x, y, z = df['a3'][labels == l], df['a4'][labels == l], df['a5'][labels == l]
            ax2.scatter(x, y, z, color=c, alpha=0.3)
            ax2.text(x * (1 + 0.01), y * (1 + 0.01), z * (1 + 0.01), counts[unique == l], fontsize=12)

    plt.show()


# optics_plots(df_connect4_encoded, optics_clusterings)

path_val = os.path.join('..', '..', 'validation', 'connect4_val.txt')
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['legend.fontsize'] = 15

colors = ['c', 'b', 'r', 'y', 'g', 'm']
plt.figure(figsize=(9, 5))
plt.grid(axis='y')
plt.xlabel('Clusters')
plt.ylabel('Samples [% over the total]')

for i, m in enumerate(metrics):
    for j, a in enumerate(algorithms):
        optics_model = optics_clusterings[i + j]
        unique, counts = np.unique(optics_model.labels_, return_counts=True)

        counts = counts / len(optics_model.labels_) * 100
        plt.bar(unique + (2*i+j) * 0.15, counts, width=0.15, align='center', alpha=0.9, color=colors[2*i+j])

        with open(path_val, 'a') as f:
            f.write(
                f'\n \n*OPTICS: min_pts = {min_pts}, unique = {unique}, counts = {counts}  metric = {m}, algorithm = {a}')

        utils.print_metrics(df_connect4_encoded_subset.iloc[:, :-1], df_connect4_encoded_subset['class'],
                            optics_model.labels_,
                            file_path=path_val, isOPTICS=True)


plt.xticks(np.arange(4)-1+0.37, ['-1', '1', '2', '3'])
plt.tight_layout()
plt.legend(['L2, kd_tree', 'L2, brute', 'L1, kd_tree', 'L1, brute', 'Chebyshev, kd_tree', 'Chebyshev, brute'])
plt.savefig(os.path.join('..', '..', 'figures', 'connect4', 'optics_barplot'))
plt.show()



"""
unique, counts = np.unique(optics_clusters.labels_, return_counts=True)
print(unique, counts)

path = os.path.join('..', '..', 'validation', 'connect4_val.txt')
with open(path, 'a') as f:
    f.write(f'\n \n*OPTICS: min_pts = {min_pts}, max_eps = {max_eps}, cluster_method= {cluster_mth}, unique = {unique}, counts = {counts}')

utils.print_metrics(df_connect4_encoded.iloc[:, :-1], df_connect4_encoded['class'], optics_clusters.labels_, file_path=path, isOPTICS=True )
"""
