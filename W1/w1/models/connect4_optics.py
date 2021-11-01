import os
import pickle
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import OPTICS

import utils

df_connect4_encoded = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'encoded_connect4.pkl'))
df_connect4_encoded_subset = pd.read_pickle(
    os.path.join('..', '..', 'datasets', 'processed', 'encoded_subset_connect4.pkl'))

min_pts= 50

metrics = ['l2', 'l1', 'chebyshev']
algorithms = ['kd_tree', 'brute']

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

path_save_model = os.path.join('..', '..', 'models_results', 'connect4', 'optics_metrics')
with open(path_save_model, 'wb') as file:
    pickle.dump(optics_clusterings, file)

comp_time = end - start
print(f'OPTICS computation time: {comp_time / 60.} minutes')

with open(path_save_model, 'rb') as file:
    optics_clusterings = pickle.load(file)

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
        optics_model = optics_clusterings[i * (len(metrics) - 1) + j]
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