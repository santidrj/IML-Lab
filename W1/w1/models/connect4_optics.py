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

start = time.time()
optics_clustering = OPTICS(min_samples=min_pts, metric='l2', algorithm='kd_tree').fit(df_connect4_encoded.iloc[:, :-1])
end = time.time()

path_save_model = os.path.join('..', '..', 'models_results', 'connect4', 'optics')
with open(path_save_model, 'wb') as file:
    pickle.dump(optics_clustering, file)

comp_time = end - start
print(f'OPTICS computation time: {comp_time / 60.} minutes')

with open(path_save_model, 'rb') as file:
    optics_clustering = pickle.load(file)

unique, counts = np.unique(optics_clustering.labels_, return_counts=True)
counts = counts / len(optics_clustering.labels_) * 100

path_val = os.path.join('..', '..', 'validation', 'connect4_val.txt')
with open(path_val, 'a') as f:
    f.write(
        f'\n \n*OPTICS: min_pts = {min_pts}, unique = {unique}, counts = {counts}  metric = L2, algorithm = kd_tree')


utils.print_metrics(df_connect4_encoded.iloc[:, :-1], df_connect4_encoded['class'],
                    optics_clustering.labels_,
                    file_path=path_val, isOPTICS=True)


metrics = ['l2', 'l1', 'chebyshev']
algorithms = ['kd_tree', 'brute']

start = time.time()

optics_clusterings = []
optics_clusterings.append(
    OPTICS(min_samples=min_pts, metric='l2', algorithm='kd_tree').fit(df_connect4_encoded_subset.iloc[:, :-1]))
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


for i, m in enumerate(metrics):
    for j, a in enumerate(algorithms):
        optics_model = optics_clusterings[i * (len(metrics) - 1) + j]
        unique, counts = np.unique(optics_model.labels_, return_counts=True)

        counts = counts / len(optics_model.labels_) * 100

        with open(path_val, 'a') as f:
            f.write(
                f'\n \n*OPTICS: min_pts = {min_pts}, unique = {unique}, counts = {counts}  metric = {m}, algorithm = {a}')

        utils.print_metrics(df_connect4_encoded_subset.iloc[:, :-1], df_connect4_encoded_subset['class'],
                            optics_model.labels_,
                            file_path=path_val, isOPTICS=True)

