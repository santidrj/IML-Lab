import time
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import utils

## OPTICS
df_connect4 = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'processed_connect4.pkl'))
df_connect4_encoded = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'encoded_connect4.pkl'))
#df_connect4_encoded_subset = df_connect4_encoded.sample(n=3000)

min_pts = 180

start = time.time()
optics_clusters = OPTICS(min_samples = min_pts).fit(df_connect4_encoded.iloc[:, :-1])
end = time.time()
"""
metrics = ['euclidean', 'l1', 'chebyshev']
algorithms = ['kd_tree', 'brute']

optics_clusterings = []
optics_clusterings.append(OPTICS(min_samples=min_pts, metric='l2', algorithm='kd_tree').fit(df_connect4_encoded.iloc[:, :-1]))
optics_clusterings.append(OPTICS(min_samples=min_pts, metric='l2', algorithm='brute').fit(df_connect4_encoded.iloc[:, :-1]))
optics_clusterings.append(OPTICS(min_samples=min_pts, metric='l1', algorithm='kd_tree').fit(df_connect4_encoded.iloc[:, :-1]))
optics_clusterings.append(OPTICS(min_samples=min_pts, metric='l1', algorithm='brute').fit(df_connect4_encoded.iloc[:, :-1]))
optics_clusterings.append(OPTICS(min_samples=min_pts, metric='chebyshev', algorithm='kd_tree').fit(df_connect4_encoded.iloc[:, :-1]))
optics_clusterings.append(OPTICS(min_samples=min_pts, metric='chebyshev', algorithm='brute').fit(df_connect4_encoded.iloc[:, :-1]))
"""
with open('connect4_optics_metrics', 'wb') as file:
    pickle.dump(optics_clusters, file)

#with open('connect4_optics_metrics', 'wb') as file:
#    pickle.dump(optics_clusterings, file)

comp_time = end-start
print(f'OPTICS computation time: {comp_time/60.} minutes')
with open('info.txt', 'w') as f:
    f.write('*OPTICS computation time: \n' + str(comp_time))

with open('connect4_optics_metrics', 'rb') as file:
    optics_clusters = pickle.load(file)

#with open('connect4_optics_metrics', 'rb') as file:
#    optics_clusterings = pickle.load(file)



"""
def optics_plots(df, models):
    labels = models.labels_
    print(labels)
    label_set = set(labels)
    colors = cm.rainbow(np.linspace(0, 1, len(label_set)))
    reachability = models.reachability_[models.ordering_]

    for l, c in zip(label_set, colors):
        #plt.scatter(df['a3'][labels == l], df['a4'][labels == l], color=c)
        if l == -1:
            plt.plot(reachability[labels == l], color='k', marker='.', ls='')
        else:
            plt.plot(reachability[labels == l], color=c, marker='.', ls='')
    plt.show()
"""
path_val = os.path.join('..', '..', 'validation', 'connect4_val.txt')
for i, m in enumerate(metrics):
    for j, a in enumerate(algorithms):
        optics_model = optics_clusterings[i + j]

        with open(path_val, 'a') as f:
            f.write(f'\n \n*OPTICS: min_pts = {min_pts}, metric = {m}, algorithm = {a}')

        utils.print_metrics(df_connect4_encoded.iloc[:, :-1], df_connect4_encoded['class'], optics_model.labels_, file_path=path, isOPTICS=True)





unique, counts = np.unique(optics_clusters.labels_, return_counts=True)
print(unique, counts)

path = os.path.join('..', '..', 'validation', 'connect4_val.txt')
with open(path, 'a') as f:
    f.write(f'\n \n*OPTICS: min_pts = {min_pts}, unique = {unique}, counts = {counts}')

utils.print_metrics(df_connect4_encoded.iloc[:, :-1], df_connect4_encoded['class'], optics_clusters.labels_, file_path=path, isOPTICS=True )
