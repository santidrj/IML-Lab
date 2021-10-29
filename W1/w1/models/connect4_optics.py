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
df_connect4_encoded_subset = df_connect4_encoded.sample(n=3000)


min_pts = 10
start = time.time()
optics_clusters = OPTICS(min_samples = min_pts).fit(df_connect4_encoded_subset.iloc[:, :-1])
end = time.time()

with open('connect4_optics_clusters_30000_ms2', 'wb') as file:
    pickle.dump(optics_clusters, file)

comp_time = end-start
print(f'OPTICS computation time: {comp_time/60.} minutes')
with open('info.txt', 'w') as f:
    f.write('*OPTICS computation time: \n' + str(comp_time))

with open('connect4_optics_clusters_30000_ms2', 'rb') as file:
    optics_clusters = pickle.load(file)

def optics_plots(df, models):
    labels = models.labels_
    print(labels)
    label_set = set(labels)
    colors = cm.rainbow(np.linspace(0, 1, len(label_set)))
    reachability = models.reachability_[models.ordering_]

    for l, c in zip(label_set, colors):
        #plt.scatter(df['a3'][labels == l], df['a4'][labels == l], color=c)
        plt.plot(reachability[labels==l], color=c, marker='.', ls='')
    plt.show()

#optics_plots(df_connect4_encoded_subset, optics_clusters)
unique, counts = np.unique(optics_clusters.labels_, return_counts=True)

path = os.path.join('..', '..', 'validation', 'connect4_val.txt')
with open(path, 'w') as f:
    f.write(f'\n*OPTICS: min_pts = {min_pts}, unique = {unique}, counts = {counts}')

utils.print_metrics(df_connect4_encoded_subset.iloc[:, :-1], df_connect4_encoded_subset['class'], optics_clusters.labels_, file_path=path, isOPTICS=True )

"""

#clusters = np.load("clusters.npy")
test = np.load('test.npy')
print(test)
#print(clusters.value_coutns())
#plt.plot(df_connect4_encoded['a1'], )
"""