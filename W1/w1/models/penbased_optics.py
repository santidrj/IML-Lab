import sys

sys.path.append("..")

import os
from sklearn.cluster import OPTICS
from matplotlib import pyplot as plt, gridspec
import pandas as pd
import numpy as np

import utils

metrics = ['l2', 'l1', 'chebyshev']
algorithms = ['kd_tree', 'brute']
minPts = 50

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_pen-based.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'pen-based_gs.pkl'))

print("Starting OPTICS in Pen-Based")
optics_clusterings = []
optics_clusterings.append(OPTICS(min_samples=minPts, metric='l2', algorithm='kd_tree').fit(df))
optics_clusterings.append(OPTICS(min_samples=minPts, metric='l2', algorithm='brute').fit(df))
optics_clusterings.append(OPTICS(min_samples=minPts, metric='l1', algorithm='kd_tree').fit(df))
optics_clusterings.append(OPTICS(min_samples=minPts, metric='l1', algorithm='brute').fit(df))
optics_clusterings.append(OPTICS(min_samples=minPts, metric='chebyshev', algorithm='kd_tree').fit(df))
optics_clusterings.append(OPTICS(min_samples=minPts, metric='chebyshev', algorithm='brute').fit(df))

for i, m in enumerate(metrics):
    for j, a in enumerate(algorithms):
        optics_model = optics_clusterings[i * (len(metrics) - 1) + j]
        # Creating a numpy array with numbers at equal spaces till
        # the specified range
        space = np.arange(len(df))

        # Storing the reachability distance of each point
        reachability = optics_model.reachability_[optics_model.ordering_]

        # Storing the cluster labels of each point
        labels = optics_model.labels_[optics_model.ordering_]
        no_clusters = len(np.unique(optics_model.labels_))
        no_noise = np.sum(np.array(optics_model.labels_) == -1, axis=0)
        # Defining the framework of the visualization
        plt.figure(figsize=(10, 10))
        G = gridspec.GridSpec(2, 1)
        ax1 = plt.subplot(G[0, :])
        ax2 = plt.subplot(G[1, :])

        # Plotting the Reachability-Distance Plot
        colors = ['c.', 'b.', 'r.', 'y.', 'g.']
        for Class, colour in zip(range(0, 5), colors):
            Xk = space[labels == Class]
            Rk = reachability[labels == Class]
            ax1.plot(Xk, Rk, colour, alpha=0.3)
        ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
        ax1.set_ylabel('Reachability Distance')
        ax1.set_title('Reachability Plot', fontsize=14)

        # Plotting the OPTICS Clustering
        colors = ['c.', 'b.', 'r.', 'y.', 'g.']
        for Class, colour in zip(range(5), colors):
            Xk = df[optics_model.labels_ == Class]
            ax2.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3)

        ax2.plot(df.iloc[optics_model.labels_ == -1, 0],
                 df.iloc[optics_model.labels_ == -1, 1],
                 'k+', alpha=0.1)
        ax2.set_title(f'OPTICS Clustering\nwith minPts = {minPts}, metric = {m} and algorithm = {a}', fontsize=14)
        print('Estimated no. of clusters: %d' % no_clusters)
        print('Estimated no. of noise points: %d' % no_noise)
        plt.tight_layout()
        plt.savefig(os.path.join('..', '..', 'figures', 'pen-based', f'penbased_optics-{m}-{a}.png'))
        plt.show()

        file_path = os.path.join('..', '..', 'validation', 'penbased_optics_val.txt')
        with open(file_path, 'a') as f:
            f.write(f'\n \nOPTICS: min_pts = {minPts}, metric = {m}, algorithm = {a}')

        utils.print_metrics(df, df_gs, optics_model.labels_, file_path, isOPTICS=True)

print("Finished OPTICS in Pen-Based")
