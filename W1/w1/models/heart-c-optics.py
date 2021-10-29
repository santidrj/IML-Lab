import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec
from sklearn.cluster import OPTICS

import utils

data_root_path = os.path.join('..', '..', 'datasets')
df_heart = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_heart-c.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_gs.pkl'))

# Use OPTICS with euclidean distance and auto algorithm for approximation to the nearest neighbour
metrics = ['euclidean', 'l1', 'chebyshev']
algorithms = ['kd_tree', 'brute']
minPts = 9

optics_clusterings = []
optics_clusterings.append(OPTICS(min_samples=minPts, metric='l2', algorithm='kd_tree').fit(df_heart))
optics_clusterings.append(OPTICS(min_samples=minPts, metric='l2', algorithm='brute').fit(df_heart))
optics_clusterings.append(OPTICS(min_samples=minPts, metric='l1', algorithm='kd_tree').fit(df_heart))
optics_clusterings.append(OPTICS(min_samples=minPts, metric='l1', algorithm='brute').fit(df_heart))
optics_clusterings.append(OPTICS(min_samples=minPts, metric='chebyshev', algorithm='kd_tree').fit(df_heart))
optics_clusterings.append(OPTICS(min_samples=minPts, metric='chebyshev', algorithm='brute').fit(df_heart))

for i, m in enumerate(metrics):
    for j, a in enumerate(algorithms):
        optics_model = optics_clusterings[i + j]
        # Creating a numpy array with numbers at equal spaces till
        # the specified range
        space = np.arange(len(df_heart))

        # Storing the reachability distance of each point
        reachability = optics_model.reachability_[optics_model.ordering_]

        # Storing the cluster labels of each point
        labels = optics_model.labels_[optics_model.ordering_]

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
        ax1.set_title('Reachability Plot')

        # Plotting the OPTICS Clustering
        colors = ['c.', 'b.', 'r.', 'y.', 'g.']
        for Class, colour in zip(range(5), colors):
            Xk = df_heart[optics_model.labels_ == Class]
            ax2.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3)

        ax2.plot(df_heart.iloc[optics_model.labels_ == -1, 0],
                 df_heart.iloc[optics_model.labels_ == -1, 1],
                 'k+', alpha=0.1)
        ax2.set_title(f'OPTICS Clustering with minPts = {minPts}, metric = {m} and algorithm = {a}')

        plt.tight_layout()
        plt.show()

        plt.savefig(os.path.join('..', '..', 'figures', 'heart-c', f'heart-c_optics-{m}-{a}.png'))

        utils.print_metrics(df_heart, df_gs, optics_model.labels_, isOPTICS=True)
