import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils
from algorithms.kmeans import Kmeans

K = 2

data_root_path = os.path.join('..', 'datasets')
df_heart = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed-heart-c.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_gs.pkl'))

kmeans = Kmeans(k=K, init='k-means++')
kmeans.fit(df_heart.iloc[:, :2])
print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(12, 12))
ax = plt.subplot(111)

colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen']
labels = kmeans.labels
for Class, colour in zip(range(9), colors):
    Xk = df_heart[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)

ax.plot(df_heart.iloc[labels == -1, 0],
        df_heart.iloc[labels == -1, 1],
        'k+', alpha=0.1)

centers = np.array(kmeans.centroids)
ax.scatter(centers[:, 0], centers[:, 1], marker="x", color="k")
ax.set_title('K-means Clustering')

plt.show()

counts = np.bincount(labels.astype(int))
maj_class = counts.argmax()
min_class = counts.argmin()
df_gs.replace({'<50': maj_class, '>50_1': min_class}, inplace=True)

utils.run_numerical_validations(df_heart, df_gs, labels, K)
