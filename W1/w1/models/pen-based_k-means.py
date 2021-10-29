import os

import numpy as np
from matplotlib import pyplot as plt

import utils
from algorithms.kmeans import Kmeans

K = 10

data_root_path = os.path.join('..', '..', 'datasets')
df = utils.load_arff(os.path.join(data_root_path, 'datasets', 'pen-based.arff'))

# TODO: get all rows
df_gs = df[df.columns[-1]]
df = df.drop(columns=df.columns[-1])

kmeans = Kmeans(k=K, init='random')
kmeans.fit(df)
print(f'Sum of squared error: {kmeans.square_error}')

plt.figure(figsize=(12, 12))
ax = plt.subplot(111)

colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']
labels = kmeans.labels
for Class, colour in zip(range(9), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)

ax.plot(df.iloc[labels == -1, 0],
        df.iloc[labels == -1, 1],
        'k+', alpha=0.1)

centers = np.array(kmeans.centroids)
ax.scatter(centers[:, 0], centers[:, 1], marker="x", color="k")
ax.set_title('K-means Clustering')

plt.show()

counts = np.bincount(labels.astype(int))
clusters = counts.argsort()
true_labels = df_gs.to_numpy(dtype='int32')
true_clusters = np.bincount(true_labels).argsort()

for i in range(len(true_labels)):
    true_labels[i] = clusters[np.where(true_clusters == true_labels[i])]

utils.print_metrics(df, true_labels, labels)
