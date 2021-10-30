import os

import pandas as pd
from matplotlib import pyplot as plt

from algorithms.fuzzycmeans import FuzzyCMeans, partition_coefficient, distance_measure

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_heart-c.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_gs.pkl'))

C = 2

fuzzy = FuzzyCMeans(C)
fuzzy.fit(df)

plt.figure(figsize=(12, 12))
ax = plt.subplot(111)

colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']
labels = fuzzy.u_matrix.argmax(axis=0)
for Class, colour in zip(range(9), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)

centers = fuzzy.centers
ax.scatter(centers[:, 0], centers[:, 1], marker="x", color="k")
ax.set_title(f'Fuzzy C-Means Clustering with C={C}')

plt.savefig(os.path.join('..', '..', 'figures', 'heart-c', f'heart-c_fuzzy-c-means-{C}.png'))
plt.show()

fig, axis = plt.subplots(C, 1, figsize=(12, 12))
for i, ax in enumerate(axis):
    ax.plot(fuzzy.u_matrix[i], colors[i])
axis[0].set_title('Membership functions')

plt.savefig(os.path.join('..', '..', 'figures', 'heart-c', f'heart-c_fuzzy-c-means-MF-{C}.png'))
plt.show()

print('Partition coefficient:', partition_coefficient(fuzzy.u_matrix, df.shape[0]))
