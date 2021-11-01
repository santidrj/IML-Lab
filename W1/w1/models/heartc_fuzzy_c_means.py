import sys

sys.path.append("..")

import os

import pandas as pd
from matplotlib import pyplot as plt

import utils
from algorithms.fuzzycmeans import FuzzyCMeans, partition_coefficient

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_heart-c.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_gs.pkl'))

C = 2

print("Starting Fuzzy C-Means in Heart-C")
fuzzy = FuzzyCMeans(C)
fuzzy.fit(df)

plt.figure(figsize=(8, 8))
ax = plt.subplot(111)

colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']
labels = fuzzy.u_matrix.argmax(axis=0)
centers = fuzzy.centers
for Class, colour in zip(range(len(centers)), colors):
    Xk = df[labels == Class]
    ax.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], 'o', color=colour, alpha=0.3)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.scatter(centers[Class, 0], centers[Class, 1], marker="x", color=colour, s=40)

ax.set_title(f'Fuzzy C-Means Clustering with C={C}')

plt.savefig(os.path.join('..', '..', 'figures', 'heart-c', f'heart-c_fuzzy-c-means-{C}.png'))
plt.show()

fig, axis = plt.subplots(C, 1, figsize=(12, 12))
for i, ax in enumerate(axis):
    ax.plot(fuzzy.u_matrix[i], colors[i])
    ax.set_ylabel(f'C{i}')
axis[0].set_title('Membership functions')

plt.savefig(os.path.join('..', '..', 'figures', 'heart-c', f'heart-c_fuzzy-c-means-MF-{C}.png'))
plt.show()

file_path = os.path.join('..', '..', 'validation', 'heart-c-fuzzy_val.txt')
with open(file_path, 'a') as f:
    f.write(f'\n\nFuzzy C-Means: C={C}')
    f.write(f'\nPartition coefficient: {partition_coefficient(fuzzy.u_matrix, df.shape[0])}\n')

utils.print_metrics(df, df_gs, labels, file_path)
print("Finished Fuzzy C-Means in Heart-C")
