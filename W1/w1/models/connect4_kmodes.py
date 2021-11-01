import sys

sys.path.append("..")

import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import utils
from algorithms import kmodes

df_connect4 = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'processed_connect4.pkl'))
df_connect4_encoded = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'encoded_connect4.pkl'))

print("Starting K-Modes in Connect-4")
k_modes_clusterings = []

k_modes_clusterings.append(kmodes.KModes(df_connect4.iloc[:, :-1], k=2, max_iter=10).run('bisecting'))
k_modes_clusterings.append(kmodes.KModes(df_connect4.iloc[:, :-1], k=3, max_iter=10).run('bisecting'))
k_modes_clusterings.append(kmodes.KModes(df_connect4.iloc[:, :-1], k=4, max_iter=10).run('bisecting'))
k_modes_clusterings.append(kmodes.KModes(df_connect4.iloc[:, :-1], k=5, max_iter=10).run('bisecting'))

path_save_model = os.path.join('..', '..', 'models_results', 'connect4', 'k-modes')
with open(path_save_model, 'wb') as file:
    pickle.dump(k_modes_clusterings, file)

with open(path_save_model, 'rb') as file:
    k_modes_clusterings = pickle.load(file)

path_val = os.path.join('..', '..', 'validation', 'connect4_val.txt')
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['axes.labelpad'] = 10
plt.rcParams['legend.fontsize'] = 18

colors = ['c', 'b', 'r', 'y', 'g']
plt.figure(figsize=(9, 5))
plt.grid(axis='y')
plt.xlabel('Clusters')
plt.ylabel('Samples [% over the total]')

for i, pred_labels in enumerate(k_modes_clusterings):
    unique, counts = np.unique(pred_labels, return_counts=True)
    counts = counts / len(pred_labels) * 100
    plt.bar(unique + i * 0.2, counts, width=0.2, align='center', alpha=0.9, color=colors[i])

    with open(path_val, 'a') as f:
        f.write(f'\n \n*K-MODES: k = {i + 2}')

    utils.print_metrics(df_connect4_encoded.iloc[:, :-1], df_connect4['class'], pred_labels,
                        file_path=path_val, isOPTICS=False)

plt.xticks(np.arange(5) + 0.3, ['0', '1', '2', '3', '4'])
plt.tight_layout()
plt.legend(['k = 2', 'k = 3', 'k = 4', 'k = 5'])
plt.savefig(os.path.join('..', '..', 'figures', 'connect4', 'kmodes_barplot'))
plt.show()
print("Finished K-Modes in Connect-4")
