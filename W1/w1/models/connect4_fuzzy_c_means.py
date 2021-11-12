import os
import sys

sys.path.append("..")
import pickle

import pandas as pd
import numpy as np

from algorithms.fuzzycmeans import FuzzyCMeans, partition_coefficient
import matplotlib.pyplot as plt

import utils

df_connect4_encoded = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'connect4_encoded.pkl'))

df = df_connect4_encoded.iloc[:, :-1]

print("Starting Fuzzy C-Means in Connect-4")
fuzzy_clusterings = []
for C in range(4):
    for i in range(10):
        fuzzy_clusterings.append(FuzzyCMeans(C + 2, max_iter=15).fit(df))
        print(f'finished {i}')

path_save_model = os.path.join('..', '..', 'models_results', 'connect4', 'fuzzy-c-means')
with open(path_save_model, 'wb') as file:
    pickle.dump(fuzzy_clusterings, file)

with open(path_save_model, 'rb') as file:
    fuzzy_clusterings = pickle.load(file)

n = len(df)

path_val = os.path.join('..', '..', 'validation', 'connect4_val.txt')
for C in range(4):
    for i in range(10):
        model = fuzzy_clusterings[(C*10) + i]
        u_matrix = model[0]
        part_coeff = partition_coefficient(u_matrix, n)

        crisp_labels = []
        for c in range(n):
            crisp_labels.append(np.argmax(u_matrix[:, c]))

        with open(path_val, 'a') as f:
            f.write(f'\n \n*FCM: C = {C + 2}, it = {i} \n Partition coefficient: {part_coeff}')

        utils.print_metrics(df_connect4_encoded.iloc[:, :-1], df_connect4_encoded['class'], crisp_labels,
                            file_path=path_val, isOPTICS=False)


best_iter = [7, 6, 5, 6]
for C in range(4):
    model = fuzzy_clusterings[(C * 10) + best_iter[C]]
    u_matrix = model[0][:, 0::200]
    colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']
    fig, axis = plt.subplots(C+2, 1, figsize=(12, 12))

    for i, ax in enumerate(axis):
        ax.plot(u_matrix[i], colors[i], lw=1.5)
        ax.set_ylabel(f'Cluster {i}', fontsize=15)
        ax.set_xticks(np.arange(0,400,50))
        ax.set_xticklabels(['0', '10000', '20000', '30000', '40000', '50000', '60000', '70000'])
    axis[0].set_title('Membership functions', fontsize=20)
    axis[-1].set_xlabel('Samples', fontsize=15)

    plt.savefig(os.path.join('..', '..', 'figures', 'connect4', f'MF-{C+2}.png'))
    plt.show()

print("Finished Fuzzy C-Means in Connect-4")
