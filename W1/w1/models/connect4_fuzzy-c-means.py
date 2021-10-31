import os
import pickle

import pandas as pd

from algorithms.fuzzycmeans import FuzzyCMeans

df_connect4_encoded = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'encoded_connect4.pkl'))

df = df_connect4_encoded.iloc[:, :-1]

fuzzy_clusterings = []
for C in range(4):
    for i in range(10):
        fuzzy_clusterings.append(FuzzyCMeans(C + 2, max_iter=15).fit(df))
        print(f'finished {i}')

with open('connect4_fuzzy-c-means', 'wb') as file:
    pickle.dump(fuzzy_clusterings, file)

with open('connect4_fuzzy-c-means', 'rb') as file:
    fuzzy_clusterings = pickle.load(file)

"""
colors = ['c', 'b', 'r', 'y', 'g', 'm', 'maroon', 'crimson', 'darkgreen', 'peru']
fig, axis = plt.subplots(C, 1, figsize=(12, 12))
for i, ax in enumerate(axis):
    ax.plot(fuzzy.u_matrix[i], colors[i])
axis[0].set_title('Membership functions')

plt.savefig(os.path.join('..', '..', 'figures', 'connect4', f'MF-{C}.png'))
plt.show()

#print('Partition coefficient:', partition_coefficient(fuzzy.u_matrix, df.shape[0]))
"""