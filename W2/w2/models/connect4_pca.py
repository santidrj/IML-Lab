import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA


data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'encoded_connect4.pkl'))
df = df.drop(columns=df.columns[-1])

pca_model = PCA().fit(df.T)
ipca_model = IncrementalPCA().fit(df.T)

"""
ax = plt.subplot(projection='3d')
ax.scatter(pca_model.components_[0], pca_model.components_[1], pca_model.components_[2], s=0.008, alpha=0.9)
plt.show()
"""
ax = plt.subplot()
ax.scatter(ipca_model.components_[0], ipca_model.components_[1], alpha=0.9, s=0.008)
plt.show()

ax = plt.subplot()
ax.scatter(pca_model.components_[0], pca_model.components_[1], alpha=0.9, s=0.008)
plt.show()

features = df[['a6', 'a7']].to_numpy()
unique, counts = np.unique(features, axis=0, return_counts=True)
fig, ax = plt.subplots()
ax.scatter(df['a6'], df['a7'])
ax.set_xticks([0,1,2])
ax.set_yticks([0,1,2])
ax.set_ylim(0-0.3, 2+0.3)
ax.set_xlim(0-0.3, 2+0.3)
for i in range(unique.shape[0]):
    ax.text(unique[i][0]+0.05, unique[i][1]+0.05, counts[i])
plt.show()


