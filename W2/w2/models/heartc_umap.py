import os.path

import pandas as pd
import umap
from matplotlib import pyplot as plt
import seaborn as sns

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_heart-c.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_gs.pkl'))

reducer = umap.UMAP(n_components=3)
reducer.fit(df)
embedding = reducer.embedding_

print(embedding.shape)

# sns.set(style='white', context='paper', rc={'figure.figsize': (14, 10)})
# plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in df_gs.map({'<50': 0, '>50_1': 1})])
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the Heart-C dataset', fontsize=24)
# plt.show()

ax = plt.subplot(projection='3d')
ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
           c=[sns.color_palette()[x] for x in df_gs.map({'<50': 0, '>50_1': 1})], s=0.1, alpha=0.9)
plt.show()
