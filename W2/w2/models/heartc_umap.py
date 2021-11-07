import os.path

import pandas as pd
import umap
from matplotlib import pyplot as plt
import seaborn as sns

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_heart-c.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_gs.pkl'))

reducer = umap.UMAP()
reducer.fit(df)
embedding = reducer.embedding_

print(embedding.shape)

sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})
plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in df_gs.map({'<50': 0, '>50_1': 1})])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Heart-C dataset', fontsize=24)
plt.show()
