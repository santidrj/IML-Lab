import os.path

import pandas as pd
import umap
from matplotlib import pyplot as plt
import seaborn as sns

N_NEIGHBORS = 40
N = 7
MIN_DIST = 0.0
RANDOM_STATE = 42

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_heart-c.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_gs.pkl'))

reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, n_components=N, min_dist=MIN_DIST, random_state=RANDOM_STATE)
embedding = reducer.fit_transform(df)

save_path = os.path.join(data_root_path, 'processed')
pd.to_pickle(pd.DataFrame(embedding), os.path.join(save_path, 'heart-c_umap.pkl'))

figs_folder_path = os.path.join('..', '..', 'figures', 'heart-c')
sns.set(style='white', context='paper', rc={'figure.figsize': (14, 10)})
plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in df_gs.map({'<50': 0, '>50_1': 1})])
plt.gca().set_aspect('equal', 'datalim')
plt.title(f'UMAP projection of the Heart-C dataset', fontsize=24)
plt.savefig(os.path.join(figs_folder_path, 'heartc_umap.png'))
plt.show()
