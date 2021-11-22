import os.path

import pandas as pd
import umap
from matplotlib import pyplot as plt
import seaborn as sns

N_NEIGHBORS = 30
N = 6
MIN_DIST = 0.0
RANDOM_STATE = 42

data_root_path = os.path.join('..', '..', 'datasets')
df = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_pen-based.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'pen-based_gs.pkl'))

reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, n_components=N, min_dist=MIN_DIST, random_state=RANDOM_STATE)
embedding = reducer.fit_transform(df)

save_path = os.path.join(data_root_path, 'processed')
pd.to_pickle(pd.DataFrame(embedding), os.path.join(save_path, 'pen-based_umap.pkl'))

figs_folder_path = os.path.join('..', '..', 'figures', 'pen-based')
sns.set(style='white', context='paper', rc={'figure.figsize': (14, 10)})
plt.scatter(embedding[:, 0], embedding[:, 1], c=[sns.color_palette()[x] for x in df_gs.astype(int)])
plt.gca().set_aspect('equal', 'datalim')
plt.title(f'UMAP projection of the Pen-Based dataset', fontsize=24)
plt.savefig(os.path.join(figs_folder_path, 'penbased_umap.png'))
plt.show()
