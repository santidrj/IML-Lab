import os

import pandas as pd
from matplotlib import pyplot as plt

from algorithms.fuzzycmeans import FuzzyCMeans

data_root_path = os.path.join('..', '..', 'datasets')
df_heart = pd.read_pickle(os.path.join(data_root_path, 'processed', 'processed_heart-c.pkl'))
df_gs = pd.read_pickle(os.path.join(data_root_path, 'processed', 'heart-c_gs.pkl'))

fuzzy = FuzzyCMeans(2)
fuzzy.fit(df_heart)

fig, ax = plt.subplots(2, 1)
ax[0].plot(fuzzy.u_matrix[0])
ax[1].plot(fuzzy.u_matrix[1])
plt.show()
