import time
import os

import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt

## OPTICS
df_connect4 = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'processed_connect4.pkl'))
df_connect4_encoded = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'encoded_connect4.pkl'))

np.save('test.npy', np.array([0,1,2,3]))
"""
start = time.time()
clusters = OPTICS(min_samples = 2).fit_predict(df_connect4_encoded.iloc[:, :-1])
end = time.time()

np.save('connect4_optics_clusters.npy', clusters)

comp_time = end-start
print(f'OPTICS computation time: {comp_time/60.} minutes')
with open('info.txt', 'w') as f:
    f.write('*OPTICS computation time: \n' + str(comp_time))


"""
#clusters = np.load("clusters.npy")
test = np.load('test.npy')
print(test)
#print(clusters.value_coutns())
#plt.plot(df_connect4_encoded['a1'], )