import os

import numpy as np
import pandas as pd

import utils
from algorithms import kmodes

df_connect4 = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'processed_connect4.pkl'))
df_connect4_encoded = pd.read_pickle(os.path.join('..', '..', 'datasets', 'processed', 'encoded_connect4.pkl'))

classes = kmodes.KModes(df_connect4.iloc[:, :-1], k=4, max_iter=10).run('bisecting')
print(classes.value_counts())
true_class = df_connect4['class']
print(true_class.value_counts())
np.save('predicted_classes_k_modes.npy', classes)

classes = np.load('../predicted_classes_k_modes.npy')
utils.print_metrics(df_connect4_encoded.iloc[:, :-1], true_class, classes)
