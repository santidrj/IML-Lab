import os

import sys

import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("..")

import pandas as pd

import utils

data_root_path = os.path.join('..', '..', 'datasets')
df = utils.load_arff(os.path.join(data_root_path, 'datasets', 'pen-based.arff'))

# Save last column for later use in validation steps
utils.convert_byte_string_to_string(df)
df_gs = df['a17']
df = df.drop(columns='a17')

print("total number of missing values: ", df.isnull().sum().sum())

# visualize the characteristic of each column using describe
print(df.describe())

save_path = os.path.join(data_root_path, 'processed')
pd.to_pickle(df, os.path.join(save_path, 'processed_pen-based.pkl'))
pd.to_pickle(df_gs, os.path.join(save_path, 'pen-based_gs.pkl'))

figs_folder_path = os.path.join('..', '..', 'figures', 'pen-based')
sns.set(style='white', context='paper', rc={'figure.figsize': (14, 10)})
ax = plt.subplot()
ax.scatter(df.iloc[:, 0], df.iloc[:, 1],
           c=[sns.color_palette()[x] for x in df_gs.astype(int)])
plt.gca().set_aspect('equal', 'datalim')
ax.set_title('Pen-Based dataset', fontsize=24)
plt.savefig(os.path.join(figs_folder_path, 'original_penbased.png'))
