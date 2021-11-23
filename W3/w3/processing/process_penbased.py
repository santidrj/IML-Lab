import os

import sys

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
