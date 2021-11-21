import os
import sys

sys.path.append("..")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler

import utils

data_root_path = os.path.join('..', '..', 'datasets')

# Load the Heart-C dataset
df_heart = utils.load_arff(os.path.join(data_root_path, 'datasets', 'heart-c.arff'))
utils.convert_byte_string_to_string(df_heart)
df_gs = df_heart['num']
df_heart.drop(columns='num', inplace=True)
print()
print('Heart-C dataset:')
print(df_heart.head())
print(df_heart.dtypes)

# Get all the categorical features of the dataset for later processing
categorical_features = df_heart.select_dtypes(include='object').columns

for feature in categorical_features:
    print(f'{df_heart.value_counts(feature)}\n')

# Treat missing values
print('Total number of missing values per feature:\n', df_heart.isnull().sum())
print()
print('ca possible values:\n', df_heart['ca'].value_counts())

df_heart['ca'] = df_heart['ca'].fillna(0.0)

print(df_heart.describe())

# Transform categorical values to numerical
df_heart_categorical = utils.categorical_to_numerical(df_heart)

# Normalize data
df_heart_numerical = df_heart.select_dtypes(include='number')
df_heart_normalized = utils.normalize_data(df_heart_numerical, df_heart_numerical.columns, RobustScaler())
df_heart_normalized = pd.concat([df_heart_normalized, df_heart_categorical.drop(columns=df_heart_normalized.columns)],
                                axis=1)
print()
print(df_heart_normalized.head())

save_path = os.path.join(data_root_path, 'processed')
pd.to_pickle(df_heart_normalized, os.path.join(save_path, 'processed_heart-c.pkl'))
pd.to_pickle(df_gs, os.path.join(save_path, 'heart-c_gs.pkl'))

figs_folder_path = os.path.join('..', '..', 'figures', 'heart-c')
sns.set(style='white', context='paper', rc={'figure.figsize': (14, 10)})
ax = plt.subplot()
ax.scatter(df_heart_normalized.iloc[:, 0], df_heart_normalized.iloc[:, 1],
           c=[sns.color_palette()[x] for x in df_gs.map({'<50': 0, '>50_1': 1})])
plt.gca().set_aspect('equal', 'datalim')
ax.set_title('Heart-C dataset', fontsize=24)
plt.savefig(os.path.join(figs_folder_path, 'original_heartc.png'))
plt.show()
