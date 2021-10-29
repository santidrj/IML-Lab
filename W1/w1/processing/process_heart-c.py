import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler

import utils

data_root_path = os.path.join('..', '..', 'datasets')

# %%
###########################
# HEART-C PRE-PROCESSING  #
###########################
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
print('Total number of missing values:\n', df_heart.isnull().sum())
print()
print('ca possible values:\n', df_heart['ca'].value_counts())

df_heart['ca'] = df_heart['ca'].fillna(0.0)

print(df_heart.describe())

# Transform categorical values to numerical
df_heart_categorical = utils.categorical_to_numerical(df_heart)

# Normalize data
df_heart_numerical = df_heart.select_dtypes(include='number')
# df_heart_normalized = utils.normalize_data(df_heart_numerical, df_heart_numerical.columns, RobustScaler())
df_heart_normalized = utils.normalize_data(df_heart_categorical, df_heart_categorical.columns, RobustScaler())
# df_heart_normalized = pd.concat([df_heart_normalized, df_heart_categorical.drop(columns=df_heart_normalized.columns)],
#                                 axis=1)
print()
print(df_heart_normalized.head())

save_path = os.path.join(data_root_path, 'processed')
pd.to_pickle(df_heart_normalized, os.path.join(save_path, 'processed_heart-c.pkl'))
pd.to_pickle(df_gs, os.path.join(save_path, 'heart-c_gs.pkl'))

figs_folder_path = os.path.join('..', '..', 'figures', 'heart-c')
for col in df_heart_numerical.columns:
    plt.clf()
    sns.boxplot(x=df_heart_numerical[col]).set_title(f'{col} data distribution')
    sns.stripplot(x=df_heart_numerical[col],
                  jitter=True,
                  marker='o',
                  alpha=0.5,
                  color='black')
    plt.savefig(os.path.join(figs_folder_path, f'heart-c_{col}_boxplot.png'))

