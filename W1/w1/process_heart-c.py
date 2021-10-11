import os.path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler

import utils

root_path = os.path.join('..', 'datasets', 'datasets')

# %%
###########################
# HEART-C PRE-PROCESSING  #
###########################
# Load the Heart-C dataset
df_heart = utils.load_arff(os.path.join(root_path, 'heart-c.arff'))
df_heart.drop(columns='num', inplace=True)
utils.convert_byte_string_to_string(df_heart)
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

# Normalize data
df_heart_numerical = df_heart.select_dtypes(include='number')
df_heart_normalized = utils.normalize_data(df_heart_numerical, df_heart_numerical.columns, RobustScaler())

# Transform categorical values to numerical
df_heart_categorical = utils.categorical_to_numerical(df_heart)
df_heart_normalized = pd.concat([df_heart_normalized, df_heart_categorical], axis=1)
print()
print(df_heart_normalized.head())

for col in df_heart_numerical.columns:
    sns.boxplot(x=df_heart_numerical[col])
    sns.stripplot(x=df_heart_numerical[col],
                  jitter=True,
                  marker='o',
                  alpha=0.5,
                  color='black')
    plt.show()
