import os.path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

import w1.myfunctions as mf

root_path = os.path.join('.', 'datasets', 'datasets')

# %%
###########################
# HEART-C PRE-PROCESSING  #
###########################
# Load the Heart-C dataset
df_heart = mf.load_arff(os.path.join(root_path, 'heart-c.arff'))
df_heart.drop(columns='num', inplace=True)
mf.convert_byte_string_to_string(df_heart)
print()
print('Heart-C dataset:')
print(df_heart.head())
print(df_heart.dtypes)

# Get all the categorical features of the dataset for later processing
categorical_features = mf.get_categorical_features(df_heart)

for feature in categorical_features:
    print(f'{df_heart.value_counts(feature)}\n')

# Treat missing values
print('Total number of missing values:\n', df_heart.isnull().sum())
print()
print('ca possible values:\n', df_heart['ca'].value_counts())

df_heart['ca'] = df_heart['ca'].fillna(0.0)

print(df_heart.describe())

# Normalize data
df_heart_numerical = df_heart.drop(columns=categorical_features)
scaler = MinMaxScaler()
df_heart_normalized = pd.DataFrame(scaler.fit_transform(df_heart_numerical), columns=df_heart_numerical.columns)

# Transform categorical values to numerical
enc = OneHotEncoder()
enc.fit(df_heart[categorical_features])
print(enc.get_feature_names_out())
transformed_features = enc.transform(df_heart[categorical_features]).toarray()
df_heart_categorical = pd.DataFrame(transformed_features, columns=enc.get_feature_names_out())
df_heart_normalized = pd.concat([df_heart_normalized, df_heart_categorical], axis=1)
print()
print(df_heart_normalized.head())
