import os.path

import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import OneHotEncoder


def load_arff(path):
    data = arff.loadarff(path)
    return pd.DataFrame(data[0])


def convert_byte_string_to_string(dataframe):
    for col in dataframe:
        if isinstance(dataframe[col][0], bytes):
            print(col, "will be transformed from bytestring to string")
            dataframe[col] = dataframe[col].str.decode("utf8")  # or any other encoding


def get_categorical_features(dataframe):
    features = []
    for column in dataframe.columns:
        if dataframe[column].dtype.kind is 'O':
            features.append(column)
    return features


root_path = os.path.join('.', 'datasets', 'datasets')

# %%
df_heart = load_arff(os.path.join(root_path, 'heart-c.arff'))
convert_byte_string_to_string(df_heart)
print('Heart-C dataset:')
print(df_heart.head())
print(df_heart.dtypes)

categorical_features = get_categorical_features(df_heart)

for feature in categorical_features:
    print(df_heart.value_counts(feature))

print('Total number of missing values:\n', df_heart.isnull().sum())

print('ca possible values:\n', df_heart['ca'].value_counts())

df_heart['ca'] = df_heart['ca'].fillna(0.0)

print(df_heart.describe())

enc = OneHotEncoder()
enc.fit(df_heart[categorical_features])
print(enc.get_feature_names_out())
transfomed_features = enc.transform(df_heart[categorical_features]).toarray()
print(transfomed_features)
df_heart_categorical = pd.DataFrame(transfomed_features, columns=enc.get_feature_names_out())
df_heart_numerical = df_heart.drop(columns=categorical_features)
df_heart_numerical = pd.concat([df_heart_numerical, df_heart_categorical], axis=1)

# %%
#############################
# CONNECT-4 PRE-Processing  #
#############################
print('Connect-4 dataset:')
df_connect = load_arff(os.path.join(root_path, 'connect-4.arff'))
convert_byte_string_to_string(df_connect)
print(df_connect.head())
print(df_connect.dtypes)
print(df_connect.value_counts())

print('Total number of missing values:\n', df_connect.isnull().sum())

# %%
print('Hypothyroid dataset:')
df_hypothyroid = load_arff(os.path.join(root_path, 'hypothyroid.arff'))
convert_byte_string_to_string(df_hypothyroid)
print(df_hypothyroid.head())
print(df_hypothyroid.dtypes)
categorical_features = get_categorical_features(df_hypothyroid)

for feature in categorical_features:
    print(df_hypothyroid.value_counts(feature))

print('Total number of missing values:\n', df_hypothyroid.isnull().sum())
