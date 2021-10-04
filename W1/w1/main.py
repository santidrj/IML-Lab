import os.path

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

from myfunctions import *

root_path = os.path.join('.', 'datasets', 'datasets')

# %%
# Load the Heart-C dataset
df_heart = load_arff(os.path.join(root_path, 'heart-c.arff'))
convert_byte_string_to_string(df_heart)
print()
print('Heart-C dataset:')
print(df_heart.head())
print(df_heart.dtypes)

# Get all the categorical features of the dataset for later processing
categorical_features = get_categorical_features(df_heart)

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