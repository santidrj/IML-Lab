import os.path

import utils

root_path = os.path.join('.', 'datasets', 'datasets')

# %%
##############################
# HYPOTHYROID PRE-PROCESSING #
##############################
print('Hypothyroid dataset:')
df_hypothyroid = utils.load_arff(os.path.join(root_path, 'hypothyroid.arff'))
utils.convert_byte_string_to_string(df_hypothyroid)
print(df_hypothyroid.head())
print(df_hypothyroid.dtypes)
categorical_features = utils.get_categorical_features(df_hypothyroid)

for feature in categorical_features:
    print(df_hypothyroid.value_counts(feature))

print('Total number of missing values:\n', df_hypothyroid.isnull().sum())
