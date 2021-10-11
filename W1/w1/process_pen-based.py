import os.path

import utils

root_path = os.path.join('.', 'datasets', 'datasets')

# %%
##############################
# Pen-based PRE-PROCESSING #
##############################
print('Pen-based dataset:')
df_penbased = utils.load_arff(os.path.join(root_path, 'pen-based.arff'))
utils.convert_byte_string_to_string(df_penbased)
print(df_penbased.head())
print(df_penbased.dtypes)
categorical_features = utils.get_categorical_features(df_penbased)

for feature in categorical_features:
    print(df_penbased.value_counts(feature))

print('Total number of missing values:\n', df_penbased.isnull().sum())
