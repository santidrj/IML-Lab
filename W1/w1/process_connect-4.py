import os.path

import w1.myfunctions as mf

root_path = os.path.join('.', 'datasets', 'datasets')

# %%
#############################
# CONNECT-4 PRE-Processing  #
#############################
print('Connect-4 dataset:')
df_connect = mf.load_arff(os.path.join(root_path, 'connect-4.arff'))
mf.convert_byte_string_to_string(df_connect)
print(df_connect.head())
print(df_connect.dtypes)
print(df_connect.value_counts())

print('Total number of missing values:\n', df_connect.isnull().sum())
