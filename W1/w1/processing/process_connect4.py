import os.path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import utils

data_root_path = os.path.join('..', '..', 'datasets')

print('Connect-4 dataset:')
df_connect4 = utils.load_arff(os.path.join(data_root_path, 'datasets', 'connect-4.arff'))
utils.convert_byte_string_to_string(df_connect4)

def column_values(df):
    for feature in df.columns:
        print(df.value_counts(feature))

def label_encoder(df):
    df_encoded = df.copy()
    for feature in df.iloc[:, :-1].columns:
        df_encoded[feature] = LabelEncoder().fit_transform(df[feature])
    return df_encoded

df_connect4_encoded = label_encoder(df_connect4)

save_path = os.path.join(data_root_path, 'processed')
pd.to_pickle(df_connect4, os.path.join(save_path, 'processed_connect4.pkl'))
pd.to_pickle(df_connect4_encoded, os.path.join(save_path, 'encoded_connect4.pkl'))
pd.to_pickle(df_connect4_encoded.sample(n=10000), os.path.join(save_path, 'encoded_subset_connect4.pkl'))

pos_values = set(df_connect4_encoded['a1'])
# print(f'Possible values after encoding: {pos_values}')


"""
connect_pca = PCA(2).fit_transform(df_connect_encoded)
df_connect_encoded_pca = pd.DataFrame(connect_pca, columns=['a1', 'a2'])
print(df_connect_encoded_pca.head())
"""
"""

classes = KModes(df_connect, k=4, max_iter=10).run('bisecting', 'simple')
print(classes.value_counts())
print(correct_class.value_counts())
np.save('predicted_classes_k_modes.npy', classes)

classes = np.load('../predicted_classes_k_modes.npy')
utils.print_metrics(df_connect_encoded, true_class, classes, 4)
"""


