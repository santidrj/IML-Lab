import os.path
import warnings

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.preprocessing import LabelEncoder

import utils

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

root_path = os.path.join('..', 'datasets', 'datasets')

#############################
# CONNECT-4 PRE-Processing  #
#############################
print('Connect-4 dataset:')
df_connect = utils.load_arff(os.path.join(root_path, 'connect-4.arff'))
utils.convert_byte_string_to_string(df_connect)


def column_values(df):
    for feature in df.columns:
        print(df.value_counts(feature))


# column_values(df_connect)
# print(df_connect.head())

correct_class = df_connect['class']
df_connect.drop(columns='class', inplace=True)


## ENCODING

def label_encoder(df):
    df_encoded = df.copy()
    for feature in df.columns:
        df_encoded[feature] = LabelEncoder().fit_transform(df[feature])
    return df_encoded


df_connect_encoded = label_encoder(df_connect)
pos_values = set(df_connect_encoded['a1'])
# print(f'Possible values after encoding: {pos_values}')

"""
## OPTICS
start = time.time()
clusters = OPTICS(min_samples = 2).fit_predict(df_connect_encoded)
end = time.time()

np.save('clusters.npy', clusters)

comp_time = end-start
print(f'OPTICS computation time: {comp_time/60.} minutes')
with open('info.txt', 'w') as f:
    f.write('*OPTICS computation time: \n' + str(comp_time))

clusters = np.load('clusters.npy')
"""

"""
connect_pca = PCA(2).fit_transform(df_connect_encoded)
df_connect_encoded_pca = pd.DataFrame(connect_pca, columns=['a1', 'a2'])
print(df_connect_encoded_pca.head())
"""


class KModes:
    def __init__(self, data, k, max_iter=30):
        self.k = k
        self.max_iter = max_iter
        self.df = data
        self.dist = pd.DataFrame(index=range(self.df.shape[0]), columns=range(k), dtype=int)
        self.centroids = pd.DataFrame(index=range(k), columns=range(self.df.shape[1]), dtype=int)
        self.df['class'] = ''

    def init_centroids(self, method):
        if method == 'random':
            self.centroids = self.df.iloc[:, :-1].sample(n=self.k)

        if method == 'bisecting':
            self.df['class'] = KModes(self.df.iloc[:, :-1], k=2, max_iter=10).run(init='random')
            max_class = self.df['class'].value_counts().idxmax()
            print(self.df['class'].value_counts())

            for k in range(self.k - 2):
                print('hey')
                X = self.df[self.df['class'] == max_class].iloc[:, :-1].reset_index(drop=True)
                classes = KModes(X, k=2, max_iter=10).run(init='random')

                mod_class = classes.copy()
                mod_class[classes == 0], mod_class[classes == 1] = max_class, self.df['class'].max() + 1

                self.df['class'][self.df['class'] == max_class] = mod_class
                print(self.df['class'].value_counts())
                max_class = self.df['class'].value_counts().idxmax().astype(int)

            self.update_centroids()
            # print(self.centroids)
            # print(self.df['class'].value_counts())
            print('K-bisecting has finished.')

    def distance(self, x, y, metric):
        y = y.astype(x.dtypes[0])
        if metric == 'simple':
            dist = x.eq(y.values, axis='columns')
            return dist.sum(axis=1).astype(x.dtypes[0])

        if metric == 'distribution':
            pass

    def fit(self, dist_metric):
        for k in range(self.k):
            A = self.distance(x=self.df.iloc[:, :-1], y=self.centroids.iloc[k, :], metric=dist_metric)
            self.dist.iloc[:, k] = A.values

        # self.df['class'] = self.dist.T.idxmax().astype(int)
        return self.dist.T.idxmax().astype(int)

    def update_centroids(self):
        for k in range(self.k):
            k_centroid = self.df[self.df['class'] == k].iloc[:, :-1].mode()

            if not k_centroid.empty:
                self.centroids.iloc[k, :] = k_centroid.iloc[0, :]

    def run(self, init='random', dist_metric='simple'):
        self.init_centroids(init)
        iters = 0
        convergence = 1
        while iters <= self.max_iter and convergence != 0:
            # last_class = self.df['class'].copy()

            self.df['class'] = self.fit(dist_metric)
            print(self.df['class'].value_counts())
            self.update_centroids()

            iters += 1

            # convergence = (last_class != self.df['class']).sum()
            # print(f'Convergence: {convergence}')

        return self.df['class']


classes_ = KModes(df_connect_encoded, k=4, max_iter=10).run('bisecting', 'simple')
print(classes_.value_counts())
print(correct_class.value_counts())
