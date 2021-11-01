import pandas as pd
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class KModes:
    def __init__(self, data, k, max_iter=30):
        """
        K-modes clustering algorithm

        :param data: The DataFrame to which the algorithm will be applied.
        :param k: The number of clusters to form.
        :param max_iter: Maximum number of iterations the algorithm will perform in a single run.
        """
        self.k = k
        self.max_iter = max_iter
        self.df = data
        self.dist = pd.DataFrame(index=range(self.df.shape[0]), columns=range(k), dtype=int)
        self.centroids = pd.DataFrame(index=range(k), columns=range(self.df.shape[1]))
        self.df['class'] = ''

    def init_centroids(self, method):
        """
        Initialization of centroids.

        :param method: Initialization method. Available methods are: 'simple' and 'bisecting'.
        """
        if method == 'random':
            self.centroids = self.df.iloc[:, :-1].sample(n=self.k)

        if method == 'bisecting':
            self.df['class'] = KModes(self.df.iloc[:, :-1], k=2, max_iter=10).run(init='random')
            max_class = self.df['class'].value_counts().idxmax()

            for k in range(self.k - 2):
                X = self.df[self.df['class'] == max_class].iloc[:, :-1].reset_index(drop=True)
                classes = KModes(X, k=2, max_iter=10).run(init='random')

                mod_class = classes.copy()
                mod_class[classes == 0], mod_class[classes == 1] = max_class, self.df['class'].max() + 1

                self.df['class'][self.df['class'] == max_class] = mod_class.values
                max_class = self.df['class'].value_counts().idxmax().astype(int)

            self.update_centroids()

    def distance(self, x, y):
        """
        Calculates the number of matches between two vectors.

        :param x: First vector. It will correspond to the samples.
        :param y: Second vector. It will correspond to a centroid.
        :return: Number of matches between x and y.
        """
        return x.eq(y.values, axis='columns').sum(axis=1)

    def fit(self):
        """
        Assigns points to clusters.

        :return: Vector of predicted labels.
        """
        for k in range(self.k):
            self.dist.iloc[:, k] = self.distance(x=self.df.iloc[:, :-1], y=self.centroids.iloc[k, :]).values
        return self.dist.T.idxmax().astype(int)

    def update_centroids(self):
        """
        Recalculation of centroids after assigning points to clusters.

        """
        for k in range(self.k):
            k_centroid = self.df[self.df['class'] == k].iloc[:, :-1].mode()

            if not k_centroid.empty:
                self.centroids.iloc[k, :] = k_centroid.iloc[0, :]

    def run(self, init='random'):
        """
        Executes the algorithm until either the maximum number
        of iterations is reached or no point changes clusters.

        :param init: Initialization method. Available methods are: 'simple' and 'bisecting'.
        :return: Vector of predicted labels.
        """
        self.init_centroids(init)
        iters = 0
        convergence = 1
        while iters <= self.max_iter and convergence != 0:
            last_class = self.df['class'].copy()

            self.df['class'] = self.fit()
            self.update_centroids()

            iters += 1
            convergence = (last_class != self.df['class']).sum()

        return self.df['class']
