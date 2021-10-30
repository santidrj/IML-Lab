import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

# VALIDATION METRIC
# PARTITION COEFFICIENT
"""
Measure the amount of overlap among clusters
the range of values is [1/c , 1] -> 
the closer to 1 =  the smaller the sharing of the vectors in data
the closer to 1/c =  X possesses no clustering structure or the algorithms fails to unravel it.
 """


def partition_coefficient(u, n):
    return (np.power(u, 2).sum()) / n


def distance_measure(point, center):
    """ euclidean distance
        Params:
          object a and b
        Return:
          distance metrics
    """
    return distance.euclidean(point, center)


def initialize_u_matrix(n: int, c: int):
    """ initialize the membership matrix
        Params:
          n the number of samples
          c the number of cluster
        Constraint:
        the sum of the membership values for each sample for all the cluster must equal to 1

        We use the dirichlet distribution which is a distribution over x
        that fulfils the conditions xi > 0 and sum(X) = 1
    """
    return np.random.dirichlet(np.ones(c), n)


class FuzzyCMeans:
    def __init__(self, c, m=2, epsilon=0.001, max_iter=12):
        self.c = c
        self.m = m
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.n = 0
        self.d = 0
        self.df = None
        self.u_matrix = []
        self.centers = []

    def initial_cluster_centers_random(self):
        """
          guess initial cluster centers
          random values for the clusters
        """
        return self.df[np.random.choice(self.df.shape[0], self.c, replace=False), :]

    def initial_cluster_centers_from_data(self, c: int):
        cc = np.zeros((self.c, self.d))
        rng = np.random.default_rng()
        for i in range(c):
            random_number = rng.integers(10992)
            random_cluster = self.df[random_number]
            cc[i] = random_cluster
        return cc

    def update_u_matrix(self, cc):
        """
        Params:
          cluster centers
        Return:
          updated menbership matrix
        """
        n = self.n
        c = self.c
        df = self.df
        m = self.m
        # initialize new matrix
        u_matrix = np.zeros((c, n))
        # column
        for i in range(c):
            # row
            for k in range(n):
                # for each cluster center
                xk = df[k]
                vi = cc[i]
                if (xk == vi).all():
                    u_matrix[i][k] = 1.0
                else:
                    numerator = distance_measure(xk, vi)
                    res = np.zeros((c,))
                    for j in range(c):
                        if (xk != cc[j]).any():
                            denominator = distance_measure(xk, cc[j])
                            res[j] = numerator / denominator
                    res = np.power(np.power(res, 2 / (m - 1)).sum(), -1)
                    u_matrix[i][k] = res
        return u_matrix

    def update_cluster_centers(self, u):
        """
        Params:
          menbership matrix
        Return:
          updated cluster centers
        """
        cc = []
        for i in range(self.c):
            denominator = np.power(u[i, :], self.m).sum()
            numerator = np.ndarray((self.n, self.d))
            for k in range(self.n):
                nk = (u[i][k] ** self.m) * self.df[k, :]
                numerator[k] = nk
            cc.append(numerator.sum(axis=0) / denominator)
        return np.array(cc)

    def compare_cluster_centers(self, c_new, c_current):
        """
        Compare unext with ucurrent to calculate the value to be compare against
        epsilon.
        """

        return np.array([distance.minkowski(c_new[i], c_current[i], p=1) for i in range(len(c_new))]).max()

    def plot_data(self, clusters):
        df = self.df

        # selecting 2 columns to visualize the data
        plt.scatter(list(df[:, 0]), list(df[:, 1]), marker='.', s=1)
        for center in clusters:
            plt.scatter(center[0], center[1], marker='.', color='r')
        plt.axis('equal')
        plt.grid()
        plt.show()

    def fit(self, data):
        """
        FCM
        Params:
          data: DataFrame to fit.
        """
        self.df = data.to_numpy()
        self.n, self.d = data.shape

        u_matrix = initialize_u_matrix(n=self.n, c=self.c)
        clusters_centers = self.initial_cluster_centers_random()
        # iterative part
        for i in range(self.max_iter):
            self.plot_data(clusters_centers)
            u_matrix_new = self.update_u_matrix(clusters_centers)
            clusters_centers_new = self.update_cluster_centers(u_matrix_new)

            delta_u = self.compare_cluster_centers(clusters_centers_new, clusters_centers)
            if delta_u <= self.epsilon:
                break
            else:
                u_matrix = u_matrix_new
                clusters_centers = clusters_centers_new

        self.u_matrix = u_matrix
        self.centers = clusters_centers
        return u_matrix, clusters_centers
