import os.path
from scipy.io import arff
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#root_path = os.path.join('..', 'datasets', 'datasets')

# %%
##############################
# Pen-based PRE-PROCESSING #
##############################

# the data set is returned as a record array, can be accessed much like a python dict of numpy arrays
# the data are accessible by attributes names
# meta contains information about the arff file such name and dtype of attributes
# can not read sparse data represented with {} in the file
# can read missing data represented with ? in the file, it assign NaN
dataset, meta = arff.loadarff('W1/datasets/datasets/pen-based.arff')
print(meta)
# converting into pandas dataframe
df = pd.DataFrame(dataset)

#dropping the last column because is the output label value for supervised learning.
df = df.drop('a17', 1)

print("total number of missing values: ", df.isnull().sum().sum())

#visualize the characteristic of each column using describe
df.describe()

# applying fuzzy c means to the data set

#initialize parameters
# number of samples
n = len(df)
# number of clusters
c = 3
#number of dimensions
d = len(df.iloc[0])
# m fuzzy parameter
m = 2
#max number of iterations
max_iterations = 12

def distance_measure(a, b):
    """ euclidean distance
        Params:
          object a and b
        Return:
          distance metrics
    """
    return np.power(np.power(a - b, 2).sum(), 0.5)


def initialize_u_matrix(n: int, c: int):
    """ initialize the menbership matrix
        Params:
          n the number of samples
          c the number of cluster
        Constraint:
        the sum of the menbership values for each sample for all the cluster must equal to 1

        We use the dirichlet distribution which is a distribution over x
        that fullfil the conditions xi > 0 and sum(X) = 1
    """
    return np.random.dirichlet(np.ones(c), n)


def initial_cluster_centers_random(c: int, d: int):
    """
      guess initial cluster centers
      random values for the clusters
    """
    return np.random.rand(c, d)


def initial_cluster_centers_from_data(c: int):
    cc = np.zeros((c, d))
    rng = np.random.default_rng()
    for i in range(c):
        random_number = rng.integers(10992)
        random_cluster = df.iloc[random_number]
        cc[i] = random_cluster
    return cc


def update_u_matrix(cc):
    """
    Params:
      cluster centers
    Return:
      updated menbership matrix
    """
    # initialize new matrix
    u_matrix = np.zeros((n, c))
    # column
    for i in range(c):
        # row
        for k in range(n):
            # for each cluster center
            xk = df.iloc[k]
            vi = cc[i]
            numerator = distance_measure(xk, vi)
            res = np.zeros((c,))
            for j in range(c):
                denominator = distance_measure(xk, cc[j])
                res[j] = numerator / denominator
            res = np.power(np.power(res, 2 / (m - 1)).sum(), -1)
            u_matrix[k][i] = res
    return u_matrix


def update_cluster_centers(u):
    """
    Params:
      menbership matrix
    Return:
      updated cluster centers
    """
    cc = []
    for i in range(c):
        denominator = np.power(u[:, i], m).sum()
        cj = []
        for x in range(d):
            numerator = (df.iloc[:, x].values * np.power(u[:, i], m)).sum()
            c_val = numerator / denominator
            cj.append(c_val)
        cc.append(cj)
    return np.array(cc)


def compare_cluster_centers(c_new, c_current):
    """
    Compare unext with ucurrent to calculate the value to be compare against
    epsilon.
    """
    return (c_new - c_current).max()


def plot_data(location_of_plot, clusters):
    plt.subplot(4, 3, location_of_plot + 1)

    # selecting 2 columns to visualize the data
    plt.scatter(list(df.iloc[:, 2]), list(df.iloc[:, 3]), marker='o')
    for center in clusters:
        plt.scatter(center[2], center[3], marker='o', color='r')
    plt.axis('equal')
    plt.xlabel('column a2', fontsize=16)
    plt.ylabel('column a3', fontsize=16)
    plt.grid()


def fuzzy_c_means(m=2, epsilon=0.001, max_iterations=12):
    """
    FCM
    Params:
      c: number of cluter
      m: fuzzy exponent parameter
      epsilon: threshold criteria
      max_iterations
    """
    u_matrix = initialize_u_matrix(n=n, c=c)
    clusters_centers = initial_cluster_centers_random(c, d)
    # iterative part
    for i in range(max_iterations):
        plot_data(i, clusters_centers)
        plt.show()
        clusters_centers_new = update_cluster_centers(u_matrix)
        u_matrix_new = update_u_matrix(clusters_centers_new)

        delta_u = compare_cluster_centers(clusters_centers_new, clusters_centers)
        if delta_u <= epsilon:
            break
        else:
            u_matrix = u_matrix_new
            clusters_centers = clusters_centers_new
    return u_matrix, clusters_centers

#VALIDATION METRIC
#PARTITION COEFFICIENT
"""
Measure the amount of overlap among clusters
the range of values is [1/c , 1] -> 
the closer to 1 =  the smaller the sharing of the vectors in data
the closer to 1/c =  X possesses no clustering structure or the algorithms fails to unravel it.
 """
def partition_coefficient(u, n):
  return (np.power(u, 2).sum())/n

partition_coefficient(uuu, n)
