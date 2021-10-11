import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.preprocessing import RobustScaler

import utils

root_path = os.path.join('..', 'datasets', 'datasets')

# %%
###########################
# HEART-C PRE-PROCESSING  #
###########################
# Load the Heart-C dataset
df_heart = utils.load_arff(os.path.join(root_path, 'heart-c.arff'))
df_heart.drop(columns='num', inplace=True)
utils.convert_byte_string_to_string(df_heart)
print()
print('Heart-C dataset:')
print(df_heart.head())
print(df_heart.dtypes)

# Get all the categorical features of the dataset for later processing
categorical_features = df_heart.select_dtypes(include='object').columns

for feature in categorical_features:
    print(f'{df_heart.value_counts(feature)}\n')

# Treat missing values
print('Total number of missing values:\n', df_heart.isnull().sum())
print()
print('ca possible values:\n', df_heart['ca'].value_counts())

df_heart['ca'] = df_heart['ca'].fillna(0.0)

print(df_heart.describe())

# Normalize data
df_heart_numerical = df_heart.select_dtypes(include='number')
df_heart_normalized = utils.normalize_data(df_heart_numerical, df_heart_numerical.columns, RobustScaler())

# Transform categorical values to numerical
df_heart_categorical = utils.categorical_to_numerical(df_heart)
df_heart_normalized = pd.concat([df_heart_normalized, df_heart_categorical], axis=1)
print()
print(df_heart_normalized.head())

# for col in df_heart_numerical.columns:
#     sns.boxplot(x=df_heart_numerical[col])
#     sns.stripplot(x=df_heart_numerical[col],
#                   jitter=True,
#                   marker='o',
#                   alpha=0.5,
#                   color='black')
#     plt.show()

optics_model = OPTICS(metric='euclidean')
optics_model.fit(df_heart_normalized)

# Producing the labels according to the DBSCAN technique
labels1 = cluster_optics_dbscan(reachability=optics_model.reachability_,
                                core_distances=optics_model.core_distances_,
                                ordering=optics_model.ordering_, eps=np.inf)

# Creating a numpy array with numbers at equal spaces till
# the specified range
space = np.arange(len(df_heart_normalized))

# Storing the reachability distance of each point
reachability = optics_model.reachability_[optics_model.ordering_]

# Storing the cluster labels of each point
labels = optics_model.labels_[optics_model.ordering_]

print(labels)

# Defining the framework of the visualization
plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])

# Plotting the Reachability-Distance Plot
colors = ['c.', 'b.', 'r.', 'y.', 'g.']
for Class, colour in zip(range(0, 5), colors):
    Xk = space[labels == Class]
    Rk = reachability[labels == Class]
    ax1.plot(Xk, Rk, colour, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability Distance')
ax1.set_title('Reachability Plot')

# Plotting the OPTICS Clustering
colors = ['c.', 'b.', 'r.', 'y.', 'g.', 'm.', 'maroon']
for Class, colour in zip(range(7), colors):
    Xk = df_heart_normalized[optics_model.labels_ == Class]
    ax2.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3)

ax2.plot(df_heart_normalized.iloc[optics_model.labels_ == -1, 0],
         df_heart_normalized.iloc[optics_model.labels_ == -1, 1],
         'k+', alpha=0.1)
ax2.set_title('OPTICS Clustering')

# Plotting the DBSCAN Clustering
colors = ['c', 'b', 'r', 'y', 'g', 'greenyellow']
for Class, colour in zip(range(0, 6), colors):
    Xk = df_heart_normalized[labels1 == Class]
    ax3.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], colour, alpha=0.3, marker='.')

ax3.plot(df_heart_normalized.iloc[labels1 == -1, 0],
         df_heart_normalized.iloc[labels1 == -1, 1],
         'k+', alpha=0.1)
ax3.set_title('DBSCAN clustering')

plt.tight_layout()
plt.show()
