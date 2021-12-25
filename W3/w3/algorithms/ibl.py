import time

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from w3.algorithms import k_ibl_utils


def preprocess(data: DataFrame):
    """
    Splits the data into numerical and categorical variables.
    It also normalizes the numerical variables.

    :param data: Dataset to be preprocessed.
    :type data: DataFrame
    :return:
        a tuple with the normalized numerical features as the first element
        and the categorical features as the second element.
    """

    labels = data.iloc[:, -1].astype('category').cat.codes
    numeric_features = data.iloc[:, :-1].select_dtypes(include="number")
    imputer = SimpleImputer(strategy="mean").fit(numeric_features)
    numeric_features_imputed = imputer.transform(numeric_features)

    # Since SimpleImputer discards columns which only contain missing values, we have to add them again with a 0 value.
    for i, stat in enumerate(imputer.statistics_):
        if np.isnan(stat):
            numeric_features_imputed = np.insert(numeric_features_imputed, i,
                                                 np.zeros((numeric_features_imputed.shape[0],)), axis=1)

    # Apply MinMaxScaler to scale features to the [0, 1] range
    normalized_num_features = MinMaxScaler().fit_transform(numeric_features_imputed)

    categorical_features = data.iloc[:, :-1].select_dtypes(include="object").to_numpy()

    return normalized_num_features, categorical_features, labels


def distance(x_numerical, y_numerical, x_categorical=None, y_categorical=None):
    if x_categorical is None or y_categorical is None:
        return np.sqrt(np.square(x_numerical - y_numerical).sum())

    return np.sqrt(num_diff(x_numerical, y_numerical) + cat_diff(x_categorical, y_categorical))


def num_diff(x, y):
    return np.square(x - y).sum()


def cat_diff(x, y):
    dist = 0
    for i in range(len(x)):
        if x[i] == '?' and y[i] == '?':
            dist += 1
        else:
            dist += (x[i] != y[i])
    return dist


def get_class(distance_list, cd_labels, method='nn', k=3, policy='most_voted'):
    """
    Compute the class of a sample using its distance to the points in the CD.

    :param distance_list: List with the distances to the points in the CD.
    :param cd_labels: List with the labels of each sample in the CD.
    :param method: Method used to obtain the class. By default, it uses Nearest Neighbors.
    :param k: Number of neighbours to use in the voting method.
    :param policy: Policy to use for the voting method.
    :return: The class of the sample.
    """

    if method == 'nn':
        return cd_labels[np.argmin(distance_list)]
    elif method == 'voting':
        if len(cd_labels) < k:
            raise ValueError('The selected K is to big for the current CD.')
        ind = np.argpartition(distance_list, k)[:k]
        return k_ibl_utils.vote([cd_labels[i] for i in ind], policy)
    else:
        raise ValueError(f'The {method} method does not exist.')


"""
Performance dimensions:
1. Generality:

2. Accuracy: concept description classification accuracy.

3. Learning rate: speed at which classification accuracy increase during training.

4. Incorporation cost: these are cost incurred while updating the CD with a single 
  instance. they include classification cost. 

5 Storage Requirement: the sizes of the Concept description,
  for IBL is the # of saved instances used for classification.
"""


class IBL:
    """
    Instance Based Learning algorithm.

    Attributes

    - dataframe: Pandas DataFrame where the last column is the class value of each row.
    - algorithm: String which determine the IBL to execute. If the value is not pass, IBL1 is selected.
    """

    def __init__(self, dataframe: DataFrame, algorithm="ibl1", k=3, measure='euclidean', policy='most_voted'):
        self.number_samples = len(dataframe)
        self.algorithm = algorithm
        self.k = k
        self.measure = measure
        self.policy = policy
        self.correct_samples = 0
        self.incorrect_samples = 0
        self.accuracy = 0
        self.saved_samples = 0
        self.execution_time = 0
        self.cd = set()

        print("\nStarting to add train samples to CD\n")
        self._run(dataframe)
        self.print_results()
        print("Finished adding train samples to CD\n")

    def _reset_evaluation_metrics(self):
        self.number_samples = 0
        self.correct_samples = 0
        self.incorrect_samples = 0
        self.accuracy = 0
        self.saved_samples = 0
        self.execution_time = 0

    def get_distance_num(self, x, metric='euclidean'):
        distance_list = []
        cd_labels = []
        for y in self.cd:
            y_num = np.asarray(y[0])
            if metric == 'hvdm':
                dist = k_ibl_utils.hvdm(x, pd.DataFrame(), y_num, pd.DataFrame())
            else:
                dist = k_ibl_utils.distance(x, np.asarray(y[0]), metric=metric)
            distance_list.append(dist)
            cd_labels.append(y[1])
        return distance_list, cd_labels

    def get_distance_mixed(self, x_num, x_cat, metric='euclidean'):
        distance_list = []
        cd_labels = []
        for y in self.cd:
            y_num = np.asarray(y[0])
            y_cat = np.asarray(y[1])
            if metric == 'hvdm':
                columns = [f'f{i}' for i in range(x_cat.shape[0])]
                dist = k_ibl_utils.hvdm(x_num, pd.DataFrame(x_cat.reshape((1, x_cat.shape[0])), columns=columns), y_num,
                                        pd.DataFrame(y_cat.reshape((1, y_cat.shape[0])), columns=columns))
            else:
                dist = k_ibl_utils.distance(x_num, y_num, x_cat, y_cat, metric)
            distance_list.append(dist)
            cd_labels.append(y[2])
        return distance_list, cd_labels

    def _ibl1(self, numerical_features, cat_features, labels):
        if cat_features.size == 0:
            self._numerical_ibl1(numerical_features, labels)
        else:
            for i in range(self.number_samples):
                x_num = numerical_features[i]
                x_cat = cat_features[i]
                label = labels[i]

                if not self.cd:
                    self.saved_samples += 1
                    self.cd.add((tuple(x_num), tuple(x_cat), label))
                else:
                    # Obtain a list with the sample distance to each point in the CD and save it together with the point
                    # class
                    distance_list, cd_labels = self.get_distance_mixed(x_num, x_cat)

                    y_max = np.argmin(distance_list)
                    if label == cd_labels[y_max]:
                        self.correct_samples += 1
                    else:
                        self.incorrect_samples += 1
                    self.saved_samples += 1
                    self.cd.add((tuple(x_num), tuple(x_cat), label))

    def _numerical_ibl1(self, numerical_features, labels):
        for i in range(self.number_samples):
            x = numerical_features[i]
            label = labels[i]

            if not self.cd:
                self.saved_samples += 1
                self.cd.add((tuple(x), label))
            else:
                # Obtain a list with the sample distance to each point in the CD and save it together with the point
                # class
                distance_list, cd_labels = self.get_distance_num(x)

                y_max = np.argmin(distance_list)
                if label == cd_labels[y_max]:
                    self.correct_samples += 1
                else:
                    self.incorrect_samples += 1

                self.saved_samples += 1
                self.cd.add((tuple(x), label))

    def _ibl1_predict(self, numerical_features, cat_features, gs):
        if cat_features.size == 0:
            return self._ibl1_predict_numerical(numerical_features, gs)

        labels = []
        for i in range(self.number_samples):
            x_num = numerical_features[i]
            x_cat = cat_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list, cd_labels = self.get_distance_mixed(x_num, x_cat)
            label = get_class(distance_list, cd_labels)
            if label == gs[i]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1

            labels.append(label)
            self.saved_samples += 1
            self.cd.add((tuple(x_num), tuple(x_cat), label))

        return labels

    def _ibl1_predict_numerical(self, numerical_features, gs):
        labels = []
        for i in range(self.number_samples):
            x = numerical_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list, cd_labels = self.get_distance_num(x)
            label = get_class(distance_list, cd_labels)
            if label == gs[i]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1

            labels.append(label)
            self.saved_samples += 1
            self.cd.add((tuple(x), label))

        return labels

    def _ibl2(self, numerical_features, cat_features, labels):
        if cat_features.size == 0:
            self._numerical_ibl2(numerical_features, labels)
        else:
            for i in range(self.number_samples):
                x_num = numerical_features[i]
                x_cat = cat_features[i]
                label = labels[i]

                if not self.cd:
                    self.saved_samples += 1
                    self.cd.add((tuple(x_num), tuple(x_cat), label))
                else:
                    distance_list, cd_labels = self.get_distance_mixed(x_num, x_cat)

                    y_max = np.argmin(distance_list)
                    if label == cd_labels[y_max]:
                        self.correct_samples += 1
                    else:
                        self.incorrect_samples += 1
                        self.saved_samples += 1
                        self.cd.add((tuple(x_num), tuple(x_cat), label))

    def _numerical_ibl2(self, numerical_features, labels):
        for i in range(self.number_samples):
            x = numerical_features[i]
            label = labels[i]

            if not self.cd:
                self.saved_samples += 1
                self.cd.add((tuple(x), label))
            else:
                # Obtain a list with the sample distance to each point in the CD and save it together with the point
                # class
                distance_list, cd_labels = self.get_distance_num(x)

                y_max = np.argmin(distance_list)
                if label == cd_labels[y_max]:
                    self.correct_samples += 1
                else:
                    self.incorrect_samples += 1
                    self.saved_samples += 1
                    self.cd.add((tuple(x), label))

    def _ibl2_predict(self, numerical_features, cat_features, gs):
        if cat_features.size == 0:
            return self._ibl2_predict_numerical(numerical_features, gs)

        labels = []
        for i in range(self.number_samples):
            x_num = numerical_features[i]
            x_cat = cat_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list, cd_labels = self.get_distance_mixed(x_num, x_cat)
            label = get_class(distance_list, cd_labels)
            if label == gs[i]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1
                self.saved_samples += 1
                self.cd.add((tuple(x_num), tuple(x_cat), label))

            labels.append(label)

        return labels

    def _ibl2_predict_numerical(self, numerical_features, gs):
        labels = []
        for i in range(self.number_samples):
            x = numerical_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list, cd_labels = self.get_distance_num(x)

            label = get_class(distance_list, cd_labels)
            if label == gs[i]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1
                self.saved_samples += 1
                self.cd.add((tuple(x), label))

            labels.append(label)

        return labels

    def _ibl3(self):
        # TODO: Implement this
        pass

    def _kibl(self, numerical_features, cat_features, labels, k=3, policy='most_voted', measure='euclidean'):
        k_ibl_utils.init_hvdm(numerical_features,
                              pd.DataFrame(cat_features, columns=[f'f{i}' for i in range(cat_features.shape[1])]),
                              labels)
        if cat_features.size == 0:
            self._numerical_kibl(numerical_features, labels, k, policy, measure)
        else:
            for i in range(self.number_samples):
                x_num = numerical_features[i]
                x_cat = cat_features[i]
                label = labels[i]

                if not self.cd:
                    self.correct_samples += 1
                    self.saved_samples += 1
                    self.cd.add((tuple(x_num), tuple(x_cat), label))
                else:
                    # Obtain a list with the sample distance to each point in the CD and save it together with the point
                    # class
                    distance_list, cd_labels = self.get_distance_mixed(x_num, x_cat, measure)

                    if len(self.cd) <= k:
                        cls = k_ibl_utils.vote(cd_labels, policy)
                    else:
                        ind = np.argpartition(distance_list, k)[:k].astype(int)
                        cls = k_ibl_utils.vote([cd_labels[i] for i in ind], policy)
                    if label == cls:
                        self.correct_samples += 1
                    else:
                        self.incorrect_samples += 1
                    self.saved_samples += 1
                    self.cd.add((tuple(x_num), tuple(x_cat), label))

    def _numerical_kibl(self, numerical_features, labels, k=3, policy='most_voted', measure='euclidean'):
        for i in range(self.number_samples):
            x = numerical_features[i]
            label = labels[i]

            if not self.cd:
                self.correct_samples += 1
                self.saved_samples += 1
                self.cd.add((tuple(x), label))
            else:
                # Obtain a list with the sample distance to each point in the CD and save it together with the point
                # class
                distance_list, cd_labels = self.get_distance_num(x, measure)

                if len(self.cd) <= k:
                    cls = k_ibl_utils.vote(cd_labels, policy)
                else:
                    ind = np.argpartition(distance_list, k)[:k]
                    cls = k_ibl_utils.vote([cd_labels[i] for i in ind], policy)
                if label == cls:
                    self.correct_samples += 1
                else:
                    self.incorrect_samples += 1

                self.saved_samples += 1
                self.cd.add((tuple(x), label))

    def _kibl_predict(self, numerical_features, cat_features, gs, k=3, policy='most_voted', measure='euclidean'):
        if cat_features.size == 0:
            return self._kibl_predict_numerical(numerical_features, gs, k, policy, measure)

        labels = []
        for i in range(self.number_samples):
            x_num = numerical_features[i]
            x_cat = cat_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list, cd_labels = self.get_distance_mixed(x_num, x_cat, measure)
            label = get_class(distance_list, cd_labels, 'voting', k, policy)

            if label == gs[i]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1

            labels.append(label)
            self.saved_samples += 1
            self.cd.add((tuple(x_num), tuple(x_cat), label))

        return labels

    def _kibl_predict_numerical(self, numerical_features, gs, k=3, policy='most_voted', measure='euclidean'):
        labels = []
        for i in range(self.number_samples):
            x = numerical_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list, cd_labels = self.get_distance_mixed(x, measure)
            label = get_class(distance_list, cd_labels, 'voting', k, policy)

            if label == gs[i]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1

            labels.append(label)
            self.saved_samples += 1
            self.cd.add((tuple(x), label))

        return labels

    def _run(self, training_set):
        numerical_features, cat_features, labels = preprocess(training_set)
        if self.algorithm in {"ibl1", "ibl2", "ibl3", "k-ibl"}:
            if self.algorithm == "ibl1":
                start = time.time()
                self._ibl1(numerical_features, cat_features, labels)
                self.execution_time = time.time() - start
            elif self.algorithm == "ibl2":
                start = time.time()
                self._ibl2(numerical_features, cat_features, labels)
                self.execution_time = time.time() - start
            elif self.algorithm == "ibl3":
                start = time.time()
                self._ibl3()
                self.execution_time = time.time() - start
            elif self.algorithm == "k-ibl":
                start = time.time()
                self._kibl(numerical_features, cat_features, labels, self.k, self.policy, self.measure)
                self.execution_time = time.time() - start

            self.accuracy = self.correct_samples / self.number_samples
        else:
            raise ValueError("You selected a wrong Instance-Based Learning algorithm")

    def ib1Algorithm(self, test_data):
        print('Starting IB1 algorithm')
        self._reset_evaluation_metrics()
        self.number_samples = test_data.shape[0]
        numerical_features, cat_features, gs = preprocess(test_data)
        start = time.time()
        labels = self._ibl1_predict(numerical_features, cat_features, gs)
        self.execution_time = time.time() - start
        self.accuracy = self.correct_samples / self.number_samples
        self.accuracy = self.correct_samples / self.number_samples
        self.print_results()
        print('Finished IB1 algorithm')

        return labels

    def ib2Algorithm(self, test_data):
        print('Starting IB2 algorithm')
        self._reset_evaluation_metrics()
        self.number_samples = test_data.shape[0]
        numerical_features, cat_features, gs = preprocess(test_data)
        start = time.time()
        labels = self._ibl2_predict(numerical_features, cat_features, gs)
        self.execution_time = time.time() - start
        self.accuracy = self.correct_samples / self.number_samples
        self.accuracy = self.correct_samples / self.number_samples
        self.print_results()
        print('Finished IB2 algorithm')

        return labels

    def kIBLAlgorithm(self, test_data, k=3, measure='euclidean', policy='most_voted'):
        print('Starting K-IBL algorithm')
        self._reset_evaluation_metrics()
        self.number_samples = test_data.shape[0]
        numerical_features, cat_features, gs = preprocess(test_data)
        start = time.time()
        labels = self._kibl_predict(numerical_features, cat_features, gs, k, policy, measure)
        self.execution_time = time.time() - start
        self.accuracy = self.correct_samples / self.number_samples
        self.print_results()
        print('Finished K-IBL algorithm')

        return labels

    def print_results(self):
        print(f"Performance metrics of the {self.algorithm} algorithm")
        print(f"Number of saved instances: {self.saved_samples}")
        print(f"Execution time: {self.execution_time} seconds")
        print(f"Accuracy: {self.accuracy}%")
        print()
