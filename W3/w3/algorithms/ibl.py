import math
import random
import time

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
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


def feature_selection(data, gs, method):
    data = data.apply(lambda x: x.astype('category').cat.codes if x.dtype == 'object' else x)
    if method == 'kbest':
        if data.shape[1] < 10:
            return SelectKBest(mutual_info_classif, k='all').fit(data, gs).scores_

        return SelectKBest(mutual_info_classif, k=10).fit(data, gs).scores_
    elif method == 'variance':
        return VarianceThreshold(0.03).fit(data, gs).get_support()


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


def get_bounds(p, z, n):
    if (p * n) == 0:
        return 0, 0

    p /= n
    left = p + z ** 2 / (2 * n)
    right = z * np.sqrt((p * (1 - p)) / n + (z ** 2 / (4 * n ** 2)))
    denominator = 1 + z ** 2 / n
    upper = (left + right) / denominator
    lower = (left - right) / denominator
    return upper, lower


def is_acceptable(p_accuracy, n_accuracy, p_frequency, n_frequency):
    _, lower_accuracy = get_bounds(p_accuracy, 0.9, n_accuracy)
    upper_frequency, _ = get_bounds(p_frequency, 0.9, n_frequency)
    return lower_accuracy > upper_frequency


def is_significantly_poor(p_accuracy, n_accuracy, p_frequency, n_frequency):
    upper_accuracy, _ = get_bounds(p_accuracy, 0.7, n_accuracy)
    _, lower_frequency = get_bounds(p_frequency, 0.7, n_frequency)
    return upper_accuracy < lower_frequency


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

    def __init__(self, dataframe: DataFrame, algorithm="ibl1", k=3, measure='euclidean', policy='most_voted',
                 selection_method='kbest'):
        self.number_samples = len(dataframe)
        self.algorithm = algorithm
        self.k = k
        self.measure = measure
        self.policy = policy
        self.selection_method = selection_method
        self.scores = None
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
                dist = k_ibl_utils.hvdm_v2(x, pd.DataFrame(), y_num, pd.DataFrame())
            else:
                dist = k_ibl_utils.distance(x, np.asarray(y[0]), metric=metric, scores=self.scores)
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
                dist = k_ibl_utils.hvdm_v2(x_num, pd.DataFrame(x_cat.reshape((1, x_cat.shape[0])), columns=columns),
                                           y_num, pd.DataFrame(y_cat.reshape((1, y_cat.shape[0])), columns=columns))
            else:
                dist = k_ibl_utils.distance(x_num, y_num, x_cat, y_cat, metric, scores=self.scores)
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

    def _ibl3(self, numerical_features, cat_features, labels):
        # if the dataframe does not contain categorical features.
        if cat_features.size == 0:
            self._numerical_ibl3(numerical_features, labels)
        else:
            accuracy = [{'count_at_least_as_close': 0, 'count_matched': 0} for _ in range(numerical_features.shape[0])]
            for i in range(self.number_samples):
                # reading each sample and its label.
                x_num = numerical_features[i]
                x_cat = cat_features[i]
                label = labels[i]

                # when the CD is empty.
                if not self.cd:
                    accuracy[i]['count_at_least_as_close'] = 1
                    accuracy[i]['count_matched'] = 1
                    self.saved_samples += 1
                    self.cd_index_ibl3 = [i]
                    self.cd.add((tuple(x_num), tuple(x_cat), label, 0, 0))
                else:
                    nearest_acceptable_index = None
                    nearest_distance = np.inf
                    distances = np.ones((len(self.cd),))
                    cd_labels = []
                    for idx, y in enumerate(self.cd):
                        y_num = y[0]
                        y_cat = y[1]
                        y_label = y[2]
                        y_dist = k_ibl_utils.distance(x_num, y_num, x_cat, y_cat)
                        distances[idx] = y_dist
                        cd_labels.append(y_label)
                        if is_acceptable(p_accuracy=y[4],  # count matched
                                         n_accuracy=y[3],  # count at least as close
                                         p_frequency=sum([1 for s in self.cd if s[2] == y_label]),
                                         n_frequency=len(self.cd)) and y_dist < nearest_distance:
                            nearest_acceptable_index = idx
                            nearest_distance = y_dist

                    if nearest_acceptable_index is None:
                        # get random
                        if distances.shape[0] == 1:
                            nearest_acceptable_index = 0
                        else:
                            nearest_acceptable_index = random.choice(range(distances.shape[0]))
                        distances.sort()
                        nearest_distance = distances[nearest_acceptable_index]

                    if label == cd_labels[nearest_acceptable_index]:
                        self.correct_samples += 1
                    else:
                        self.incorrect_samples += 1
                        self.saved_samples += 1
                        self.cd.add((tuple(x_num), tuple(x_cat), label, 0, 0))

                    # x_num, x_cat, ..., nearest_distance
                    for j, y in enumerate(self.cd.copy()):
                        y_num = np.asarray(y[0])
                        y_cat = np.asarray(y[1])
                        y_label = y[2]
                        count_as_close = y[3]
                        count_matched = y[4]
                        y_distance = k_ibl_utils.distance(x_num, y_num, x_cat, y_cat)
                        if y_distance <= nearest_distance:
                            self.cd.remove(y)
                            count_as_close += 1
                            if label == y_label:
                                count_matched += 1
                            self.cd.add((tuple(y_num), tuple(y_cat), y_label, count_as_close, count_matched))

                            if is_significantly_poor(p_accuracy=count_matched, n_accuracy=count_as_close,
                                                     p_frequency=sum(
                                                         [1 for s in range(len(self.cd)) if labels[s] == y_label]),
                                                     n_frequency=len(self.cd)):
                                self.cd.remove((tuple(y_num), tuple(y_cat), y_label, count_as_close, count_matched))

    def _numerical_ibl3(self, numerical_features, labels):
        accuracy = [{'count_at_least_as_close': 0, 'count_matched': 0} for _ in range(numerical_features.shape[0])]
        for i in range(self.number_samples):
            # reading each sample and its label.
            x = numerical_features[i]
            label = labels[i]

            # when the CD is empty.
            if not self.cd:
                accuracy[i]['count_at_least_as_close'] = 1
                accuracy[i]['count_matched'] = 1
                self.saved_samples += 1
                self.cd.add((tuple(x), label, 0, 0))
            else:
                nearest_acceptable_index = None
                nearest_distance = np.inf
                distances = np.ones((len(self.cd),))
                cd_labels = []
                for idx, y in enumerate(self.cd):
                    y_num = y[0]
                    y_label = y[1]
                    y_dist = k_ibl_utils.distance(x, y_num)
                    distances[idx] = y_dist
                    cd_labels.append(y_label)
                    if is_acceptable(p_accuracy=y[3],
                                     n_accuracy=y[2],
                                     p_frequency=sum([1 for s in self.cd if s[1] == y_label]),
                                     n_frequency=len(self.cd)) and y_dist < nearest_distance:
                        nearest_acceptable_index = idx
                        nearest_distance = y_dist

                if nearest_acceptable_index is None:
                    # get random
                    if distances.shape[0] == 1:
                        nearest_acceptable_index = 0
                    else:
                        nearest_acceptable_index = random.choice(range(distances.shape[0]))
                    distances.sort()
                    nearest_distance = distances[nearest_acceptable_index]

                if label == labels[nearest_acceptable_index]:
                    self.correct_samples += 1
                else:
                    self.incorrect_samples += 1
                    self.saved_samples += 1
                    self.cd.add((tuple(x), label, 0, 0))

                for j, y in enumerate(self.cd.copy()):
                    y_num = np.asarray(y[0])
                    y_label = y[1]
                    count_as_close = y[2]
                    count_matched = y[3]
                    y_distance = k_ibl_utils.distance(x, y_num)
                    if y_distance <= nearest_distance:
                        self.cd.remove(y)
                        count_as_close += 1
                        if label == y_label:
                            count_matched += 1

                        self.cd.add((tuple(y_num), y_label, count_as_close, count_matched))

                        if is_significantly_poor(p_accuracy=count_matched,
                                                 n_accuracy=count_as_close, p_frequency=sum(
                                    [1 for s in range(len(self.cd)) if labels[s] == labels[j]]),
                                                 n_frequency=len(self.cd)):
                            self.cd.remove((tuple(y_num), y_label, count_as_close, count_matched))

    def _ibl3_predict(self, numerical_features, cat_features, gs):
        # if the dataframe does not contain categorical features.
        if cat_features.size == 0:
            self._ibl3_predict_numerical(numerical_features, gs)
        else:
            accuracy = [{'count_at_least_as_close': 0, 'count_matched': 0} for _ in range(len(self.cd))]
            for i in range(self.number_samples):
                # reading each sample and its label.
                x_num = numerical_features[i]
                x_cat = cat_features[i]

                nearest_acceptable_index = None
                nearest_distance = np.inf
                distances = np.ones((len(self.cd),))
                cd_labels = []
                for idx, y in enumerate(self.cd):
                    y_num = np.array(y[0])
                    y_cat = np.array(y[1])
                    cd_labels.append(y[2])
                    y_dist = k_ibl_utils.distance(x_num, y_num, x_cat, y_cat)
                    distances[idx] = y_dist
                    if is_acceptable(p_accuracy=y[4],
                                     n_accuracy=y[3],
                                     p_frequency=sum([1 for s in self.cd if s[2] == y[2]]),
                                     n_frequency=len(self.cd)) and y_dist < nearest_distance:
                        nearest_acceptable_index = idx
                        nearest_distance = y_dist

                if nearest_acceptable_index is None:
                    # get random
                    if distances.shape[0] == 1:
                        nearest_acceptable_index = 0
                    else:
                        nearest_acceptable_index = random.choice(range(distances.shape[0]))
                    distances.sort()
                    nearest_distance = distances[nearest_acceptable_index]

                label = cd_labels[nearest_acceptable_index]
                if label == gs[i]:
                    self.correct_samples += 1
                else:
                    self.incorrect_samples += 1
                    self.saved_samples += 1
                    self.cd.add((tuple(x_num), tuple(x_cat), label, 0, 0))

                for j, y in enumerate(self.cd.copy()):
                    y_num = np.asarray(y[0])
                    y_cat = np.asarray(y[1])
                    y_label = y[2]
                    count_as_close = y[3]
                    count_matched = y[4]
                    y_distance = k_ibl_utils.distance(x_num, y_num, x_cat, y_cat)
                    if y_distance <= nearest_distance:
                        self.cd.remove(y)
                        count_as_close += 1
                        if label == y_label:
                            count_matched += 1
                        self.cd.add((tuple(y_num), tuple(y_cat), y_label, count_as_close, count_matched))

                        if is_significantly_poor(p_accuracy=count_matched, n_accuracy=count_as_close,
                                                 p_frequency=sum([1 for s in self.cd if s[2] == y_label]),
                                                 n_frequency=len(self.cd)):
                            self.cd.remove((tuple(y_num), tuple(y_cat), y_label, count_as_close, count_matched))

    def _ibl3_predict_numerical(self, numerical_features, gs):
        accuracy = [{'count_at_least_as_close': 0, 'count_matched': 0} for _ in range(len(self.cd))]
        for i in range(self.number_samples):
            # reading each sample and its label.
            x_num = numerical_features[i]
            nearest_acceptable_index = None
            nearest_distance = np.inf
            distances = np.ones((len(self.cd),))
            cd_labels = []
            for idx, y in enumerate(self.cd):
                y_num = np.array(y[0])
                cd_labels.append(y[1])
                y_dist = k_ibl_utils.distance(x_num, y_num)
                distances[idx] = y_dist
                if is_acceptable(p_accuracy=y[3],
                                 n_accuracy=y[2],
                                 p_frequency=sum([1 for s in self.cd if s[1] == y[1]]),
                                 n_frequency=len(self.cd)) and y_dist < nearest_distance:
                    nearest_acceptable_index = idx
                    nearest_distance = y_dist

            if nearest_acceptable_index is None:
                # get random
                if distances.shape[0] == 1:
                    nearest_acceptable_index = 0
                else:
                    nearest_acceptable_index = random.choice(range(distances.shape[0]))
                distances.sort()
                nearest_distance = distances[nearest_acceptable_index]

            label = cd_labels[nearest_acceptable_index]
            if label == gs[i]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1
                self.saved_samples += 1
                self.cd.add((tuple(x_num), label, 0, 0))

            for j, y in enumerate(self.cd.copy()):
                y_num = np.asarray(y[0])
                y_label = y[1]
                count_as_close = y[2]
                count_matched = y[3]
                y_distance = k_ibl_utils.distance(x_num, y_num)
                if y_distance <= nearest_distance:
                    self.cd.remove(y)
                    count_as_close += 1
                    if label == y_label:
                        count_matched += 1
                    self.cd.add((tuple(y_num), y_label, count_as_close, count_matched))

                    if is_significantly_poor(p_accuracy=count_matched, n_accuracy=count_as_close,
                                             p_frequency=sum([1 for s in self.cd if s[1] == y_label]),
                                             n_frequency=len(self.cd)):
                        self.cd.remove((tuple(y_num), y_label, count_as_close, count_matched))

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
            distance_list, cd_labels = self.get_distance_num(x, measure)
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
        if self.algorithm in {"ibl1", "ibl2", "ibl3", "k-ibl", "selection-k-ibl"}:
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
                self._ibl3(numerical_features, cat_features, labels)
                self.execution_time = time.time() - start
            elif self.algorithm == "k-ibl":
                start = time.time()
                self._kibl(numerical_features, cat_features, labels, self.k, self.policy, self.measure)
                self.execution_time = time.time() - start
            elif self.algorithm == "selection-k-ibl":
                self.scores = feature_selection(
                    pd.concat([pd.DataFrame(numerical_features), pd.DataFrame(cat_features)], axis=1), labels,
                    self.selection_method)
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

    def ib3Algorithm(self, test_data):
        print('Starting IB3 algorithm')
        self._reset_evaluation_metrics()
        self.number_samples = test_data.shape[0]
        numerical_features, cat_features, gs = preprocess(test_data)
        start = time.time()
        self._ibl3_predict(numerical_features, cat_features, gs)
        self.execution_time = time.time() - start
        self.accuracy = self.correct_samples / self.number_samples
        self.accuracy = self.correct_samples / self.number_samples
        self.print_results()
        print('Finished IB3 algorithm')

    def _run_kibl_predict(self, k, measure, policy, test_data):
        self._reset_evaluation_metrics()
        self.number_samples = test_data.shape[0]
        numerical_features, cat_features, gs = preprocess(test_data)
        start = time.time()
        labels = self._kibl_predict(numerical_features, cat_features, gs, k, policy, measure)
        self.execution_time = time.time() - start
        self.accuracy = self.correct_samples / self.number_samples
        self.print_results()
        return labels

    def kIBLAlgorithm(self, test_data, k=3, measure='euclidean', policy='most_voted'):
        print(f'Starting K-IBL algorithm with k={k}, measure={measure}, policy={policy}')
        labels = self._run_kibl_predict(k, measure, policy, test_data)
        print('Finished K-IBL algorithm')

        return labels

    def selectionkIBLAlgorithm(self, test_data, k=3, measure='euclidean', policy='most_voted'):
        if self.scores is None:
            raise ValueError('You initialized the IBL instance with a method that is not selection-k-ibl')

        print(f'Starting Selection K-IBL algorithm with k={k}, measure={measure}, policy={policy}')
        labels = self._run_kibl_predict(k, measure, policy, test_data)
        print('Finished Selection K-IBL algorithm')

        return labels

    def print_results(self):
        print(f"Performance metrics of the {self.algorithm} algorithm")
        print(f"Number of saved instances: {self.saved_samples}")
        print(f"Execution time: {self.execution_time} seconds")
        print(f"Accuracy: {self.accuracy}%")
        print()
