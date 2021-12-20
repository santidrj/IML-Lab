import time

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


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

    numeric_features = data.select_dtypes(include="number")
    # numeric_features = SimpleImputer(strategy="mean").fit_transform(numeric_features)

    # Apply MinMaxScaler to scale features to the [0, 1] range
    normalized_num_features = MinMaxScaler().fit_transform(numeric_features)

    categorical_features = data.select_dtypes(include="object").to_numpy()

    return normalized_num_features, categorical_features


def num_distance(x, y):
    """
    x and y should be numpy arrays
    parameters:
      x and y: sample of n dimension
    return:
      the similarity distance between x and y
    """
    x_aux = x.copy()
    y_aux = y.copy()
    x_nan = np.isnan(x)
    y_nan = np.isnan(y)
    x_low = x < 0.5
    x_high = x >= 0.5
    y_low = y < 0.5
    y_high = y >= 0.5

    x_aux[x_nan & y_nan] = 0
    x_aux[x_nan & y_low] = 1
    x_aux[x_nan & y_high] = 0
    y_aux[y_nan & x_nan] = 1
    y_aux[y_nan & x_low] = 1
    y_aux[y_nan & x_high] = 0

    return np.sqrt(np.square(x_aux - y_aux).sum())

    # diff_sum = 0
    # for i in range(len(x)):
    #     if np.isnan(x[i]) and np.isnan(y[i]):
    #         diff_sum += 1
    #     elif np.isnan(x[i]) and y[i] < 0.5:
    #         diff_sum += ((1 - y[i]) ** 2)
    #     elif np.isnan(x[i]) and y[i] >= 0.5:
    #         diff_sum += ((-y[i]) ** 2)
    #     elif np.isnan(y[i]) and x[i] < 0.5:
    #         diff_sum += ((x[i] - 1) ** 2)
    #     elif np.isnan(y[i]) and x[i] >= 0.5:
    #         diff_sum += (x[i] ** 2)
    #     else:
    #         diff_sum += ((x[i] - y[i]) ** 2)
    #
    # return math.sqrt(diff_sum)


def cat_distance(x, y):
    dist = 0
    for i in range(len(x)):
        if x[i] is None or y[i] is None:
            dist += 1
        else:
            dist += (x[i] != y[i])
    return dist


def get_class(distance_list, cd_labels, method='nn'):
    """
    Compute the class of a sample using its distance to the points in the CD.

    :param distance_list: List with the distances to the points in the CD.
    :param cd_labels: List with the labels of each sample in the CD.
    :param method: Method used to obtain the class. By default, it uses Nearest Neighbors.
    :return: The class of the sample.
    """

    # TODO: implement voting system
    if method == 'nn':
        return cd_labels[np.argmin(distance_list)]
    return 0


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

    def __init__(self, dataframe: DataFrame, algorithm="ibl1"):
        self.number_samples = len(dataframe)
        self.algorithm = algorithm
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
        self.correct_samples = 0
        self.incorrect_samples = 0
        self.accuracy = 0
        self.saved_samples = 0
        self.execution_time = 0

    def _ibl1(self, numerical_features, cat_features, labels):
        if cat_features.size == 0:
            self._numerical_ibl1(numerical_features, labels)
        else:
            for i in range(self.number_samples):
                x_num = numerical_features[i]
                x_cat = cat_features[i]
                label = labels[i]

                if not self.cd:
                    self.cd.add((tuple(x_num), tuple(x_cat), label))
                else:
                    # Obtain a list with the sample distance to each point in the CD and save it together with the point
                    # class
                    distance_list, cd_labels = self.get_distance_mixed(x_cat, x_num)

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

    def _ibl1_predict(self, numerical_features, cat_features):
        if cat_features.size == 0:
            return self._ibl1_predict_numerical(numerical_features)

        labels = []
        for i in range(numerical_features.shape[0]):
            x_num = numerical_features[i]
            x_cat = cat_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list = []
            cd_labels = []
            for y in self.cd:
                distance_list.append(num_distance(x_num, np.asarray(y[0])) + cat_distance(x_cat, np.asarray(y[1])))
                cd_labels.append(y[2])
                num_distance(x_num, np.asarray(y[0])) + cat_distance(x_cat, np.asarray(y[1]))

            most_similar = np.argmin(distance_list)
            label = get_class(distance_list, cd_labels)
            if label == cd_labels[most_similar]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1

            labels.append(label)
            self.saved_samples += 1
            self.cd.add((tuple(x_num), tuple(x_cat), label))

        return labels

    def _ibl1_predict_numerical(self, numerical_features):
        labels = []
        for i in range(numerical_features.shape[0]):
            x = numerical_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list, cd_labels = self.get_distance_num(x)

            most_similar = np.argmin(distance_list)
            label = get_class(distance_list, cd_labels)
            if label == cd_labels[most_similar]:
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
                    self.cd.add((tuple(x_num), tuple(x_cat), label))
                else:
                    distance_list, cd_labels = self.get_distance_mixed(x_cat, x_num)

                    y_max = np.argmin(distance_list)
                    if label == cd_labels[y_max]:
                        self.correct_samples += 1
                    else:
                        self.incorrect_samples += 1
                        self.saved_samples += 1
                        self.cd.add((tuple(x_num), tuple(x_cat), label))

    def _numerical_ibl2(self, numerical_features, labels):
        for i in range(numerical_features.shape[0]):
            x = numerical_features[i]
            label = labels[i]

            if not self.cd:
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

    def _ibl2_predict(self, numerical_features, cat_features):
        if cat_features.size == 0:
            return self._ibl2_predict_numerical(numerical_features)

        labels = []
        for i in range(numerical_features.shape[0]):
            x_num = numerical_features[i]
            x_cat = cat_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list = []
            cd_labels = []
            for y in self.cd:
                distance_list.append(num_distance(x_num, np.asarray(y[0])) + cat_distance(x_cat, np.asarray(y[1])))
                cd_labels.append(y[2])
                num_distance(x_num, np.asarray(y[0])) + cat_distance(x_cat, np.asarray(y[1]))

            most_similar = np.argmin(distance_list)
            label = get_class(distance_list, cd_labels)
            if label == cd_labels[most_similar]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1
                self.saved_samples += 1
                self.cd.add((tuple(x_num), tuple(x_cat), label))

            labels.append(label)

        return labels

    def _ibl2_predict_numerical(self, numerical_features):
        labels = []
        for i in range(numerical_features.shape[0]):
            x = numerical_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list, cd_labels = self.get_distance_num(x)

            most_similar = np.argmin(distance_list)
            label = get_class(distance_list, cd_labels)
            if label == cd_labels[most_similar]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1
                self.saved_samples += 1
                self.cd.add((tuple(x), label))

            labels.append(label)

        return labels

    def get_distance_num(self, x):
        distance_list = []
        cd_labels = []
        for y in self.cd:
            distance_list.append(num_distance(x, np.asarray(y[0])))
            cd_labels.append(y[1])
        return distance_list, cd_labels

    def get_distance_mixed(self, x_cat, x_num):
        distance_list = []
        cd_labels = []
        for y in self.cd:
            distance_list.append(
                num_distance(x_num, np.asarray(y[0])) + cat_distance(x_cat, np.asarray(y[1]))
            )
            cd_labels.append(y[2])
        return distance_list, cd_labels

    def _ibl3(self):
        # TODO: Implement this
        pass

    def _run(self, training_set):
        # TODO: Implement below
        labels = training_set.iloc[:, -1]
        numerical_features, cat_features = preprocess(training_set.iloc[:, :-1])
        if self.algorithm in {"ibl1", "ibl2", "ibl3"}:
            if self.algorithm is "ibl1":
                start = time.time()
                self._ibl1(numerical_features, cat_features, labels)
                end = time.time()
                self.execution_time = end - start
            elif self.algorithm is "ibl2":
                start = time.time()
                self._ibl2(numerical_features, cat_features, labels)
                end = time.time()
                self.execution_time = end - start
            elif self.algorithm is "ibl3":
                start = time.time()
                self._ibl3()
                end = time.time()
                self.execution_time = end - start

            self.accuracy = self.correct_samples / self.number_samples
        else:
            raise ValueError("You selected a wrong Instance-Based Learning algorithm")

    def ib1Algorithm(self, test_data):
        # TODO: reset accuracy before prediction to avoid counting train samples?
        # self._reset_evaluation_metrics()
        self.number_samples += test_data.shape[0]
        numerical_features, cat_features = preprocess(test_data.iloc[:, :-1])
        start = time.time()
        labels = self._ibl1_predict(numerical_features, cat_features)
        self.execution_time = time.time() - start
        # self.accuracy = self.correct_samples / self.number_samples

        return labels

    def ib2Algorithm(self, test_data):
        # TODO: reset accuracy before prediction to avoid counting train samples?
        # self._reset_evaluation_metrics()
        self.number_samples += test_data.shape[0]
        numerical_features, cat_features = preprocess(test_data.iloc[:, :-1])
        start = time.time()
        labels = self._ibl2_predict(numerical_features, cat_features)
        self.execution_time = time.time() - start
        # self.accuracy = self.correct_samples / self.number_samples

        return labels

    def print_results(self):
        print(f"Performance metrics of the {self.algorithm} algorithm")
        print(f"Number of saved instances: {self.saved_samples}")
        print(f"Execution time: {self.execution_time} seconds")
        print(f"Accuracy: {self.accuracy}%")
