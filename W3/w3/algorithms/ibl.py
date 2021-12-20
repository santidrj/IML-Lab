import time

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler


def num_distance(x, y):
    """
    x and y should be numpy arrays
    parameters:
      x and y: sample of n dimension
    return:
      the similarity distance between x and y
    """
    return np.sqrt(np.square(x - y).sum())


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
    normalized_num_features = RobustScaler(unit_variance=True).fit_transform(
        numeric_features
    )

    categorical_features = data.select_dtypes(include="object")

    return normalized_num_features, categorical_features


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


def cat_distance(x, y):
    dist = 0
    for i in range(len(x)):
        if x is None or y is None:
            dist += 1
        else:
            dist += (x[i] != y[i])
    return dist


def get_class(distance_list, method='nn'):
    """
    Compute the class of a sample using its distance to the points in the CD.

    :param distance_list: List with the distances to the points in the CD and their classes.
    :param method: Method used to obtain the class. By default, it uses Nearest Neighbors.
    :return: The class of the sample.
    """

    # TODO: implement voting system
    if method == 'nn':
        return distance_list[distance_list[:, 0].argmin(), 1]
    return 0


class IBL:
    """
    Instance Based Learning algorithm.

    Attributes

    - dataframe: Pandas DataFrame where the last column is the class value of each row.
    - algorithm: String which determine the IBL to execute. If the value is not pass, IBL1 is selected.
    """

    def __init__(self, dataframe: DataFrame, algorithm="ibl1"):
        self.training_set = dataframe
        self.number_samples = len(dataframe)
        self.algorithm = algorithm
        self.correct_samples = 0
        self.incorrect_samples = 0
        self.accuracy = 0
        self.saved_samples = 0
        self.execution_time = 0
        self.cd = set()

        self._run()
        self.print_results()

    def _ibl1(self, numerical_features, cat_features, labels):
        for i in range(self.number_samples):
            x_num = numerical_features[i]
            x_cat = cat_features[i]
            label = labels[i]

            if not self.cd:
                self.cd.add((x_num, x_cat, label))
            else:
                # Obtain a list with the sample distance to each point in the CD and save it together with the point
                # class
                distance_list = [np.concatenate(num_distance(x_num, y[0]) + cat_distance(x_cat, y[1]), y[2]) for y
                                 in self.cd]
                y_max = np.argmin(distance_list[:, 0])
                if label == distance_list[y_max][1]:
                    self.correct_samples += 1
                else:
                    self.incorrect_samples += 1
            self.cd.add((x_num, x_cat, label))

    def _ibl1_predict(self, numerical_features, cat_features):
        labels = []
        for i in range(numerical_features.shape[0]):
            x_num = numerical_features[i]
            x_cat = cat_features[i]

            # Obtain a list with the sample distance to each point in the CD and save it together with the point class
            distance_list = [np.concatenate(num_distance(x_num, y[0]) + cat_distance(x_num, y[1]), y[2]) for y in
                             self.cd]
            most_similar = np.argmin(distance_list[:, 0])
            label = get_class(distance_list)
            if label == distance_list[most_similar][1]:
                self.correct_samples += 1
            else:
                self.incorrect_samples += 1

            labels.append(label)
            self.cd.add((x_num, x_cat, label))

        return labels

    def _ibl2(self):
        for i in range(self.number_samples):
            x = self.training_set.iloc[i]
            if not self.cd:
                self.cd.add(x)
            else:
                similarity_list = [num_distance(x[:-1], y[:-1]) for y in self.cd]
                y_max = min(similarity_list)
                if x[-1] == y_max[-1]:
                    self.correct_samples += 1
                else:
                    self.incorrect_samples += 1
                    self.cd.add(x)

    def _ibl3(self):
        for i in range(self.number_samples):
            x = self.training_set.iloc[i]

    def _run(self):
        # TODO: Implement below
        labels = self.training_set.iloc[:, -1]
        numerical_features, cat_features = preprocess(self.training_set)
        if self.algorithm in {"ibl1", "ibl2", "ibl3"}:
            if self.algorithm is "ibl1":
                start = time.time()
                self._ibl1(numerical_features, cat_features, labels)
                end = time.time()
                self.execution_time = end - start
            elif self.algorithm is "ibl2":
                start = time.time()
                self._ibl2()
                end = time.time()
                self.execution_time = end - start
            elif self.algorithm is "ibl3":
                start = time.time()
                self._ibl3()
                end = time.time()
                self.execution_time = end - start

            self.accuracy = self.correct_samples / self.number_samples
        else:
            print("You selected a wrong Instance-Based Learning algorithm")

    def ib1Algorithm(self, test_data):
        # TODO: reset accuracy before prediction to avoid counting train samples?
        numerical_features, cat_features = preprocess(test_data)
        start = time.time()
        labels = self._ibl1_predict(numerical_features, cat_features)
        self.execution_time = time.time() - start

        return labels

    def print_results(self):
        print(f"Performance metrics of the {self.algorithm} algorithm")
        print(f"Number of saved instances: {self.saved_samples}")
        print(f"Execution time: {self.execution_time}")
        print(f"Accuracy: {self.accuracy}%")
