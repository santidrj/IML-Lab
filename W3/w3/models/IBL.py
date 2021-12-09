import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff

def similarity(x, y):
    """
    x and y should be numpy arrays
    parameters:
      x and y: sample of n dimension
    return:
      the similarity distance between x and y
    """
    return np.sqrt(np.square(x - y).sum())


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
    dataframe: should be a Pandas Dataframe with the features normalize
        the last column is the class value of each row
    algorithm: string which determine the IBL to execute

        if the value is not pass, IBL1 is selected.
    """
    def __init__(self, dataframe, algorithm='ibl1'):
        self.training_set = dataframe
        self.number_samples = len(dataframe)
        self.algorithm = algorithm
        self.correct_samples = 0
        self.incorrect_samples = 0
        self.accuracy = None
        self.saved_samples = 0
        self.execution_time = 0
        self.cd = set()

        self._run()
        self.print_results()

    def _ibl1(self):
        for i in range(self.number_samples):
            x = self.training_set.iloc[i]
            if not self.cd:
                self.cd.add(x)
            else:
                similarity_list = [similarity(x[:-1], y[:-1]) for y in cd]
                y_max = min(similarity_list)
                if x[-1] == y_max[-1]:
                    self.correct_samples += 1
                else:
                    self.incorrect_samples += 1
            self.cd.add(x)

    def _ibl2(self):
        for i in range(self.number_samples):
            x = self.training_set.iloc[i]
            if not cd:
                self.cd.add(x)
            else:
                similarity_list = [similarity(x[:-1], y[:-1]) for y in cd]
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
        if self.algorihtm in {'ibl1', 'ibl2', 'ibl3'}:
            if self.algorithm is 'ibl1':
                start = time.time()
                self._ibl1()
                end = time.time()
                self.execution_time = end - start
            elif self.algorithm is 'ibl2':
                start = time.time()
                self._ibl2()
                end = time.time()
                self.execution_time = end - start
            elif self.algorithm is 'ibl3':
                start = time.time()
                self._ibl3()
                end = time.time()
                self.execution_time = end - start

            self.accuracy = str(self.correct_samples / self.number_samples) + '%'
        else:
            print("You selected a wrong Instance based learning algorithm")

    def print_results(self):
        print("Performance metrics of the", self.algorithm, " algorithm")
        print("Number of saved instances: ", self.saved_samples)
        print("Execution time: ", self.execution_time)
        print("Accuracy: " + self.accuracy)
