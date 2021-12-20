import os
import sys

sys.path.append('..')

from algorithms.ibl import IBL
import utils

import numpy as np


class IBLEval:
    def __init__(self, dataset_path):
        self.acc_fold = dict(ibl1=[], ibl2=[], ibl3=[])
        self.acc_mean = dict(ibl1=[], ibl2=[], ibl3=[])
        self.time_fold = dict(ibl1=[], ibl2=[], ibl3=[])
        self.time_mean = dict(ibl1=[], ibl2=[], ibl3=[])
        self.dataset_path = dataset_path

    def feed_data(self, train_file_name, test_file_name, algorithm='ibl1'):
        train_data = utils.load_arff(os.path.join(self.dataset_path, train_file_name))
        test_data = utils.load_arff(os.path.join(self.dataset_path, test_file_name))

        utils.convert_byte_string_to_string(train_data)
        utils.convert_byte_string_to_string(test_data)

        ibl = IBL(train_data, algorithm)
        if algorithm == 'ibl1':
            pred_labels = ibl.ib1Algorithm(test_data)
        elif algorithm == 'ibl2':
            pred_labels = ibl.ib2Algorithm(test_data)

        return ibl.accuracy, ibl.execution_time

    def feed_folds(self, algorithm='ibl1'):
        file_names = sorted(os.listdir(self.dataset_path))

        for fold in range(0, len(file_names), 2):
            acc, time = self.feed_data(file_names[fold + 1], file_names[fold], algorithm)
            self.acc_fold[algorithm].append(acc)
            self.time_fold[algorithm].append(time)

        self.acc_mean[algorithm] = np.mean(self.acc_fold[algorithm])
        self.time_mean[algorithm] = np.mean(self.time_fold[algorithm])

    def run(self, algorithms=None):
        if algorithms is None:
            algorithms = ['ibl1', 'ibl2', 'ibl3']
        for alg in algorithms:
            self.feed_folds(alg)

    def write_results(self, file, algorithms=None):
        if algorithms is None:
            algorithms = ['ibl1', 'ibl2', 'ibl3']

        with open(file, 'a') as f:
            f.write('Dataset: {}'.format(self.dataset_path.rsplit('/', 1)[-1]))
            if 'ibl1' in algorithms:
                f.write('\n--IBL1 results--')
                f.write('Accuracy per fold: {}'.format(self.acc_fold['ibl1']))
                f.write('Mean accuracy: {}'.format(self.acc_mean['ibl1']))
                f.write('Execution time per fold: {}'.format(self.time_fold['ibl1']))
                f.write('Mean execution time: {}'.format(self.time_mean['ibl1']))
            if 'ibl2' in algorithms:
                f.write('\n--IBL2 results--')
                f.write('Accuracy per fold: {}'.format(self.acc_fold['ibl2']))
                f.write('Mean accuracy: {}'.format(self.acc_mean['ibl2']))
                f.write('Execution time per fold: {}'.format(self.time_fold['ibl2']))
                f.write('Mean execution time: {}'.format(self.time_mean['ibl2']))
            if 'ibl3' in algorithms:
                f.write('\n--IBL3 results--')
                f.write('Accuracy per fold: {}'.format(self.acc_fold['ibl3']))
                f.write('Mean accuracy: {}'.format(self.acc_mean['ibl3']))
                f.write('Execution time per fold: {}'.format(self.time_fold['ibl3']))
                f.write('Mean execution time: {}'.format(self.time_mean['ibl3']))
            f.write('\n\n')

    def print_results(self, algorithms=None):
        if algorithms is None:
            algorithms = ['ibl1', 'ibl2', 'ibl3']

        print('Dataset: {}'.format(self.dataset_path.rsplit('/', 1)[-1]))
        if 'ibl1' in algorithms:
            print('\n--IBL1 results--')
            print('Accuracy per fold: {}'.format(self.acc_fold['ibl1']))
            print('Mean accuracy: {}'.format(self.acc_mean['ibl1']))
            print('Execution time per fold: {}'.format(self.time_fold['ibl1']))
            print('Mean execution time: {}'.format(self.time_mean['ibl1']))
        if 'ibl2' in algorithms:
            print('\n--IBL2 results--')
            print('Accuracy per fold: {}'.format(self.acc_fold['ibl2']))
            print('Mean accuracy: {}'.format(self.acc_mean['ibl2']))
            print('Execution time per fold: {}'.format(self.time_fold['ibl2']))
            print('Mean execution time: {}'.format(self.time_mean['ibl2']))
        if 'ibl3' in algorithms:
            print('\n--IBL3 results--')
            print('Accuracy per fold: {}'.format(self.acc_fold['ibl3']))
            print('Mean accuracy: {}'.format(self.acc_mean['ibl3']))
            print('Execution time per fold: {}'.format(self.time_fold['ibl3']))
            print('Mean execution time: {}'.format(self.time_mean['ibl3']))
