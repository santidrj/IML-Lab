import os

from w3.algorithms.ibl import IBL
from w3 import utils

import numpy as np

K = [3, 5, 7]
measures = ['euclidean', 'manhattan', 'canberra', 'hvdm']
policies = ['most_voted', 'mod_plurality', 'borda_count']


class IBLEval:
    def __init__(self, dataset_path):
        self.acc_fold = dict(ibl1=[], ibl2=[], ibl3=[])
        self.acc_mean = dict(ibl1=[], ibl2=[], ibl3=[])
        self.time_fold = dict(ibl1=[], ibl2=[], ibl3=[])
        self.time_mean = dict(ibl1=[], ibl2=[], ibl3=[])
        self.dataset_path = dataset_path

    def feed_data(self, train_file_name, test_file_name, algorithm='ibl1', k=3, policy='most_voted',
                  measure='euclidean'):
        train_data = utils.load_arff(os.path.join(self.dataset_path, train_file_name))
        test_data = utils.load_arff(os.path.join(self.dataset_path, test_file_name))

        utils.convert_byte_string_to_string(train_data)
        utils.convert_byte_string_to_string(test_data)

        if algorithm == 'k-ibl':
            ibl = IBL(train_data, algorithm, k, measure, policy)
            ibl.kIBLAlgorithm(test_data, k, measure, policy)
        else:
            ibl = IBL(train_data, algorithm)
            if algorithm == 'ibl1':
                ibl.ib1Algorithm(test_data)
            elif algorithm == 'ibl2':
                ibl.ib2Algorithm(test_data)

        return ibl.accuracy, ibl.execution_time

    def feed_folds(self, algorithm='ibl1', config='ibl1', k=None, measure=None, policy=None):
        file_names = sorted(os.listdir(self.dataset_path))

        if algorithm == 'k-ibl':
            self.acc_fold[config] = []
            self.acc_mean[config] = []
            self.time_fold[config] = []
            self.time_mean[config] = []

        for fold in range(0, len(file_names), 2):
            if algorithm == 'k-ibl':
                acc, time = self.feed_data(file_names[fold + 1], file_names[fold], algorithm, k=k, measure=measure,
                                           policy=policy)
            else:
                acc, time = self.feed_data(file_names[fold + 1], file_names[fold], algorithm)

            self.acc_fold[config].append(acc)
            self.time_fold[config].append(time)

        self.acc_mean[config] = np.mean(self.acc_fold[config])
        self.time_mean[config] = np.mean(self.time_fold[config])

    def run(self, algorithms=None):
        if algorithms is None:
            algorithms = ['ibl1', 'ibl2', 'ibl3', 'k-ibl']
        for alg in algorithms:
            if alg == 'k-ibl':
                for k in K:
                    for measure in measures:
                        for policy in policies:
                            self.feed_folds(alg, config=f'kibl-{k}-{measure}-{policy}', k=k, measure=measure,
                                            policy=policy)
            else:
                self.feed_folds(alg)

    def write_results(self, file, algorithms=None):
        if algorithms is None:
            algorithms = ['ibl1', 'ibl2', 'ibl3', 'k-ibl']

        with open(file, 'a') as f:
            f.write('Dataset: {}'.format(self.dataset_path.rsplit(os.path.sep, 1)[-1]))
            if 'ibl1' in algorithms:
                f.write('\n--IBL1 results--\n')
                f.write('Accuracy per fold: {}\n'.format(self.acc_fold['ibl1']))
                f.write('Mean accuracy: {}\n'.format(self.acc_mean['ibl1']))
                f.write('Execution time per fold: {}\n'.format(self.time_fold['ibl1']))
                f.write('Mean execution time: {}\n'.format(self.time_mean['ibl1']))
            if 'ibl2' in algorithms:
                f.write('\n--IBL2 results--\n')
                f.write('Accuracy per fold: {}\n'.format(self.acc_fold['ibl2']))
                f.write('Mean accuracy: {}\n'.format(self.acc_mean['ibl2']))
                f.write('Execution time per fold: {}\n'.format(self.time_fold['ibl2']))
                f.write('Mean execution time: {}\n'.format(self.time_mean['ibl2']))
            if 'ibl3' in algorithms:
                f.write('\n--IBL3 results--\n')
                f.write('Accuracy per fold: {}\n'.format(self.acc_fold['ibl3']))
                f.write('Mean accuracy: {}\n'.format(self.acc_mean['ibl3']))
                f.write('Execution time per fold: {}\n'.format(self.time_fold['ibl3']))
                f.write('Mean execution time: {}\n'.format(self.time_mean['ibl3']))
            if 'k-ibl' in algorithms:
                for k in K:
                    for measure in measures:
                        for policy in policies:
                            f.write('\n--K-IBL results--\n')
                            f.write(f'Configuration: k={k}, measure={measure}, policy={policy}\n')
                            f.write('Accuracy per fold: {}\n'.format(self.acc_fold[f'kibl-{k}-{measure}-{policy}']))
                            f.write('Mean accuracy: {}\n'.format(self.acc_mean[f'kibl-{k}-{measure}-{policy}']))
                            f.write(
                                'Execution time per fold: {}\n'.format(self.time_fold[f'kibl-{k}-{measure}-{policy}']))
                            f.write('Mean execution time: {}\n'.format(self.time_mean[f'kibl-{k}-{measure}-{policy}']))
            f.write('\n')
            f.write('-' * 120)
            f.write('\n')

    def print_results(self, algorithms=None):
        if algorithms is None:
            algorithms = ['ibl1', 'ibl2', 'ibl3', 'k-ibl']

        print('Dataset: {}'.format(self.dataset_path.rsplit(os.path.sep, 1)[-1]))
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
        if 'k-ibl' in algorithms:
            for k in K:
                for measure in measures:
                    for policy in policies:
                        print('\n--K-IBL results--\n')
                        print(f'Configuration: k={k}, measure={measure}, policy={policy}\n')
                        print('Accuracy per fold: {}\n'.format(self.acc_fold[f'kibl-{k}-{measure}-{policy}']))
                        print('Mean accuracy: {}\n'.format(self.acc_mean[f'kibl-{k}-{measure}-{policy}']))
                        print('Execution time per fold: {}\n'.format(self.time_fold[f'kibl-{k}-{measure}-{policy}']))
                        print('Mean execution time: {}\n'.format(self.time_mean[f'kibl-{k}-{measure}-{policy}']))
