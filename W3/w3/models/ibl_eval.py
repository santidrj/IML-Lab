import os
import sys

sys.path.append('..')

from w3.algorithms import k_ibl_utils
from w3.algorithms.ibl import IBL
from w3 import utils

import numpy as np


def set_default_params(algorithms, ks, measures, policies, selection_methods):
    if algorithms is None:
        algorithms = ['ibl1', 'ibl2', 'ibl3', 'k-ibl', 'selection-k-ibl']
    if measures is None:
        measures = ['euclidean', 'manhattan', 'canberra', 'hvdm']
    if ks is None:
        ks = [3, 5, 7]
    if policies is None:
        policies = ['most_voted', 'mod_plurality', 'borda_count']
    if selection_methods is None:
        selection_methods = ['kbest', 'variance']
    return algorithms, ks, measures, policies, selection_methods


class IBLEval:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.acc_fold = dict(ibl1=[], ibl2=[], ibl3=[])
        self.acc_mean = dict(ibl1=[], ibl2=[], ibl3=[])
        self.time_fold = dict(ibl1=[], ibl2=[], ibl3=[])
        self.time_mean = dict(ibl1=[], ibl2=[], ibl3=[])
        self.kibl_acc = []
        self.ff = None
        self.crit_val = None
        self.which_diff = None
        self.stat_diff_best = None
        self.crit_dist = None

    def feed_data(self, train_file_name, test_file_name, algorithm='ibl1', k=3, policy='most_voted',
                  measure='euclidean', selection_method='kbest'):
        train_data = utils.load_arff(os.path.join(self.dataset_path, train_file_name))
        test_data = utils.load_arff(os.path.join(self.dataset_path, test_file_name))
        utils.convert_byte_string_to_string(train_data)
        utils.convert_byte_string_to_string(test_data)

        if algorithm == 'k-ibl':
            ibl = IBL(train_data, algorithm, k, measure, policy)
            ibl.kIBLAlgorithm(test_data, k, measure, policy)
        elif algorithm == 'selection-k-ibl':
            ibl = IBL(train_data, algorithm, k, measure, policy, selection_method)
            ibl.selectionkIBLAlgorithm(test_data, k, measure, policy)
        else:
            ibl = IBL(train_data, algorithm)
            if algorithm == 'ibl1':
                ibl.ib1Algorithm(test_data)
            elif algorithm == 'ibl2':
                ibl.ib2Algorithm(test_data)
            elif algorithm == 'ibl3':
                ibl.ib3Algorithm(test_data)
            else:
                raise ValueError('The selected algorithm is not supported.')

        return ibl.accuracy, ibl.execution_time

    def feed_folds(self, algorithm='ibl1', config='ibl1', k=None, measure=None, policy=None, selection_method='kbest'):
        file_names = sorted(os.listdir(self.dataset_path))

        if algorithm in ['k-ibl', 'selection-k-ibl']:
            self.acc_fold[config] = []
            self.acc_mean[config] = []
            self.time_fold[config] = []
            self.time_mean[config] = []
            aux_acc = []

        for fold in range(0, len(file_names), 2):
            print(f'Feeding fold number {int(fold // 2)}')
            if algorithm in ['k-ibl', 'selection-k-ibl']:
                acc, time = self.feed_data(file_names[fold + 1], file_names[fold], algorithm, k=k, policy=policy,
                                           measure=measure, selection_method=selection_method)
                aux_acc.append(acc)
            else:
                acc, time = self.feed_data(file_names[fold + 1], file_names[fold], algorithm)

            self.acc_fold[config].append(acc)
            self.time_fold[config].append(time)

        if algorithm == 'k-ibl':
            self.kibl_acc.append(aux_acc)
        self.acc_mean[config] = np.mean(self.acc_fold[config])
        self.time_mean[config] = np.mean(self.time_fold[config])

    def run(self, algorithms=None, ks=None, measures=None, policies=None, selection_methods=None, output_file=None):
        algorithms, ks, measures, policies, selection_methods = set_default_params(algorithms, ks, measures, policies,
                                                                                   selection_methods)

        for alg in algorithms:
            if alg == 'k-ibl':
                for measure in measures:
                    for k in ks:
                        for policy in policies:
                            config = f'kibl-{k}-{measure}-{policy}'
                            self.feed_folds(alg, config=config, k=k, measure=measure, policy=policy,
                                            selection_method=selection_methods)
                            if output_file is not None:
                                self.write_result(output_file, alg, config)

                self.ff, self.crit_val, self.which_diff, self.crit_dist = k_ibl_utils.friedman_nemenyi(
                    np.array(self.kibl_acc))
                if output_file is not None:
                    self.write_statistical_analysis(output_file, alg)

            elif alg == 'selection-k-ibl':
                for sel in selection_methods:
                    k = ks[0]
                    measure = measures[0]
                    policy = policies[0]
                    config = f'selection-kibl-{k}-{measure}-{policy}-{sel}'
                    self.feed_folds(alg, config=config, k=k, measure=measure, policy=policy,
                                    selection_method=sel)
                    if output_file is not None:
                        self.write_result(output_file, alg, config)

            else:
                self.feed_folds(alg, config=alg, selection_method=selection_methods)
                if output_file is not None:
                    self.write_result(output_file, alg, alg)

    def write_result(self, file, algorithm, config):
        with open(file, 'a') as f:
            f.write('Dataset: {}'.format(self.dataset_path.rsplit(os.path.sep, 1)[-1]))
            f.write(f'\n--{algorithm.upper()} results--\n')
            if algorithm == 'k-ibl':
                params = config.split('-')
                f.write(f'Configuration: k={params[1]}, measure={params[2]}, policy={params[3]}\n')
            elif algorithm == 'selection-k-ibl':
                params = config.split('-')
                f.write(
                    f'Configuration: k={params[2]}, measure={params[3]}, policy={params[4]}, selection method={params[5]}\n')
            f.write('Accuracy per fold: {}\n'.format(self.acc_fold[config]))
            f.write('Mean accuracy: {}\n'.format(self.acc_mean[config]))
            f.write('Execution time per fold: {}\n'.format(self.time_fold[config]))
            f.write('Mean execution time: {}\n'.format(self.time_mean[config]))
            f.write('\n')
            f.write('-' * 120)
            f.write('\n\n')

    def write_statistical_analysis(self, file, algorithm):
        with open(file, 'a') as f:
            f.write('Dataset: {}\n'.format(self.dataset_path.rsplit(os.path.sep, 1)[-1]))
            f.write(f'--{algorithm.upper()} statistical analysis results--\n')
            f.write(f'FF value: {self.ff}\n')
            f.write(f'Critical values: {self.crit_val}\n')
            f.write(f'stat_diff_best=\n{self.stat_diff_best}\n')
            f.write(f'crit_dist={self.crit_dist}\n')

    def print_statistical_analysis(self, algorithm):
        print('Dataset: {}\n'.format(self.dataset_path.rsplit(os.path.sep, 1)[-1]))
        print(f'--{algorithm.upper()} statistical analysis results--\n')
        print(f'FF value: {self.ff}\n')
        print(f'Critical values: {self.crit_val}\n')
        # print(f'which_diff=\n{self.which_diff}\n')
        print(f'stat_diff_best=\n{self.stat_diff_best}\n')
        print(f'crit_dist={self.crit_dist}\n')

    def write_results(self, file, algorithms=None, ks=None, measures=None, policies=None, selection_methods=None):
        algorithms, ks, measures, policies, selection_methods = set_default_params(algorithms, ks, measures, policies,
                                                                                   selection_methods)

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
                for k in ks:
                    for measure in measures:
                        for policy in policies:
                            f.write('\n--K-IBL results--\n')
                            f.write(f'Configuration: k={k}, measure={measure}, policy={policy}\n')
                            f.write('Accuracy per fold: {}\n'.format(self.acc_fold[f'kibl-{k}-{measure}-{policy}']))
                            f.write('Mean accuracy: {}\n'.format(self.acc_mean[f'kibl-{k}-{measure}-{policy}']))
                            f.write(
                                'Execution time per fold: {}\n'.format(self.time_fold[f'kibl-{k}-{measure}-{policy}']))
                            f.write('Mean execution time: {}\n'.format(self.time_mean[f'kibl-{k}-{measure}-{policy}']))
            if 'selection-k-ibl' in algorithms:
                f.write('\n--Selection K-IBL results--\n')
                for sel in selection_methods:
                    k = ks[0]
                    measure = measures[0]
                    policy = policies[0]
                    f.write(f'Configuration: k={k}, measure={measure}, policy={policy}, selection method={sel}')
                    f.write(
                        'Accuracy per fold: {}\n'.format(self.acc_fold[f'selection-kibl-{k}-{measure}-{policy}-{sel}']))
                    f.write('Mean accuracy: {}\n'.format(self.acc_mean[f'selection-kibl-{k}-{measure}-{policy}-{sel}']))
                    f.write('Execution time per fold: {}\n'.format(
                        self.time_fold[f'selection-kibl-{k}-{measure}-{policy}-{sel}']))
                    f.write(
                        'Mean execution time: {}\n'.format(
                            self.time_mean[f'selection-kibl-{k}-{measure}-{policy}-{sel}']))
            f.write('\n')
            f.write('-' * 120)
            f.write('\n\n')

    def print_results(self, algorithms=None, ks=None, measures=None, policies=None, selection_methods=None):
        algorithms, ks, measures, policies, selection_methods = set_default_params(algorithms, ks, measures, policies,
                                                                                   selection_methods)

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
            print('\n--K-IBL results--\n')
            for k in ks:
                for measure in measures:
                    for policy in policies:
                        print(f'Configuration: k={k}, measure={measure}, policy={policy}')
                        print('Accuracy per fold: {}\n'.format(self.acc_fold[f'kibl-{k}-{measure}-{policy}']))
                        print('Mean accuracy: {}\n'.format(self.acc_mean[f'kibl-{k}-{measure}-{policy}']))
                        print('Execution time per fold: {}\n'.format(self.time_fold[f'kibl-{k}-{measure}-{policy}']))
                        print('Mean execution time: {}\n'.format(self.time_mean[f'kibl-{k}-{measure}-{policy}']))
                        print()
        if 'selection-k-ibl' in algorithms:
            print('\n--Selection K-IBL results--\n')
            for sel in selection_methods:
                k = ks[0]
                measure = measures[0]
                policy = policies[0]
                print(f'Configuration: k={k}, measure={measure}, policy={policy}, selection method={sel}')
                print('Accuracy per fold: {}\n'.format(self.acc_fold[f'selection-kibl-{k}-{measure}-{policy}-{sel}']))
                print('Mean accuracy: {}\n'.format(self.acc_mean[f'selection-kibl-{k}-{measure}-{policy}-{sel}']))
                print('Execution time per fold: {}\n'.format(
                    self.time_fold[f'selection-kibl-{k}-{measure}-{policy}-{sel}']))
                print(
                    'Mean execution time: {}\n'.format(self.time_mean[f'selection-kibl-{k}-{measure}-{policy}-{sel}']))
                print()

        if self.ff is not None:
            print('--K-IBL statistical analysis results--')
            print(f'FF value: {self.ff}')
            print(f'Critical values: {self.crit_val}')
            print(f'which_diff={self.which_diff}')
            print(f'crit_dist={self.crit_dist}')
