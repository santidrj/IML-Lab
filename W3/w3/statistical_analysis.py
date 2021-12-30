import os

import numpy as np

from algorithms import k_ibl_utils
from models.ibl_eval import IBLEval


def extract_accuracies(lines, offset=0):
    folds_acc = np.array([list(line[line.find('['):-1].strip('[]').split(', ')) for line in lines if
                          line.startswith('Accuracy per fold')][offset:])
    alg_acc = np.array([line.split(':')[1].strip(' \n') for line in lines if line.startswith('Mean accuracy')][offset:])
    return folds_acc.astype(float), alg_acc.astype(float)


data_root_path = os.path.join('..', 'datasets', 'datasets')
results_path = os.path.join('..', 'results')

# Select the dataset to which perform the statistical analysis.
# dataset = 'grid'
dataset = 'vowel'

kibl_results_file = os.path.join(results_path, f'{dataset}-results.txt')
with open(kibl_results_file, 'r') as f:
    lines = f.readlines()
    f.close()

# Get accuracy per fold from the results file. We omit the three first ones since they belong to the ib1, ib2 and ib3.
# If you have these results in a different file, please remove the offset parameter.
kibl_folds_acc, kibl_alg_acc = extract_accuracies(lines, 3)

# Perform a statistical analysis between the different K-IBL configurations.
ibl_eval = IBLEval(os.path.join(data_root_path, dataset))
ibl_eval.ff, ibl_eval.crit_val, ibl_eval.which_diff, ibl_eval.crit_dist = k_ibl_utils.friedman_nemenyi(
    kibl_folds_acc, alpha=0.1)

# Indexes of the ten best algorithms
best_alg_idxs = np.argsort(kibl_alg_acc)[::-1][0:10]

configs = np.array([line[line.find('k'):].strip('\n') for line in lines if line.startswith('Configuration')])
print(f'Best 10 K-IBL configurations:\n{dict(zip(configs[best_alg_idxs], kibl_alg_acc[best_alg_idxs]))}\n')

if type(ibl_eval.which_diff) == np.ndarray:
    ibl_eval.stat_diff_best = k_ibl_utils.readable_diff(ibl_eval.which_diff, best_alg_idxs)

# You can use the line below to to run the algorithms.
# The second function call will save the results in the output file.
ibl_eval.print_statistical_analysis('k-ibl')
# ibl_eval.write_statistical_analysis(os.path.join(results_path, f'{dataset}-kibl-analysis.txt'), 'k-ibl')

# Perform a statistical analysis with the best K-IBL and its corresponding Selection K-IBL algorithms
select_kibl_results_file = os.path.join(results_path, f'{dataset}-select-kibl-results.txt')
with open(select_kibl_results_file, 'r') as f:
    lines = f.readlines()
    f.close()

sel_kibl_folds_acc, sel_kibl_alg_acc = extract_accuracies(lines)
ibl_eval = IBLEval(os.path.join(data_root_path, dataset))
ibl_eval.ff, ibl_eval.crit_val, ibl_eval.which_diff, ibl_eval.crit_dist = k_ibl_utils.friedman_nemenyi(
    np.concatenate((kibl_folds_acc[best_alg_idxs[0]].reshape(1, kibl_folds_acc.shape[1]), sel_kibl_folds_acc)),
    alpha=0.1)

if type(ibl_eval.which_diff) == np.ndarray:
    ibl_eval.stat_diff_best = k_ibl_utils.readable_diff(ibl_eval.which_diff, best_alg_idxs)

# You can use the line below to to run the algorithms.
# The second function call will save the results in the output file.
ibl_eval.print_statistical_analysis('selection-k-ibl')
# ibl_eval.write_statistical_analysis(os.path.join(results_path, f'{dataset}-selection-kibl-analysis.txt'), 'k-ibl')
