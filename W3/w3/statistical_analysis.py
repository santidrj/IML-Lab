import os
import sys

sys.path.append("..")

import numpy as np

from w3.models.ibl_eval import IBLEval
from w3.algorithms import k_ibl_utils

data_root_path = os.path.join('..', 'datasets', 'datasets')
results_path = os.path.join('..', 'results')
dataset = 'vowel'

file = os.path.join(results_path, f'{dataset}-results.txt')
with open(file, 'r') as f:
    lines = f.readlines()
    f.close()

# Get accuracy per fold from the results file. We omit the first ones since they belong to the ib1, ib2 and ib3.
folds_acc = np.array([list(line[line.find('['):-1].strip('[]').split(', ')) for line in lines if
                      line.startswith('Accuracy per fold')][3:])
ibl_eval = IBLEval(os.path.join(data_root_path, dataset))
ibl_eval.ff, ibl_eval.crit_val, ibl_eval.which_diff, ibl_eval.crit_dist = k_ibl_utils.friedman_nemenyi(
    folds_acc.astype(float), alpha=0.1)

alg_acc = np.array([line.split(':')[1].strip(' \n') for line in lines if line.startswith('Mean accuracy')][3:])
# Indexes of the ten best algorithms
best_alg_idxs = np.argsort(alg_acc)[::-1][0:10]
ibl_eval.stat_diff_best = k_ibl_utils.readable_diff(ibl_eval.which_diff, best_alg_idxs)
# ibl_eval.write_statistical_analysis(file, 'k-ibl')
ibl_eval.print_statistical_analysis('k-ibl')
