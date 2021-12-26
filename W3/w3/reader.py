import os

import numpy as np

from w3.models.ibl_eval import IBLEval
from w3.algorithms import k_ibl_utils

data_root_path = os.path.join('..', 'datasets', 'datasets')
results_path = os.path.join('..', 'results')
dataset = 'sick'

file = os.path.join(results_path, f'{dataset}-results.txt')
with open(file, 'r') as f:
    lines = f.readlines()

lines = np.array([list(line[line.find('['):-1].strip('[]').split(', ')) for line in lines if
                  line.startswith('Accuracy per fold')][2:])
ibl_eval = IBLEval(os.path.join(data_root_path, dataset))
ibl_eval.ff, ibl_eval.crit_val, ibl_eval.which_diff, ibl_eval.crit_dist = k_ibl_utils.friedman_nemenyi(lines)
ibl_eval.write_statistical_analysis(file, 'k-ibl')
