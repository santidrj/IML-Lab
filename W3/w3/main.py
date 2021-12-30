import os

from models.ibl_eval import IBLEval

data_root_path = os.path.join('..', 'datasets', 'datasets')
results_path = os.path.join('..', 'results')

# TODO: REMOVE THIS FILE
dataset = 'vowel'
ibl_eval = IBLEval(os.path.join(data_root_path, dataset))
# algorithms = ['ibl1', 'ibl2', 'ibl3', 'k-ibl']
algorithms = ['selection-k-ibl']
# K = [3, 5, 7]
K = [3]
# measures = ['euclidean', 'manhattan', 'canberra', 'hvdm']
# measures = ['canberra', 'hvdm']
# measures = ['hvdm']
measures = ['euclidean']
# measures = ['euclidean', 'manhattan', 'canberra']
# policies = ['most_voted', 'mod_plurality', 'borda_count']
policies = ['mod_plurality']
ibl_eval.run(algorithms=algorithms, ks=K, measures=measures, policies=policies,
             output_file=os.path.join(results_path, f'{dataset}-results-sel.txt'))
ibl_eval.print_results(algorithms, K, measures, policies)
