import os

from models.ibl_eval import IBLEval

data_root_path = os.path.join('..', 'datasets', 'datasets')
results_path = os.path.join('..', 'results')

dataset = 'sick'
ibl_eval = IBLEval(os.path.join(data_root_path, dataset))
algorithms = ['ibl1', 'ibl2', 'k-ibl']
# algorithms = ['k-ibl']
ibl_eval.run(algorithms=algorithms, output_file=os.path.join(results_path, f'{dataset}-results.txt'))
# ibl_eval.write_results(os.path.join(results_path, f'{dataset}-results.txt'), algorithms=algorithms)
ibl_eval.print_results(algorithms=algorithms)
