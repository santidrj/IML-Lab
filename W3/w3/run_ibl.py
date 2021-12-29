import os

from models.ibl_eval import IBLEval

data_root_path = os.path.join('..', 'datasets', 'datasets')
results_path = os.path.join('..', 'results')

# Add or remove datasets at will. The name should be the name of the dataset folder.
datasets = ['grid', 'vowel']
for dataset in datasets:
    ibl_eval = IBLEval(os.path.join(data_root_path, dataset))
    # Add or remove the algorithms at will.
    algorithms = ['ibl1', 'ibl2', 'ibl3']
    ibl_eval.run(algorithms=algorithms)
    # You can use the line below to save the algorithms results to a file.
    # ibl_eval.run(algorithms=algorithms, output_file=os.path.join(results_path, 'output_file'))
    # Comment out the line below to show the results in the console
    # ibl_eval.print_results(algorithms)
