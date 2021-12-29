import os

from models.ibl_eval import IBLEval

data_root_path = os.path.join('..', 'datasets', 'datasets')
results_path = os.path.join('..', 'results')

# datasets = ['grid', 'vowel']
datasets = ['vowel']
parameters = [
    # dict(k=[7], measure=['euclidean'], policy=['most_voted']),
    dict(k=[3], measure=['euclidean'], policy=['mod_plurality'])
]
for i, dataset in enumerate(datasets):
    ibl_eval = IBLEval(os.path.join(data_root_path, dataset))
    algorithms = ['selection-k-ibl']
    K = parameters[i]['k']
    measures = parameters[i]['measure']
    policies = parameters[i]['policy']
    # selection_methods = ['kbest', 'variance']
    selection_methods = ['variance']
    # Choose one of the invocations below to run the algorithm.
    # In the second one the results are written to the specified file.
    # ibl_eval.run(algorithms=algorithms, ks=K, measures=measures, policies=policies, selection_methods=selection_methods)
    ibl_eval.run(algorithms=algorithms, ks=K, measures=measures, policies=policies, selection_methods=selection_methods,
                 output_file=os.path.join(results_path, f'{dataset}-select-kibl-results.txt'))

    # Comment out the line below to show the results in the console
    ibl_eval.print_results(algorithms, K, measures, policies, selection_methods)
