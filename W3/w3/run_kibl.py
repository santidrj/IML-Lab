import os

from models.ibl_eval import IBLEval

data_root_path = os.path.join('..', 'datasets', 'datasets')
results_path = os.path.join('..', 'results')

datasets = ['grid', 'vowel']
for dataset in datasets:
    ibl_eval = IBLEval(os.path.join(data_root_path, dataset))
    algorithms = ['k-ibl']
    K = [3, 5, 7]
    measures = ['euclidean', 'manhattan', 'canberra', 'hvdm']
    policies = ['most_voted', 'mod_plurality', 'borda_count']
    # Choose one of the invocations below to run the algorithm.
    # In the second one the results are written to the specified file.
    ibl_eval.run(algorithms=algorithms, ks=K, measures=measures, policies=policies)
    # ibl_eval.run(algorithms=algorithms, ks=K, measures=measures, policies=policies,
    #              output_file=os.path.join(results_path, f'{dataset}-results.txt'))

    # Comment out the line below to show the results in the console
    ibl_eval.print_results(algorithms, K, measures, policies)

    # Select one of the invocations below to see the statistical analysis
    ibl_eval.print_statistical_analysis('k-ibl')
    # ibl_eval.write_statistical_analysis(os.path.join(results_path, f'{dataset}-kibl-analysis.txt'), 'k-ibl')
