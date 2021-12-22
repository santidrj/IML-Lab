import os

from models.ibl_eval import IBLEval

data_root_path = os.path.join('..', 'datasets', 'datasets')
# ibl_eval = IBLEval(os.path.join(data_root_path, 'pen-based'))
ibl_eval = IBLEval(os.path.join(data_root_path, 'adult'))
ibl_eval.run(algorithms=['ibl1', 'ibl2'])
# ibl_eval.write_results(os.path.join('..', 'results', 'pen-based-results.txt'), algorithms=['ibl1', 'ibl2'])
ibl_eval.write_results(os.path.join('..', 'results', 'adult-results.txt'), algorithms=['ibl1', 'ibl2'])
ibl_eval.print_results(algorithms=['ibl1', 'ibl2'])
# accuracy, time = ibl_eval.feed_data('autos.fold.000000.train.arff', 'autos.fold.000000.test.arff')
# print(accuracy)
# print(time)