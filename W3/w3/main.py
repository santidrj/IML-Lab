import os

from models.ibl_eval import IBLEval

data_root_path = os.path.join('..', 'datasets', 'datasets')
ibl_eval = IBLEval(os.path.join(data_root_path, 'pen-based'))
ibl_eval.feed_data('pen-based.fold.000000.train.arff', 'pen-based.fold.000000.test.arff')
ibl_eval.print_results(algorithms=['ibl1'])
