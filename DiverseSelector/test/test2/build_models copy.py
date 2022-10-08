import argparse
import cProfile
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from joblib import dump
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.metrics import pairwise_distances
import openpyxl
from sklearn.model_selection import KFold, cross_val_score

from DiverseSelector.selectors import MaxMin, MaxSum, OptiSim, DirectedSphereExclusion, GridPartitioning, KDTree


def print_metrics(loss, best_param, y_train, y_test,
                  y_train_predict, y_test_predict, start, num_eval, counter=1):
    # Create or append logfile
    with open('{}.log'.format(args.output), 'a') as fn:
        fn.write('Outer-CV #: {}\n'.format(counter))
        fn.write('Best parameters: {}\n'.format(best_param))
        fn.write(classification_report(y_test, y_test_predict))
        fn.write('\n')

    # Generate XLSX row
    report_row = {
        'CV_LOOP': counter,
        'Best_parameters': best_param,
        'Train_roc_auc': min(loss) * -1,
        'Test_roc_auc': roc_auc_score(y_test, y_test_predict),
        'Train_acc': accuracy_score(y_train, y_train_predict),
        'Test_acc': accuracy_score(y_test, y_test_predict),
        'Time_elapsed': time.time() - start,
        'Num_evals': num_eval}
    print(report_row)

    return report_row

def get_array(string):
    return np.array([int(elem) for elem in string])


func = pairwise_distances
selectors = [MaxMin(func_distance=func),
             MaxSum(func_distance=func),
             OptiSim(),
             DirectedSphereExclusion(),
             GridPartitioning(cells=10),
             KDTree()]

data_1024_all = pd.read_excel('BBB_SECFP6_1024.xlsx')
data_2048_all = pd.read_excel('BBB_SECFP6_2048.xlsx')
data_1024 = np.vstack(pd.read_excel('BBB_SECFP6_1024.xlsx').fingerprint.apply(get_array).values)
data_2048 = np.vstack(pd.read_excel('BBB_SECFP6_2048.xlsx').fingerprint.apply(get_array).values)

print(data_1024.shape, data_2048.shape)

num_selected = 100

tester = data_1024[:, :]
print(tester.shape)

cProfile.run(f"MaxMin(lambda x: pairwise_distances(x, metric='euclidean')).select(arr=tester, num_selected={num_selected})")
cProfile.run(f"DirectedSphereExclusion(tolerance=10000000000000, r=0.065).select(arr=tester, num_selected={num_selected})")
cProfile.run(f"KDTree().select(arr=tester, num_selected={num_selected})")
cProfile.run(f"OptiSim(tolerance=100000000000000, r=0.05).select(arr=tester, num_selected={num_selected})")

