from numpy import unique
from pandas import read_csv
import numpy as np
import os 
def load_creditcard():
    dir = os.path.dirname (os.path.abspath(__file__))
    # load the dataset
    dataframe = read_csv(dir + '/creditcard.csv')
    # get the values
    values = dataframe.values
    X, y = values[:, :-1], values[:, -1]
    # gather details
    n_rows = X.shape[0]
    n_cols = X.shape[1]
    classes = unique(y)
    n_classes = len(classes)
    # summarize
    print('N Examples: %d' % n_rows)
    print('N Inputs: %d' % n_cols)
    print('N Classes: %d' % n_classes)
    print('Classes: %s' % classes)
    print('Class Breakdown:')
    # class breakdown
    breakdown = ''
    for c in classes:
        total = len(y[y == c])
        ratio = (total / float(len(y))) * 100
        print(' - Class %s: %d (%.5f%%)' % (str(c), total, ratio))
    return np.array(X),np.array(y) 