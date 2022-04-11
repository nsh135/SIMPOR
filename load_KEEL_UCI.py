from numpy import unique
from pandas import read_csv
import numpy as np
import os 
from  imblearn.datasets import fetch_datasets

def load_UCI(dataset):
    data = fetch_datasets( random_state = 13, verbose = True, data_home = 'DATA/')[dataset]
    for i,val in enumerate(data.target): 
        if val == 1 : data.target[i] = 0
        else : data.target[i] = 1
    # print(data.data, data.target)
    print(data.data.shape, data.target.shape)
    return np.array(data.data),np.array(data.target)

def load_KEEL(dataset):
    dir = os.path.dirname (os.path.abspath(__file__))  + "/DATA/"+"{}".format(dataset)
    # load the dataset
    trainFiles = []
    testFiles = []
    X = None
    y = None
    for filename in os.listdir(dir):
        print(filename)
        if "1tra.dat" in filename: trainFiles.append( filename)
        elif "1tst.dat" in filename: testFiles.append( filename)
    
    for  trainFile, testFile in zip(trainFiles, testFiles):
        trainframe = read_csv(dir + '/' + trainFile,header=None,sep=",", comment = '@' )
        testframe = read_csv(dir + '/' + testFile,header=None, sep=",",  comment = '@' )
        if 'abalone' in trainFile: 
            trainframe[0].replace(['M', 'F', 'I'],[0, 1, 2], inplace=True)
            testframe[0].replace(['M', 'F', 'I'],[0, 1, 2], inplace=True)

        nan = trainframe.isnull().sum().sum()
        if nan != 0: raise   Exception("Data contains NAN !!!!!!")
        # get the values
        X_train, y_train = trainframe.values[:, :-1], [0 if y.strip()=="positive" else 1 for y in trainframe.values[:, -1] ]
        X_test, y_test = testframe.values[:, :-1],  [0 if y.strip()=="positive" else 1 for y in testframe.values[:, -1]]
        X_tmp , y_tmp = np.concatenate( (X_train,X_test) ) , y_train+y_test
        print(X_train)
        if not (X is None): X, y = np.concatenate( (X,X_tmp) ) , y+y_tmp
        else: 
            X = X_tmp
            y = y_tmp
    # gather details
    n_rows = X.shape[0] 
    n_cols = X.shape[1]
    classes = unique(y)
    n_classes = len(classes) 
    if n_classes != 2: raise Exception("There are more than 2 classes!!!!!!")
    # summarize
    print('N Examples: %d' % n_rows)
    print('N Inputs: %d' % n_cols)
    print('N Classes: %d' % n_classes)
    print('Classes: %s' % classes)
    return np.array(X),np.array(y),  

#load_KEEL("glass1")
# load_UCI("yeast_me2")