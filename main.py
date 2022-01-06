

#  need tensorflow environment
import builtins as __builtin__
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets import make_gaussian_quantiles,make_classification
from sklearn.utils.multiclass import unique_labels
import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from sklearn.decomposition import PCA
import pandas as pd

import pickle
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  , Dropout
from simpor import max_FracPosterior_balancing

from sklearn.model_selection import train_test_split
from keras.datasets import mnist
import datetime

from DATA.diabetes.loaddata import load_diabetes
from DATA.creditcardfraud.load_data import load_creditcard

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)



def log_prepare(dataset_name):
    global log_dir, log_file, figure_path, result_dir, checkpoint

    log_dir = "./LOG/"+dataset_name+str(datetime.datetime.now()).replace(":",'.')
    log_file = os.path.join(log_dir,"log.txt")
    figure_path = os.path.join(log_dir,"Figures")
    result_dir = os.path.join(log_dir,"Results")
    checkpoint = os.path.join(log_dir,"Checkpoint")
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    if not os.path.isdir(figure_path): os.makedirs(figure_path)
    if not os.path.isdir(result_dir): os.makedirs(result_dir)
    if not os.path.isdir(checkpoint): os.makedirs(checkpoint)

def print(*args, **kwargs):
    """My custom print() function.
       Print out and also keep loging 
    """
    with open(log_file, "a") as log:  
        log.write("\n"+str(datetime.datetime.now())+": ")
        log.writelines(args)
    return __builtin__.print(*args, **kwargs)

def data_gen(dataset_name, IR):
    """
    Generate synthetic imbalanced dataset
    dataset_name: 'breast_cancer', 'moon', 'creditcard', 'mnist'
    IR: artificial Imabalance Ratio 
    -----
    Return 
    X_train, X_test, y_train, y_test
    """
    global k, k_neighbors, n_neuron,epoch , n_layers
    global X, labels

    if dataset_name == 'breast_cancer':
        dataset = sklearn.datasets.load_breast_cancer()  ##{1: 357, 0: 212}
        n_neuron = 10; 
        n_layers = 3
        epoch = 150
        X = dataset['data']
        labels = dataset['target']
        cls_need_removal = [0]
        rm_ratio = [1-(357/(IR*212))] # #removal percentage
        k =  1  # for maxFracPosterior, generate k neighbors for each found minima in informative set 
        k_neighbors = 10  #for SMOTE
    elif dataset_name == 'moon':
        n_samples = 3000 # for moon dataset
        X,labels  = sklearn.datasets.make_moons(n_samples=n_samples, noise=.2, random_state =RandomSeed)
        n_neuron = 200; 
        n_layers = 3
        epoch = 200
        cls_need_removal = [0]
        rm_ratio = [1-(1/IR)]## removal percentage
        k =  1  # for maxFracPosterior, generate k neighbors for each found minima in informative set 
        k_neighbors = 5 #for SMOTE

    elif dataset_name == 'creditcard':  #{0: 284314, 1: 492}
        ## this will take long
        X,labels  = load_creditcard()
        n_neuron = 300;
        n_layers = 5 
        epoch =400
        cls_need_removal = [0, 1]  # majority
        rm_ratio = [ 1-(IR*492/284314) ] #removal percentage 
        k =  1  # for maxFracPosterior, generate k neighbors for each found minima in informative set 
        k_neighbors = 20 #for SMOTE

    elif dataset_name == 'mnist':
        (X_train, y_train),(X_test,y_test) = mnist.load_data()
        ## take only 2 classes: 4 and 7 {0:5,000, 1: 5,000}
        X_train_47 = np.concatenate( (X_train[y_train == 4], X_train[y_train == 7]))
        y_train_47 = np.concatenate( (y_train[y_train == 4], y_train[y_train == 7]))
        X = X_train_47.reshape(X_train_47.shape[0],-1)
        labels = np.array([0 if y == 4 else 1 for y in y_train_47] )
        #params
        epoch =400
        n_layers = 3
        n_neuron = 400
        cls_need_removal = [0]  
        rm_ratio = [1-1/IR] # remove portion 
        k =  1
        k_neighbors = 5 #for SMOTE
        
    print("\r==================={}===================\n\n".format(dataset_name))
    print("Params \nRandomSeed : {}\n n_neuron : {}\n n_Layers : {}\n epoch: {} \n k : {} \n rm_ratio: {}\n cls_need_removal:{}"\
          .format(RandomSeed, n_neuron, n_layers, epoch,k,rm_ratio,cls_need_removal))
    ##scale data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    ##shuffle data
    from sklearn.utils import shuffle
    X, labels = shuffle(X, labels)


    plt.scatter(X[:, 0], X[:, 1], s=2, c = labels)
    # plt.title("Original Moon Dataset")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2");
    plt.axis('off')
    # from google.colab import files
    plt.savefig(figure_path + "/org_{}.jpg".format(dataset_name))


    from collections import Counter
    c=Counter(labels)
    print("Total original data {}".format(c))


    ###remove some sample for unbalance 
    for cls,ratio in zip(cls_need_removal,rm_ratio):
        cnt = int(len(labels[labels==cls])*ratio) #remove rm_ratio of train data
        rm=[]
        for i,l in enumerate(labels):
            if l==cls and cnt>0 :
                rm.append(i)
                cnt -=1
        X = np.delete(X, rm, axis=0)
        labels = np.delete(labels, rm, axis=None)


    c_total_after_rm = Counter(labels);
    print("Total counter after Imabalnce Removal: {}".format(c_total_after_rm))
    c_0 = c_total_after_rm[0.0]
    c_1 = c_total_after_rm[1.0]
    print("*** Imabalance Ratio (IR): {}\n".format( c_0/c_1 if c_0 > c_1 else c_1/c_0 )  )

    ####SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state = RandomSeed)
    plt.scatter(X_train[:, 0], X_train[:, 1], s=2, c = y_train)
    # plt.title("Imbalanced Moon Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2");
    plt.axis('off')
    plt.savefig(figure_path + "/{}.jpg".format(dataset_name))
    # plt.show()
    print("X_train: {}".format( X_train.shape))
    print("y_train: {}".format( y_train.shape) )
    print("X_test: {}".format(X_test.shape))
    print("y_test: {}".format( y_test.shape))
    print("Counting the number of samples in each class")
    c_train = Counter(y_train); print("Train set {}".format(c_train))
    c_test = Counter(y_test);   print("Test set {}".format(c_test ))
    return X_train, X_test, y_train, y_test

##create imbalanced dataset by removing samples in minority classes
## testing on widely-used methods


def normalize1d(X):   
    return (X - X.min() ) / (( X.max() - X.min() ) + 0.00000001)

def plotboundary(clf, X,y, model_name="DNN", title='',save=''):
    input_dim= X.shape[1]             
    # create a mesh to plot in
    x0 = X[:, 0]
    x1 = X[:, 1]
    x_min, x_max =  x0.min() - 0.2, x0.max() + 0.2
    y_min, y_max =  x1.min() - 0.2, x1.max() + 0.2
    
    if input_dim==2:
        h = .02  # step size in the mesh   
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, m_max]x[y_min, y_max].
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
    # Put the result into a color plot  
    fig = plt.figure()
    plt.scatter(x0, x1, s=2, c = y)
    if input_dim==2: 
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
#     plt.axis('off')
    plt.title(title)
    plt.savefig(figure_path + "/"+save+".jpg")
#     plt.show()
    
    



## create simple deep fully connected neural network
class My_classifier():   
    def __init__(self,epochs = 200, n_neuron= 300, batch_size=64,n_layers=3):      
        # Create a MirroredStrategy.## model checkpoint
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint,
            save_weights_only=False,
            monitor='loss',
            mode='min',
            save_best_only=True)
        ##early stop callback
        early_stop =  tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15,restore_best_weights=True)
        Callbacks = [early_stop, model_checkpoint_callback]
        
        strategy = tf.distribute.MirroredStrategy( cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
#         print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        # Open a strategy scope.
        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            self.epochs = epochs
            self.batch_size=batch_size
            self.callbacks= Callbacks
            self.model = Sequential()
            self.model.add(Dense(n_neuron, input_dim=input_dim, activation='relu'))
            for i in range(n_layers-1):
                # self.model.add(Dropout(0.2))
                self.model.add(Dense(n_neuron, activation='relu'))
            self.model.add(Dense(2, activation='softmax'))
            # compile the keras model
            opt = tf.keras.optimizers.SGD(learning_rate=0.01)
            self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            
    def fit(self,X,y, X_val, y_val, sample_weight = None):     
        y = to_categorical(y, num_classes=2)
        y_val = to_categorical(y_val, num_classes=2)
        if sample_weight is not None:
            return self.model.fit(X, y, validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks= self.callbacks, sample_weight= sample_weight )
        else:
            return self.model.fit(X, y,validation_data=(X_val, y_val), epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks= self.callbacks,)
    def predict_proba(self,X):
        return self.model.predict(X)
    def predict(self,X):
        return np.argmax(self.model.predict(X),axis=-1)

    
def loss_plot(history, file_path, title):
    history_loss_path = file_path.replace('.png','') + '.csv'
    # print("history.history.keys(): {}".format( history.history.keys()) ) 
    # keys history : ['loss', 'accuracy', 'val_loss', 'val_accuracy']
    data = [history.history[k] for k in history.history.keys()]
    df = pd.DataFrame(data)
    df.to_csv(history_loss_path)

def data_save(X,y, file_path):
    #save data 
    data_path = file_path.replace('png','csv').replace("train loss","data")
    print("X shape, y shape : {} {}".format(X.shape, y.shape ))
    y_expanded = np.expand_dims(y,axis=1)
    df = pd.DataFrame(np.concatenate( (X,y_expanded ),axis=1  ) )
    df.to_csv(data_path, index=False)

def n_classification(X_train, y_train,X_val,y_val, X_test, y_test, n=1, epoch=120,**kwargs):
    """
    Classify n times and  return the average
    n : number of runs
    epoch: training epochs
    return 
    F1Scores: array of n f1 scores 
    """ 
    method = kwargs.get("method")
    num_runs = kwargs.get("num_runs")
    if method is not None and num_runs is not None:
        no_log = False
    else: no_log = True
    
    F1Scores = []
    pred_array = []
    for i in range(n):    
        classifier = My_classifier(epochs=epoch,n_neuron=n_neuron,n_layers=n_layers)
        history = classifier.fit(X_train, y_train, X_val, y_val)
        y_pred = classifier.predict(X_test)
        y_pred_prob = classifier.predict_proba(X_test)[:,1]
        pred_array.append(y_pred_prob)
        report = classification_report(y_test, y_pred, digits=4, output_dict= True)
        F1Scores.append(report['macro avg']['f1-score'] )
        #plot
        title = "{}-train loss-run-{}-{}".format(method,num_runs,i)
        loss_path = os.path.join(result_dir,title+'.png')
        #save loss plot
        loss_plot(history,loss_path, title )
        data_save(X_train, y_train, loss_path)

    ##logging
    if not no_log:
        #adding groundtruth at the last column
        pred_array.append(y_test)
        predict_log_path = os.path.join(result_dir,"{}-pred_and_label.{}-{}.csv".format(method,num_runs,i))
        #save prediction
        df = pd.DataFrame(pred_array)
        df.to_csv(predict_log_path)
        
    return np.array(F1Scores),classifier

def OtherMethod_n_runs(**kwargs):
    """
    Run different times, 
    """
    method = kwargs.get("method")
    X_train = kwargs.get("X_train"); y_train = kwargs.get("y_train"); 
    X_test = kwargs.get("X_test"); y_test = kwargs.get("y_test");
    num_runs = kwargs.get("num_runs")
    F1 = []
    randomSplit = True # random split data each test
    print("\nStart testing on {} method.".format(method)) 
    for i in range(num_runs): 
        rand = RandomSeed+i+1
        if randomSplit: X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state = rand)
        t = time.time()
        if method=='SMOTE':
            sm = SMOTE( k_neighbors = 5)
            X_train_new, y_train_new = sm.fit_resample(X_train, y_train)
        elif method == 'Oversampling':
            oversample  = RandomOverSampler(sampling_strategy='minority')
            X_train_new, y_train_new = oversample.fit_resample(X_train, y_train) 
        elif method == 'BorderlineSMOTE':
            X_train_new, y_train_new =  BorderlineSMOTE().fit_resample(X_train, y_train)
        elif method == 'ADASYN':
            X_train_new, y_train_new  = ADASYN().fit_resample(X_train, y_train)
        else:
            X_train_new = X_train
            y_train_new = y_train
        #split to training and validation sets 
        print("-Balancing time: {:.2f} Minutes".format( (time.time() - t)/60 ) )
        temp_f1, _ = n_classification(X_train_new, y_train_new, X_test, y_test, X_test, y_test, 1, epoch, method=method,num_runs=i)
        F1.append(temp_f1.mean())
        print("-Time for each run including classification: {:.2f} Minutes".format( (time.time() - t)/60 ) )
        print("-F1 Score on this run: {:.5f}".format(temp_f1.mean()))
    return np.array(F1)


def SIMPOR_n_runs(**kwargs):  
    k = kwargs.get("k"); h = kwargs.get("h"); p = kwargs.get("p"); AL_classifier = kwargs.get("AL_classifier"); 
    epoch = kwargs.get("epoch")
    X_train = kwargs.get("X_train"); y_train = kwargs.get("y_train"); 
    X_test = kwargs.get("X_test"); y_test = kwargs.get("y_test");
    num_runs = kwargs.get("num_runs")
    n_threads = kwargs.get("n_threads"); CUDA = kwargs.get("CUDA")
    SIMPOR_F1_informative = []
    SIMPOR_F1_full = []
    randomSplit = True # random split data each test
    method = 'SIMPOR'

    for i in range(num_runs):
        rand = RandomSeed+i+1
        if randomSplit: X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state = rand)
        t = time.time()
        X_full, y_full, X_informative, y_informative = \
            max_FracPosterior_balancing(X_train, y_train,k=k, h=h, AL_classifier = AL_classifier, \
                                            informative_threshold=p, n_threads= n_threads, CUDA=CUDA)
        print("-Balancing data time: {:.2f} Minutes".format( (time.time() - t)/60 ) )
        #train on full SIMPOR global (including final step for balancing)
        #split to training and validation sets 
        F1_full, _ =  n_classification( X_full, y_full, X_test, y_test, X_test, y_test, 1, epoch, method=method, num_runs=i)
        print("-SIMPOR at run ({}) F1 Score: {:.5f}\n".format(i ,F1_full.mean()) )
        SIMPOR_F1_full.append(F1_full.mean())
        print("-Time for each run including classification: {:.2f} Minutes".format( (time.time() - t)/60 ) )
    return np.array( SIMPOR_F1_full)


import argparse
if __name__== "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="DataSets: moon, breast_cancer, creditcard",
                        choices={'moon', 'breast_cancer' ,'creditcard', 'mnist'},
                        type=str, default="moon")
    parser.add_argument("--n_threads", help="number of threads, default 34, If using CUDA -> n_threads is set to 1",
                        type=int, default=10)
    parser.add_argument("--n_runs", help="number of trials, default 5",
                        type=int, default=5)
    parser.add_argument("--randseed", help="Random seed to split data, default 99",
                        type=int, default=99)
    parser.add_argument("--IR", help="Imbalance Ratio, default =3",
                        type=int, default=3)
    parser.add_argument("--cuda", help="using cuda, boolean{True,False}, default True",
                        type=bool, default=True)
    parser.add_argument("--gridSearch", help="Find the best params for Simpor, boolean{True,False}, would take a while, default True",
                        type=bool, default=True)
    args = parser.parse_args()
    
    global dataset_name, input_dim, X_train, X_test, y_train, y_test, RandomSeed
    dataset_name = args.dataset
    n_threads = args.n_threads
    num_runs = args.n_runs 
    RandomSeed = args.randseed
    IR = args.IR
    gridSearch = args.gridSearch
    CUDA = args.cuda
    # if CUDA: n_threads=1 

    # data preparation
    log_prepare(dataset_name)
    X_train, X_test, y_train, y_test = data_gen(dataset_name,IR)
    input_dim = X_train.shape[1]
    
    ###############################################
    #Searching for the best parameters   
    
    org_classifier = My_classifier(epochs= epoch,n_neuron=n_neuron, n_layers = n_layers)
    org_classifier.fit(X_train, y_train, X_test,y_test)
    y_pred = org_classifier.predict(X_test)
    acc =  accuracy_score(y_test, y_pred)
    print("\n\n\nReport On raw data \
        (This is one time running, check the stable result at the final test): ")
    print("Acc: {}".format(acc) )
    print("F1_score".format( str(f1_score(y_test, y_pred, average = 'macro')) ) )
    print(str(confusion_matrix( y_test, y_pred) ) )
    print(str(classification_report(y_test, y_pred, digits=4)) )  
    plotboundary(org_classifier, X_train ,y_train , model_name="dnn", title='Imbalanced Data', save='ImbalancedData')

    ### if search for best paras
    if gridSearch:
        Plot= True
        h_list = [  0.1] # bandwidth of Gaussian kernel
        entropy_threshold_list = [0.1, 0.2, 0.3]
        runs  = len(h_list) * len(entropy_threshold_list)
        max_f1 = {'f1':0,'h':0,'p':0}
        print("\n\n------------------Finding best Params SIMPOR in {} runs-----------------------\n".format(runs))
        for h in h_list:
            for p in entropy_threshold_list:
                # balancing data using SIMPOR
                print("\n---Params h={}   k={}   Threshold={}   ---".format(h,k,p))
                t= time.time()
                AL_classifier = org_classifier
                X_full, y_full, X_informative, y_informative = \
                max_FracPosterior_balancing(X_train, y_train,k=k, h=h, AL_classifier = AL_classifier, \
                                            informative_threshold=p, n_threads = n_threads, CUDA= CUDA)
                print("X_full shape: {}".format( str(X_full.shape) ) )    
                SIMPOR_F1_informative, classifier_informative = n_classification(X_informative, y_informative,X_test, y_test, X_test, y_test, 1, epoch)
                SIMPOR_F1_full, classifier_full =  n_classification(X_full, y_full, X_test, y_test,X_test, y_test, 1, epoch)
                if SIMPOR_F1_full.mean() > max_f1['f1']: #update paras to find the best
                    max_f1['f1'] = SIMPOR_F1_full.mean(); max_f1['h']=h;  max_f1['p']=p; print("SIMPOR k={} Threshold={}  ".format(k, p))
                print("SIMPOR_F1_informative score:{} std:{} ".format(str(SIMPOR_F1_informative.mean()), str(SIMPOR_F1_informative.std()) ) )
                print("SIMPOR_F1_full score: {} std:{}".format(str(SIMPOR_F1_full.mean()),  str(SIMPOR_F1_full.std())) )
                if Plot  : 
                    plotboundary(classifier_informative, X_informative ,y_informative , model_name="dnn", title='SIMPOR', save="SIMPOR k={} Threshold={} Local".format(k, p ))
                    plotboundary(classifier_full, X_full ,y_full , model_name="dnn", title='SIMPOR', save="SIMPOR k={} Threshold={} Global".format(k, p))
                print("Time for this run (including classfication): {:.3f} seconds ".format( (time.time()-t) ) )
    else:
        #default 
        max_f1 = {'f1':0,'h':0.1,'p':0.2}

                        
    #######################################################
    ##final test with the best params
    print("\n\n====================Final Test on {} runs====================".format(str((num_runs)) ) )
    h = max_f1['h']
    p = max_f1['p']
    AL_classifier = org_classifier

    
    t = time.time()
    print("SIMPOR k={} h = {} Threshold={}  ".format(k, h, p))
    SIMPOR_F1_full =  SIMPOR_n_runs(X_train=X_train, y_train=y_train,X_test=X_test, y_test=y_test, k=k, h=h, AL_classifier = AL_classifier, \
                                            p=p,  num_runs=num_runs, epoch=epoch, n_threads=n_threads,  CUDA= CUDA)
    
    ##Test on original data 
    t1 = time.time()
    Original_F1_scores =  OtherMethod_n_runs(method = 'RawData', X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_runs=num_runs, epoch=epoch)

    #test Other methods
    t2 = time.time() 
    BorderlineSMOTE_F1_scores =  OtherMethod_n_runs(method = 'BorderlineSMOTE', X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_runs=num_runs, epoch=epoch)
    
    
    t3 = time.time() 
    ADASYN_F1_scores =  OtherMethod_n_runs(method = 'ADASYN', X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_runs=num_runs, epoch=epoch)
    
    
    t4 = time.time() 
    SMOTE_F1_scores =  OtherMethod_n_runs(method = 'SMOTE', X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_runs=num_runs, epoch=epoch)
    

    t5 = time.time() 
    Oversampling_F1_scores =  OtherMethod_n_runs(method = 'Oversampling', X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test, num_runs=num_runs, epoch=epoch)
    

    t6 = time.time() 
    print("\n\n=====Summary after {} runs====".format(num_runs))
    print("SIMPOR_F1 mean: {:.5f}  std: {:.5f} Time:{:.4f} minutes".format( SIMPOR_F1_full.mean(), SIMPOR_F1_full.std() ,(t1- t)/60 )  )
    print("BorderlineSMOTE_F1_scores mean: {:.5f}  std: {:.5f} Time:{:.4f} minutes".format( BorderlineSMOTE_F1_scores.mean(),BorderlineSMOTE_F1_scores.std(), (t3 - t2)/60  ))
    print("ADASYN_F1_scores mean: {:.5f}  std: {:.5f} Time:{:.4f} minutes".format(ADASYN_F1_scores.mean() ,ADASYN_F1_scores.std()  ,(t4 - t3)/60 ) )
    print("SMOTE_F1_scores mean: {:.5f}  std: {:.5f} Time:{:.4f} minutes".format( SMOTE_F1_scores.mean() , SMOTE_F1_scores.std()  ,(t5 - t4)/60 ) )
    print("Oversampling_F1_scores mean: {:.5f}  std: {:.5f} Time:{:.4f} minutes".format( Oversampling_F1_scores.mean() , Oversampling_F1_scores.std() ,(t6 - t5)/60 )  )
    print("Original_F1_scores mean: {:.5f}  std: {:.5f} Time:{:.4f} minutes".format(Original_F1_scores.mean() ,Original_F1_scores.std() ,(t2- t1)/60 ) )