
"""
This file will go through all the experiments directories 
and generate ROC curve plot for each exp
"""

from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_fscore_support
import pickle as csv
import os
import re
from sklearn.preprocessing import label
import pandas as pd
import numpy as np
from matplotlib import  pyplot as plt
from collections import defaultdict
from scipy import interp
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
np.random.seed(1)
import plotly.figure_factory as ff
import re

rootdir = "LOG"
expDict = {}
expDictLoss = {}
expData = {}

# define which class considered positive (minority class) when compute AUC and ROC 
pos_label_dict = { 'breast_cancer':0, 'creditcard': 1, 'moon':0}

def histogram_intersection(h1, h2, bin_num):
    sm = 0
    for i in range(bin_num):
        sm += min(h1[i], h2[i])
    return sm



def listdirs(rootdir):
    for it in os.scandir(rootdir):
        if it.is_dir() and 'old' not in it.name :
            listdirs(it)
        elif it.name.endswith('.csv') and 'label' in it.name: 
            exp_dir = it.path.split('/')[-3]
            expDict.setdefault(exp_dir,[]).append(it.path)
        elif it.name.endswith('.csv') and 'loss' in it.name: 
            exp_dir = it.path.split('/')[-3]
            expDictLoss.setdefault(exp_dir,[]).append(it.path)
        elif it.name.endswith('.csv') and 'data' in it.name: 
            exp_dir = it.path.split('/')[-3]
            expData.setdefault(exp_dir,[]).append(it.path)

listdirs(rootdir)
cwd = os.getcwd()
#plot loss
for exp in expDictLoss.keys():
    print("\n\n========================================")
    print("Exp: {}".format(exp))
    #only rerender loss figures if missing _ROC.jpg in the directory
    log_dir = cwd+'/LOG'
    roc_file = os.path.join(exp, 'Figures/_ROC.jpg')
    loss_dir = log_dir +'/'+exp+ '/Figures' + '/LOSS'
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
    if not os.path.exists(os.path.join(log_dir, roc_file)):
        
        for idx, csvfile in enumerate(expDictLoss[exp]):
            
            #get file name and method 
            fileName = csvfile.split('/')[-1]
            data = pd.read_csv(csvfile)
            data = np.array(data).T
            line_space = range(len(data[1:,0]))
            epoch = len(data[1:,0])

            #plot loss graphs
            plt.figure()
            plt.plot(line_space, data[1:,0], label='Train_loss')
            # plt.plot(line_space, data[1:,1], label='train_acc')
            var = np.var(data[int(epoch/2):,2])
            plt.plot(line_space, data[1:,2], label='Test_loss\nVar:{:.1e}'.format(var))
            # plt.plot(line_space, data[1:,3], label='val_acc')
            plt.legend()
            plt.savefig(csvfile.replace('.csv', "loss.jpg").replace('Results','Figures/LOSS') )
            plt.close()
 

##----------------------------------       
#plot data
for exp in expData.keys():  
    if False:  
        print("\n\n========================================")
        print("Exp: {}".format(exp))
    #only rerender loss figures if missing _ROC.jpg in the directory
    log_dir = cwd+'/LOG'
    roc_file = os.path.join(exp, 'Figures/_ROC.jpg')
    pca_dir = log_dir +'/'+exp+ '/Figures' + '/PCA'
    if not os.path.exists(pca_dir):
        os.makedirs(pca_dir)
    if not os.path.exists(os.path.join(log_dir, roc_file)):
        for idx, csvfile in enumerate(expData[exp]):
            fileName = csvfile.split('/')[-1]
            data = pd.read_csv(csvfile)
            data = data.dropna()
            y  = data.iloc[:,-1]
            if len(data.columns)>3:
                dim_reduce = PCA(n_components=2, svd_solver='full')
                # dim_reduce = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
                twoD_results = dim_reduce.fit_transform(data.iloc[:,0:-1])
                if  ('SIMPOR' in fileName) or ('Over' in fileName) or ('SMOTE' in fileName and 'Border' not in fileName) :
                    ##rotate images
                    x1 = twoD_results[:,0]
                    x0 = twoD_results[:,1] 
                else:
                    x0 = twoD_results[:,0]
                    x1 = twoD_results[:,1] 
                        
                ##reduce to 1d
                dim_reduce = PCA(n_components=1, svd_solver='full')
                oneD_results = dim_reduce.fit_transform(data.iloc[:,0:-1])
                x1D = oneD_results[:,0]
                #scale to 0-1
                x1D = x1D-np.min(x1D) 
                x1D  /= np.max(x1D)
                           
            else:
                x0 = data.iloc[:,0]
                x1 = data.iloc[:,1]

            df = pd.DataFrame(data = zip(x0,x1,y), columns = ['x0','x1','y'] )
            ##plot 2D
            colors = np.array(['rgb(0, 0, 100)' if i==1 else 'rgb(0, 200, 200)' for i in y ])
            trace0 = go.Scatter(x = x0[y==0],y = x1[y==0], name='Normal',  mode='markers',marker=go.scatter.Marker(size=8, color= 'rgb(0, 0, 100)', opacity=1, symbol = ['star']*len(x0[y==0])), ) 
            trace1 = go.Scatter(x = x0[y==1],y = x1[y==1], name='Fraudulent', mode='markers',marker=go.scatter.Marker(size=8, color= 'rgb(0, 200, 200)', opacity=1, symbol = ['star']*len(x0[y==1])), ) 
            fig = go.Figure([trace0,trace1])
            fig.update_layout(showlegend=True, legend= {'x':0.7,'y':0.9,'itemsizing': 'constant', 'font':{'size':24}, })
            fig.update_yaxes(visible=False, showticklabels=False,)
            fig.update_xaxes(visible=False, showticklabels=False, )
            fig.write_image(csvfile.replace('.csv', ".jpg").replace('Results','Figures/PCA'), width=1080, height=1080)
            
            
            ##plot Histogram
            bins = np.linspace(0,1,20)
            
            fig = ff.create_distplot([x1D[y==0], x1D[y==1] ], [ 'Normal', 'Fraudulent'],colors=['rgb(0, 0, 100)', 'rgb(0, 200, 200)'], bin_size=bins, show_rug=False)
            fig.update_layout(showlegend=True, legend= {'x':0.7,'y':0.9, 'itemsizing': 'constant', 'font':{'size':24}, })
            fig.update_yaxes(visible=False, showticklabels=False,)
            fig.update_xaxes(visible=False, showticklabels=False, )
            fig.write_image(csvfile.replace('.csv', "_DIST.jpg").replace('Results','Figures/PCA'), width=1200, height=800)

            #histogram overlap 
            if False:
                count0, bin0 = np.histogram(x1D[y==0], bins=bins)
                count1, bin1 = np.histogram(x1D[y==1], bins=bins)
                sm = histogram_intersection( count0, count1, bin_num = len(bins)-1 )
                print("File name: {} ---  Similarity: {}  total: {}  ratio: {:.2f}%".format(fileName,sm, len(x1D), sm/len(x1D)*100) )

##----------------------------------
# for each experiment folder
for exp in expDict.keys():
    print("\n\n========================================")
    print("Exp: {}".format(exp))
    #only rerender loss figures if missing _ROC.jpg in the directory
    log_dir = cwd+'/LOG'
    roc_file = os.path.join(exp, 'Figures/_ROC.jpg')
    if not os.path.exists(os.path.join(log_dir, roc_file)):
        # run through all method
        
        
        plt.figure()
        dataset = exp.split("202")[0]
        print("Minority Class:", pos_label_dict[dataset])
        results = {}
        tprs =  defaultdict(list)
        auc = defaultdict(list)
        f1 = defaultdict(list)
        precision = defaultdict(list)
        recall = defaultdict(list)
        base_fpr = np.linspace(0, 1, 101)
        
        for idx, csvfile in enumerate(expDict[exp]):
            #get file name and method 
            fileName = csvfile.split('/')[-1]
            method = fileName.split('-')[0]
            # print("Method: ",method)
            data = pd.read_csv(csvfile)
            data = np.array(data).T
            # print("data  shape:", data.shape)
            #extract groundtruth in the last comlumn
            testy = data[:,-1] 
            #averaging different experiments
            try:
                pred = np.mean(data[:,:1],axis=1)
            except:
                print(fileName)
                print(data[:,:1])
                raise os.error
            fpr, tpr, thresholds = roc_curve(testy, pred, pos_label = pos_label_dict[dataset] )
            tpr_temp = interp(base_fpr, fpr, tpr)
            tpr_temp[0] = 0.0
            tprs[method].append(tpr_temp)
            auc[method].append(roc_auc_score(testy, pred))
            #compute f1
            # tmp_f1 = f1_score(testy, np.rint(pred), average = 'macro') 
            tmp_ps,tmp_re,tmp_f1,_ = precision_recall_fscore_support(testy, np.rint(pred), average='macro', pos_label = pos_label_dict[dataset])
            f1[method].append(tmp_f1)
            precision[method].append(tmp_ps)
            recall[method].append(tmp_re)

        for m in tprs.keys():
            tprs[m] = np.array(tprs[m])
            mean_tprs = tprs[m].mean(axis=0)
            auc[m] = np.array(auc[m])
            mean_auc = auc[m].mean(axis=0)
            std = tprs[m].std(axis=0)
            #f1 average
            f1[m] = np.array(f1[m])
            mean_f1 = f1[m].mean(axis=0)
            precision[m] = np.array(precision[m])
            mean_precision = precision[m].mean(axis=0)
            recall[m] = np.array(recall[m])
            mean_recall= recall[m].mean(axis=0)


            tprs_upper = np.minimum(mean_tprs + std, 1)
            tprs_lower = mean_tprs - std
            print('{:20s}:  AUC: {:.3f}  F1: {:.3f}  Precision: {:.3f}  Recall: {:.3f}'.format(m, mean_auc, mean_f1, mean_precision, mean_recall ))
            results[m] = ['{:.3f}'.format(mean_f1),'{:.3f}'.format(mean_auc),'{:.3f}'.format(mean_precision),'{:.3f}'.format(mean_recall)  ] 
            # plot model roc curve
            if 'SIMPOR' in m: alpha=1 ; linewidth=1.5; marker='*'
            else: alpha = 0.8; linewidth=1; marker=''

            plt.plot(base_fpr, mean_tprs, label=m, alpha=alpha, linewidth= linewidth, marker=marker, markevery=5 )
            # plt.fill_between(base_fpr, tprs_lower, tprs_upper,  alpha=0.3)

            # axis labels
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            
            # show the plot
        plt.plot([0, 1], [0, 1],'k--', label="Random Guess", alpha = 0.8, linewidth=1)
        # show the legend
        plt.legend(loc='lower right')
        plt.savefig(csvfile.replace(fileName, "_ROC.jpg").replace("Results", "Figures") )
        plt.close
        #LOG TO CSV FILE
        df = pd.DataFrame(results, index=['F1', 'AUC', 'Precision', 'Recall'])
        # df.rename(columns={0:'F1', 1:'AUC', 2:'Precision', 3:'Recall' }, inplace=True)
        df.to_csv(csvfile.replace(fileName, "_Final_Results.csv"), index=True)
