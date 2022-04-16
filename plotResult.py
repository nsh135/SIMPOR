
"""
This file will go through all the experiments directories 
and generate ROC curve plot for each exp
"""

from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_recall_fscore_support
import pickle as csv
import os
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
import builtins as __builtin__
import sys
from imblearn.metrics import geometric_mean_score
from sklearn.manifold import TSNE
import seaborn as sns

rootdir = "LOG"
expDict = {}
expDictLoss = {}
expData = {}
exclude = ['None','RawData' ]
# define which class considered positive (minority class) when compute AUC and ROC 

methods = [  'SIMPOR','GDO', 'SMOTE', 'BorderlineSMOTE' ,'ADASYN',  'ROS',]
###Generating visualization plot pdf
def generateTemplate(orgTemplate,editedTemplate,replaceDict ):
    # %%place holders need to be replaced
    # %% @xDataset: dataset name
    # %% @xMinCount, @xMajCount, @xInterCount, @xHDR
    # %% @1_2dPath, @1_histPath
    # Read in the file
    # print("**************{}".format(replaceDict))
    with open(orgTemplate, 'r') as file :
        filedata = file.read()

    # Replace the target string
    for i,method in enumerate(methods):
        minor,major,inter,hdr,path2d,histpath = replaceDict[method]
        filedata = filedata.replace('@{}Dataset'.format(i+1), method)
        filedata = filedata.replace('@{}MinCount'.format(i+1), str(minor))
        filedata = filedata.replace('@{}MajCount'.format(i+1), str(major))
        filedata = filedata.replace('@{}InterCount'.format(i+1), str(inter))
        filedata = filedata.replace('@{}HDR'.format(i+1), "{:.2f}".format(hdr) ) 
        filedata = filedata.replace('@{}_2dPath'.format(i+1), str(path2d))
        filedata = filedata.replace('@{}_histPath'.format(i+1), str(histpath))
    # Write the file out again
    with open(editedTemplate, 'w') as file:
        file.write(filedata)
    os.system("pdflatex  -output-directory={} -aux-directory={}  {}".format( os.path.dirname(editedTemplate),  os.path.dirname(editedTemplate), editedTemplate ) )

def print(*args, **kwargs):
    """My custom print() function.
       Print out and also keep loging 
    """
    with open(log_file, "a") as log:  
        log.write("\n")
        log.writelines(args)
    return __builtin__.print(*args, **kwargs)


def histogram_intersection(h1, h2, bin_num):
    sm = 0
    for i in range(bin_num):
        sm += min(h1[i], h2[i])
    return sm



def listdirs(rootdir,expdir ):
    for it in os.scandir(rootdir):
        if it.is_dir() and 'old' not in it.name:
            listdirs(it,expdir)
        elif it.name.endswith('.csv') and 'label' in it.name and expdir in it.path: 
            exp_dir = it.path.split('/')[-3]
            expDict.setdefault(exp_dir,[]).append(it.path)
        elif it.name.endswith('.csv') and 'loss' in it.name and expdir in it.path: 
            exp_dir = it.path.split('/')[-3]
            expDictLoss.setdefault(exp_dir,[]).append(it.path)
        elif it.name.endswith('.csv') and 'data' in it.name and expdir in it.path: 
            exp_dir = it.path.split('/')[-3]
            expData.setdefault(exp_dir,[]).append(it.path)

def plotResult(expdir=''):
    listdirs(rootdir,expdir )
    cwd = os.getcwd()
    global log_file 
    #plot loss
    for exp in expDictLoss.keys():
        #only rerender loss figures if missing _ROC.jpg in the directory
        log_dir = cwd+'/LOG'
        roc_file = os.path.join(exp, 'Figures/_ROC.jpg')
        loss_dir = log_dir +'/'+exp+ '/Figures' + '/LOSS'
        log_file = log_dir +'/'+exp+ '/log.txt'

        print("\n\n========================================")
        print("Exp: {}".format(exp))
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)
            
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
        #only rerender loss figures if missing _ROC.jpg in the directory

        log_dir = cwd+'/LOG'
        roc_file = os.path.join(exp, 'Figures/_ROC.jpg')
        pca_dir = log_dir +'/'+exp+ '/Figures' + '/PCA'
        tsne_dir = log_dir +'/'+exp+ '/Figures' + '/TSNE'
        log_file = log_dir +'/'+exp+ '/log.txt'
        
        reduceMethods = ['PCA']  #['TSNE', 'PCA']
        templateDict = {}
        if False:  
            print("\n\n========================================")
            print("Exp: {}".format(exp))
        if not os.path.exists(pca_dir):
            os.makedirs(pca_dir)
        if not os.path.exists(tsne_dir):
            os.makedirs(tsne_dir)

        for idx, csvfile in enumerate(expData[exp]):
            fileName = csvfile.split('/')[-1]
            data = pd.read_csv(csvfile)
            data = data.dropna()
            method = fileName.split('-')[0]
            y  = data.iloc[:,-1]
            for reduceMethod in reduceMethods:
                if len(data.columns)>=3:
                    if reduceMethod == 'TSNE':
                        dim_reduce = TSNE(n_components=2, verbose=0,perplexity=40, n_iter=300, random_state=123)
                    elif reduceMethod == 'PCA':
                        dim_reduce = PCA(n_components=2, svd_solver='full')
                    else: break

                    twoD_results = dim_reduce.fit_transform(data.iloc[:,0:-1])
                    if  ('SIMPOR' in fileName) or ('ROS' in fileName) or ('SMOTE' in fileName and 'Border' not in fileName) :
                        ##rotate images
                        x1 = twoD_results[:,0]
                        x0 = twoD_results[:,1] 
                    else:
                        x0 = twoD_results[:,0]
                        x1 = twoD_results[:,1] 
                            
                    ##reduce to 1d
                    if reduceMethod == 'TSNE':
                        dim_reduce = TSNE(n_components=1, verbose=0,perplexity=40, n_iter=300, random_state=123)
                    elif reduceMethod == 'PCA':
                        dim_reduce = PCA(n_components=1, svd_solver='full')
                    else: break
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
                trace0 = go.Scatter(x = x0[y==0],y = x1[y==0], name='Majority',  mode='markers',marker=go.scatter.Marker(size=8, color= 'rgb(0, 0, 100)', opacity=1, symbol = ['star']*len(x0[y==0])), ) 
                trace1 = go.Scatter(x = x0[y==1],y = x1[y==1], name='Minority', mode='markers',marker=go.scatter.Marker(size=8, color= 'rgb(0, 200, 200)', opacity=1, symbol = ['star']*len(x0[y==1])), ) 
                fig = go.Figure([trace0,trace1])
                fig.update_layout(showlegend=True, legend= {'x':0.7,'y':0.9,'itemsizing': 'constant', 'font':{'size':24}, })
                fig.update_yaxes(visible=False, showticklabels=False,)
                fig.update_xaxes(visible=False, showticklabels=False, )
                path2D = csvfile.replace('.csv', ".jpg").replace('Results','Figures/{}'.format(reduceMethod))
                fig.write_image(path2D, width=1080, height=1080)
                
                
                ##plot Histogram
                bins = np.linspace(0,1,20)
                if len(data.columns)>=3:
                    fig = ff.create_distplot([x1D[y==0], x1D[y==1] ], [ 'Majority', 'Minority'],colors=['rgb(0, 0, 100)', 'rgb(0, 200, 200)'], bin_size=bins, show_rug=False)
                    fig.update_layout(showlegend=True, legend= {'x':0.7,'y':0.9, 'itemsizing': 'constant', 'font':{'size':24}, })
                    fig.update_yaxes(visible=False, showticklabels=False,)
                    fig.update_xaxes(visible=False, showticklabels=False, )
                    path1D = csvfile.replace('.csv', "_DIST.jpg").replace('Results','Figures/{}'.format(reduceMethod))
                    fig.write_image(path1D, width=1200, height=800)

                #histogram 1d 
                if len(data.columns)>=3 and method not in exclude:
                    count0, bin0 = np.histogram(x1D[y==0], bins=bins)
                    count1, bin1 = np.histogram(x1D[y==1], bins=bins)
                    sm = histogram_intersection( count0, count1, bin_num = len(bins)-1 )
                    # print("***File name: {} ---  Similarity: {}  total: {}  ratio: {:.2f}%".format(fileName,sm, len(x1D), sm/len(x1D)*100) )
                    templateDict[method] = (np.sum(count0),np.sum(count1),sm, sm/len(x1D)*100,path2D ,path1D ) # @xMinCount, @xMajCount, @xInterCount, @xHDR
        ###generate entire plot
        if 'moon' not in csvfile:
            editedTemplate = csvfile.replace('.csv', ".tex").replace('Results','Figures/{}'.format(reduceMethod)).replace(method,'')
            generateTemplate('figureGeneratorTemplate.tex',editedTemplate,templateDict )
    ##----------------------------------
    # for each experiment folder
    for exp in expDict.keys():
        
        #only rerender loss figures if missing _ROC.jpg in the directory
        log_dir = cwd+'/LOG'
        roc_file = os.path.join(exp, 'Figures/_ROC.jpg')
        log_file = log_dir +'/'+exp+ '/log.txt'

        print("\n\n========================================")
        print("Exp: {}".format(exp))

        # run through all method
        plt.figure()
        global dataset, MinorityClass
        dataset = exp.split("202")[0]
        if dataset == 'creditcard': MinorityClass = 1
        else: MinorityClass = 0
        print("Minority Class:{}".format( MinorityClass))
        results = {}
        tprs =  defaultdict(list)
        auc = defaultdict(list)
        macf1 = defaultdict(list)
        micf1 = defaultdict(list)
        macgmean = defaultdict(list)
        micgmean = defaultdict(list)
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
                print(str(fileName) )
                print("{}".format(data[:,:1]))
                raise os.error
            fpr, tpr, thresholds = roc_curve(testy, pred, pos_label = MinorityClass )
            tpr_temp = interp(base_fpr, fpr, tpr)
            tpr_temp[0] = 0.0
            tprs[method].append(tpr_temp)
            auc[method].append(roc_auc_score(testy, pred))
            #compute f1
            # tmp_f1 = f1_score(testy, np.rint(pred), average = 'macro') 
            tmp_ps,tmp_re,tmp_f1,_ = precision_recall_fscore_support(testy, np.rint(pred), average='macro', pos_label = MinorityClass)
            macf1[method].append(tmp_f1)
            precision[method].append(tmp_ps)
            recall[method].append(tmp_re)
            tmp_ps,tmp_re,tmp_f1,_ = precision_recall_fscore_support(testy, np.rint(pred), average='micro', pos_label = MinorityClass)
            micf1[method].append(tmp_f1)
            # compute G-mean
            macgmean[method].append(geometric_mean_score(testy, np.rint(pred), average='macro', pos_label = MinorityClass) )
            micgmean[method].append(geometric_mean_score(testy, np.rint(pred), average='micro', pos_label = MinorityClass) )
        for m in tprs.keys():
            tprs[m] = np.array(tprs[m])
            mean_tprs = tprs[m].mean(axis=0)
            auc[m] = np.array(auc[m])
            mean_auc = auc[m].mean(axis=0)
            std = tprs[m].std(axis=0)
            #f1 average
            macf1[m] = np.array(macf1[m])
            micf1[m] = np.array(micf1[m])
            mean_macf1 = macf1[m].mean(axis=0)
            mean_micf1 = micf1[m].mean(axis=0)
            precision[m] = np.array(precision[m])
            mean_precision = precision[m].mean(axis=0)
            recall[m] = np.array(recall[m])
            mean_recall= recall[m].mean(axis=0)
            #
            #G-mean average
            macgmean[m] = np.array(macgmean[m])
            micgmean[m] = np.array(micgmean[m])
            mean_macgmean = macgmean[m].mean(axis=0)
            mean_micgmean = micgmean[m].mean(axis=0)

            tprs_upper = np.minimum(mean_tprs + std, 1)
            tprs_lower = mean_tprs - std
            print('{:20s}:  AUC: {:.3f}  MacroF1: {:.3f}  MicroF1: {:.3f}  Precision: {:.3f}  Recall: {:.3f}   MacroGmean: {:.3f}  MicroGmean: {:.3f} '.format(m, mean_auc, mean_macf1, mean_micf1, mean_precision, mean_recall, mean_macgmean, mean_micgmean ))
            results[m] = ['{:.3f}'.format(mean_macf1),'{:.3f}'.format(mean_micf1),'{:.3f}'.format(mean_auc),'{:.3f}'.format(mean_precision),'{:.3f}'.format(mean_recall), '{:.3f}'.format(mean_macgmean) ,'{:.3f}'.format(mean_micgmean)   ] 
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
        df = pd.DataFrame(results, index=['MacroF1','MicroF1', 'AUC', 'Precision', 'Recall', 'MacroGmean', 'MicroGmean'])
        # df.rename(columns={0:'F1', 1:'AUC', 2:'Precision', 3:'Recall' }, inplace=True)
        df.to_csv(csvfile.replace(fileName, "_Final_Results.csv"), index=True)


if __name__ == '__main__':
    """
    sys.argv[0] : the experiment dir. E.g., moon2022-04-02 19.53.09.484218
    """
    plotResult(sys.argv[1])