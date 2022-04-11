



import pickle as csv
import os
import pandas as pd
import numpy as np

      
logDir = 'LOG/'
metrics = ['MacroF1', 'MicroF1', 'AUC', 'Precision', 'Recall', 'MacroGmean', 'MicroGmean']
methods = ['RawData', 'BorderlineSMOTE' ,'ADASYN','SMOTE',  'ROS','GDO', 'SIMPOR']
datasets = []
dfDict = {}
exclude = ['RawData']
###Read all result files
for subdir, dirs, files in os.walk(logDir):
    for file in files:
        if file == '_Final_Results.csv': 
            filePath = os.path.join(subdir, file)
            print(filePath)
            df = pd.read_csv(filePath,header=0, index_col=0)
            dataset = filePath.split('/')[1].split('2022')[0]
            if dataset  not in datasets: datasets.append(dataset)
            df= df.drop(exclude, axis=1)
            dfDict[dataset] = df
            print(dfDict[dataset])



metricTableDict = {}
##concatenate all dataset
for metric in metrics:
    metricTableDict[metric]  = pd.concat([dfDict[dataset].loc[[metric],:] for dataset in datasets ], axis=0)
    metricTableDict[metric].index = [dataset for dataset in datasets]

#### Sample
#metricTableDict['MacroF1'] 
#          method1 method2 method3
#dataset1    0.5     0.8    0.9
#dataset2    0.5     0.8    0.9
#dataset3    0.5     0.8    0.9


writer = pd.ExcelWriter("CollectedResult.xlsx", engine='xlsxwriter')
workbook  = writer.book
bold_format = workbook.add_format({'bold': True})
header_format  = header_format = workbook.add_format({
    'bold': True,
    'text_wrap': True,
    'valign': 'top',
    'fg_color': '#D7E4BC',
    'border': 1})

for metric in metrics:
    dfTable = metricTableDict[metric] 
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    worksheet = workbook.add_worksheet(metric) #sheet name
    # Add a header format.
    
    # Write the column headers with the defined format.
    print(dfTable)
    #write header
    for col_num, value in enumerate(dfTable.columns.values):
        worksheet.write(0, col_num + 1, value)
    #write row names 
    for row_num, value in enumerate(dfTable.index):
        worksheet.write(row_num+1, 0, value)
    #write values
    print("---------------------------------\n{}".format(dfTable ))
    for i, row in enumerate(dfTable.values):
        max_val = dfTable.max(axis=1).loc[datasets[i]]
        for j, val in enumerate(row):
            if val >= max_val: 
                worksheet.write(i+1, j+1, val, bold_format)
            else: 
                worksheet.write(i+1, j+1, val)

    # Close the Pandas Excel writer and output the Excel file.
writer.save()

