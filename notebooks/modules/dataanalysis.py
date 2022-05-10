#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:08:14 2022

@author: tugcecatal
"""
import seaborn as sns
import matplotlib.pyplot as plt

def targetAnalysis(data,col):
    """
    The function is created to see the distribution of the target
    """
    print('The value counts of the default column:\n',data[col].value_counts(dropna=False))

    countNoDefault = len(data[data[col]==0])
    countDefault = len(data[data[col]==1])
    
    rateOfNoDefault = (countNoDefault / (countNoDefault + countDefault)) * 100
    print('The percentage of No Default: {} '.format(str(round(rateOfNoDefault,2)) + "%"))
    
    rateOfDefault = (countDefault / (countNoDefault + countDefault)) * 100
    print('The percentage of No Default: {}'.format(str(round(rateOfDefault,2)) + "%"))
    
    sns.countplot(x=col, data=data, palette='husl')
    
def uniqueValue(data, cols):
    """
    This function is created to get distinct values in a df column according to given cols.
    The aim is to see categorical columns which has a data type int/float.
    """
    uniqueValues = {}
    for col in data[cols]:
        uniqueValues[col] = data[col].nunique()
    return uniqueValues

def numbericalCountPlot(data, col, target='default'):
    """
    This function is created to see the distribution for numerical values and the target column
    """
    sns.kdeplot(data=data, x=col, hue=target, common_norm=False, bw_method=0.15)
    plt.show()
    
def categoricalCountPlot(data, col, target='default'):
    """
    This function is created to see the counts for categorical values and the targt column
    """
    data_counts = (data
                    .groupby([target,col])
                    .agg({col: 'count'})
                    .rename(columns = {col: 'count'}))
    print(data_counts)
    ax = sns.countplot(x=col, data=data, hue =target, palette="Set2")
    plt.title(col , fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'black'}) 
    plt.legend(loc = "upper right")
    plt.xticks(rotation=90, ha='right')
    plt.show()
    
# Show correlation between data columns
def make_corr(data):  
    plt.figure(figsize=(20, 15), dpi=80)
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(data.corr() ,cmap="PiYG", annot=True, fmt='.2f',
                annot_kws={"size": 8}, linewidths=0.8, linecolor='white'
               )
    
def correlation(data, threshold):
    """
    This function is created to get the higly correlated columns
    """
    # define corr dict
    col_corr = {} 
    corr = data.corr()
    # check correlated variables
    for i in range(len(corr.columns)):
        for j in range(i):
            if (corr.iloc[i, j] >= threshold) and (corr.columns[j] not in col_corr):
                # add correlated column to the dict
                col_corr[corr.columns[i]] = corr.columns[j] 
    return col_corr
    

