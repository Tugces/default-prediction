#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 10:48:31 2022

@author: tugcecatal
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def missingAnalysis(data):
    """
    This function is created to get the number of missing in a column and percentages.
    """
    dataMissingNumbers = (pd
                          .DataFrame(data.isna().sum())
                          .rename(columns = {0:'numberOfMissing'})
                          .loc[lambda d: d.numberOfMissing > 0]
                          .assign(missingRate = lambda d: (d.numberOfMissing / data.shape[0]).round(decimals = 2))
                         )
    return dataMissingNumbers

def missingAnalysisPlot(data):
    # Plot the percentage of missing for each column 
    plot_colors = ['hotpink','darkviolet','mediumblue', 'orange', 'green', 'blue', 'red']

    fig, ax = plt.subplots(figsize = (15,5))
    ax.bar(data.index, data.missingRate, width=0.4, color = plot_colors)
    ax.set_xticklabels(labels = data.index, rotation=90)
    ax.set_title('The missing percentage of each column')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Percentage')
    
# Fill missing column with zero for a specified columns
def missingImputationZero(data, col):
    data[col] =data[col].replace(np.nan, 0)
    return data

# Fill missing column with mean for a specified columns
def calculateMean(data_train, target, col):
    """
    This function is created to get the mean of given column(s) according to the target
    """
    mean_0 = data_train[data_train[target] == 0][col].mean()
    mean_1 = data_train[data_train[target] == 1][col].mean()
    return mean_0, mean_1

def missingImputationMean(data_train, data, target, col):
    """
    This function is created to fill data with the related means
    """
    mean_0, mean_1 = calculateMean(data_train, target, col)
    
    data[col] = np.where(data[col].notnull(), 
                                         data[col],
                                         np.where(data[target] == 0,
                                                  mean_0,
                                                  mean_1
                                                 )
                                         )
    return data

# Fill missing column with mode for a specified columns
def calculateMode(data_train, target, col):
    """
    This function is created to get the mode of given column(s) according to the target
    """
    mode = data_train[col].mode().values[0]
    return mode

def missingImputationMode(data_train, data, target, col):
    """
    This function is created to fill data with the related means
    """
    mode = calculateMode(data_train, target, col)
    data[col] =data[col].replace(np.nan, mode)
    return data