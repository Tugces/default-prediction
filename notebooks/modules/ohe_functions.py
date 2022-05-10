#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  8 11:57:55 2022

@author: tugcecatal
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def runOhe(data, catColumns):
    """
    This function is to created to fit ohe
    """
    # define ohe
    ohe = OneHotEncoder(drop='first')
    
    # apply ohe on train
    data_ohe = ohe.fit_transform(data[catColumns])
    column_names = ohe.get_feature_names(catColumns)
    data_ohe = pd.DataFrame(data_ohe.toarray(), columns = column_names, index = data.index)
    
    return ohe, data_ohe

def oheConcat(data, data_ohe, catColumns):
    """
    This function is to merge ohe set and other set
    """
    dataConcat = (pd
                     .merge(data.drop(columns = catColumns, axis = 1),
                            data_ohe,
                            right_index = True,
                            left_index = True

                           )
                    )
    return dataConcat

def oheTransform(ohe, data, catColumns):
    
    #This function is to created to transform ohe
   
    #ohe, train_ohe = oheTrain(train, catColumns)
    
    # apply ohe on test
    data_ohe = ohe.transform(data[catColumns])
    column_names = ohe.get_feature_names(catColumns)
    data_ohe = pd.DataFrame(data_ohe.toarray(), columns = column_names, index = data.index)
    
    return data_ohe