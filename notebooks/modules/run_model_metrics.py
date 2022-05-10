#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:28:38 2022

@author: tugcecatal
"""

from sklearn.metrics import accuracy_score,plot_roc_curve, confusion_matrix, precision_score, recall_score, f1_score, classification_report

import warnings
warnings.filterwarnings('ignore')

def metrics(model_name, scores_train, pred_train, y_train, scores_test, pred_test, y_test):
    #train
    train_accuracy = accuracy_score(y_train, pred_train)
    train_precision_score = precision_score(y_train, pred_train)
    train_recall_score = recall_score(y_train, pred_train)
    train_f1_score = f1_score(y_train, pred_train)
    print("\nMetrics for Train Data:") 
    print("Cross-validation score: ", scores_train) 
    print("Confusion matrix: \n", confusion_matrix(y_train, pred_train))
    print("precision:{:.2f}".format(train_precision_score))
    print("recall:  {:.2f} ".format(train_recall_score))
    print("f1 score: {:.2f} ".format(train_f1_score))
    print(classification_report(y_train, pred_train))

    #test
    test_accuracy = accuracy_score(y_test, pred_test)
    test_precision_score = precision_score(y_test, pred_test)
    test_recall_score = recall_score(y_test, pred_test)
    test_f1_score = f1_score(y_test, pred_test)
    print("\nMetrics for Test Data:") 
    print("Cross-validation score: ", scores_test) 
    print("Confusion matrix: \n", confusion_matrix(y_test, pred_test))
    print("precision:{:.2f}".format(test_precision_score))
    print("recall:  {:.2f} ".format(test_recall_score))
    print("f1 score: {:.2f} ".format(test_f1_score))
    print(classification_report(y_test, pred_test))
    #skplt.metrics.plot_roc_curve(y_test, preds_proba)
    
    model_name2 = model_name
    metrics_dict = {"model": model_name2,
                    "train_accuracy": train_accuracy,
                    "train_precision_score": train_precision_score,
                    "train_recall_score": train_recall_score,
                    "train_f1_score": train_f1_score,
                    "test_accuracy": test_accuracy,
                    "test_precision_score": test_precision_score,
                    "test_recall_score": test_recall_score,
                    "test_f1_score": test_f1_score
                    }    
    return metrics_dict
    
def run_pipeline(model_name, pipeline, X_train, y_train, X_test, y_test):
    #fit the model
    pipeline.fit(X_train, y_train)
    #run the model for the train set
    pred_train = pipeline.predict(X_train)
    scores_train = pipeline.score(X_train, y_train)
    #run the model for the test set
    pred_test = pipeline.predict(X_test)
    scores_test = pipeline.score(X_test, y_test)
    #get predictions for ROC AUC curve
    preds_proba = pipeline.predict_proba(X_test)
    plot_roc_curve(pipeline, X_test, y_test)
    
    metrics_dict = metrics(model_name, scores_train, pred_train, y_train, scores_test, pred_test, y_test)
    #scores_train, pred_train, scores_test, pred_test, preds_proba
    return metrics_dict, pred_train, pred_test, preds_proba
