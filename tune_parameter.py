# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:22:29 2015

Tune the parameters for different classifiers.

Note:
Valid options of GridSearchCV scoring are:
['accuracy', 'adjusted_rand_score', 'average_precision', 
'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 
'log_loss', 'mean_absolute_error', 'mean_squared_error', 
'median_absolute_error', 'precision', 'precision_macro', 
'precision_micro', 'precision_samples', 'precision_weighted', 
'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 
'recall_weighted', 'roc_auc']



@author: Neo
"""
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score, make_scorer


def my_score():
    return make_scorer(recall_score, precision_score)



def tune_parameter_values(labels, features, folds, pipe_line, 
                          parameters):
    """
    Get the optimal values for the parameters from grid search based on the
    score fucntion
    """
    
    cvss = StratifiedShuffleSplit(labels, folds, random_state = 42)
    for train_idx, test_idx in cvss: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
    
    
    clf = GridSearchCV(pipe_line, parameters, cv=3, scoring = my_score())
    t0 = time()
    print "Grid Searching ......"
    clf.fit(features_train, labels_train)    
    print "Grid Searching finished in %.3f seconds"  % (time() - t0)
    print "Best Score: %.3f" % clf.best_score_
    print "Best Parameters:"
    best_param_val = clf.best_estimator_.get_params()
    param_val_final = {}
    for param in parameters.keys():
        print "\t{0}: {1}".format(param, best_param_val[param])
        param_val_final[param] = best_param_val[param]
    return clf

    
    
    