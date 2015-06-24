# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 16:22:29 2015

Tune the parameters for different classifiers.


@author: Neo
"""
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV
from time import time
from sklearn.metrics import accuracy_score, precision_score, recall_score,\
f1_score


def tune_parameter_values(labels, features, folds, pipe_line, 
                          parameters):
    """
    Get the optimal values for the parameters from grid search based on the
    score fucntion
    """
    
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    for train_idx, test_idx in cv: 
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
    
    
    clf = GridSearchCV(pipe_line, parameters)
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
    return param_val_final
    
    
    