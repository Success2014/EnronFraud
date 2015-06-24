# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:28:55 2015

Engineer features.

@author: Neo
"""

from sklearn.feature_selection import SelectKBest
from feature_format import featureFormat, targetFeatureSplit


def addFeatures(my_dataset):
    """my_dataset is the original data dictionary.
    Add new features to the dataset."""
    for name in my_dataset.keys():
        features = my_dataset[name]
        
        # New feature: 'hasEmail'. Being 1 if email exists, otherwise 0.
        if features['email_address'] == "NaN":
            features['hasEmail'] = 1 
        else:
            features['hasEmail'] = 0
        
                    
        
        # New feature: 'fromPoiRatio', the ratio of email from poi among 
        # all of his email received. If there is "NaN", set it to 0.
        if features['from_poi_to_this_person'] == "NaN"  or \
            features['to_messages'] == "NaN":
                features['fromPoiRatio'] = 0       
        else:
            features['fromPoiRatio'] = features['from_poi_to_this_person'] / \
                                    float(features['to_messages'])
        
        
        # New feature: 'toPoiRatio', the ratio of email to poi among 
        # all of his email sent. If there is "NaN", set it to 0.
        if features['from_this_person_to_poi'] == "NaN"  or \
            features['from_messages'] == "NaN":
                features['toPoiRatio'] = 0 
        else:
            features['toPoiRatio'] = features['from_this_person_to_poi'] / \
                                    float(features['from_messages'])
        
        # New feature: 'total_money', the sum of total_payments and 
        # total_stock_value
        features['total_money'] = 0
        if features['total_payments'] != "NaN":
            features['total_money'] += features['total_payments']
        if features['total_stock_value'] != "NaN":
            features['total_money'] += features['total_stock_value']
                                    
        
            
        
    print "New feature 'hasEmail' added!"
    print "New feature 'fromPoiRatio' added!"
    print "New feature 'toPoiRatio' added!"
    print "New feature 'total_money' added!"
    return my_dataset


def selectKFeatures(my_dataset, features_list, k):
    """
    select k best features using sklearn SelectKBest function.
    """
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)    
    
    kb = SelectKBest(k = k)
    kb.fit(features, labels)
    scores = kb.scores_
    fea_sco_unsort = zip(features_list[1:],scores) # pair of feature and score
    fea_sco_sorted = list(reversed(sorted(fea_sco_unsort, key=lambda x: x[1])))
    kb_features_list = [y[0] for y in fea_sco_sorted[:k]]
    print "{0} best features selected: {1}".format(k, kb_features_list)
    return ['poi'] + kb_features_list
    
    