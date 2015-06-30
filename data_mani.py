# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:38:59 2015

Various operations on the dataset.

@author: Neo
"""
import matplotlib.pyplot as plt
from feature_format import featureFormat, targetFeatureSplit

def show_data(data_dict, s, feature_x, feature_y):
    """Plot the data points using feature total_payments and from_messages"""
    features_list = ['poi'] + [feature_x] + [feature_y]
    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    x = map(lambda x: x[0], features)
    y = map(lambda x: x[1], features)
    z = ['blue'] * len(labels)
    for i, label in enumerate(labels):
        if label > 0.5:
            z[i] = "red"
    plt.figure()
    plt.scatter(x,y,c=z)
    plt.xlabel(feature_x + ' of this employee')
    plt.ylabel(feature_y + ' of this employee')
    plt.title(s)
    ### Notice that SHAPIRO RICHARD S has to_messages 15149, KEAN STEVEN J has
    ### to_messages 12754. But they are not considered to be outliers.


def data_explore(data_dict):
    """Explore the most important characteristics of the dataset."""

    num_person = len(data_dict)
    print "The total number of data points is {0}.".format(num_person)
    
    num_poi = 0
    num_quantified_salary = 0
    num_email = 0
    num_totalpayment = 0
    num_loan_advances = 0
    num_exercised_stock_options = 0
    num_from_messages = 0
    num_to_messages = 0
    for key, value in data_dict.items():
        num_features = len(value)
        if value['poi']:
            num_poi += 1
        if value['total_payments'] != "NaN":
            num_totalpayment += 1
        if value['salary'] != "NaN":
            num_quantified_salary += 1
        if value['email_address'] != "NaN":
            num_email += 1
        if value['loan_advances'] != "NaN":
            num_loan_advances +=1
        if value['exercised_stock_options'] != "NaN":
            num_exercised_stock_options +=1
        if value['from_messages'] != "NaN":
            num_from_messages +=1
        if value['to_messages'] != "NaN":
            num_to_messages +=1
    
    print "There are {0} POI and {1} non-POI.".format(num_poi, num_person-num_poi)
    print "The number of features is {0}.".format(num_features)
    print "There are {0} missing values in feature total_payments.".format\
            (num_person - num_totalpayment)
    print "There are {0} missing values in feature salary.".format\
            (num_person - num_quantified_salary)
    print "There are {0} missing values in feature email_address.".format\
            (num_person - num_email)
    print "There are {0} missing values in feature loan_advances.".format\
            (num_person - num_loan_advances)
    print "There are {0} missing values in feature exercised_stock_options.".format\
            (num_person - num_exercised_stock_options)
    print "There are {0} missing values in feature from_messages.".format\
            (num_person - num_from_messages)
    print "There are {0} missing values in feature to_messages.".format\
            (num_person - num_to_messages)










def removeOutliers(my_dataset):
    """
    Usually, a person at least has first, middle and last names.
    Although some may not have middle name and some may have suffix, 
    a person's name should be 2 to 4 words. We should pay attention
    to someone whose name length is not in that range.
    """
    for name in my_dataset.keys():
        if len(name.split(" ")) < 2 or len(name.split(" ")) > 4:
            print "Possible Outliers: {0}".format(name)
    
    outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']

    for outlier in outliers:
        try:
            my_dataset.pop(outlier)
            print "Outlier {0} removed!".format(outlier)
        except:
            raise ValueError("No such person in this dataset")
    
    return my_dataset
    