# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:38:59 2015

@author: Neo
"""

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
    