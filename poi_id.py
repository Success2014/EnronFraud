#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from pprint import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from feature_engi import addFeatures, selectKFeatures
from remove_outlier import removeOutliers
from tune_parameter import tune_parameter_values

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income',\
                'director_fees', 'exercised_stock_options', 'expenses',\
                'from_messages','from_poi_to_this_person',\
                'from_this_person_to_poi',\
                'loan_advances', 'long_term_incentive', 'other',\
                'restricted_stock', 'restricted_stock_deferred',\
                'salary', 'shared_receipt_with_poi', 'to_messages',\
                'total_payments', 'total_stock_value', 'hasEmail',\
                'fromPoiRatio', 'toPoiRatio']                

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
my_dataset = data_dict
my_dataset = removeOutliers(my_dataset)

print "============================="

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = addFeatures(my_dataset)

print "============================="

### Select k best features
#k = 10 # number of features to choose
#my_features_list = selectKFeatures(my_dataset, features_list, k)

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, my_features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)
#from sklearn import preprocessing
#scaler = preprocessing.MinMaxScaler()
#features = scaler.fit_transform(features)

print "============================="
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



def classifiers(clf_name):
    """
    A variety of classifiers. Available names: "lr", "nb", "rfc", "abc", 
    "knc", "gbc"
    """
    if clf_name == "lr":
        ### Logistic Regression
        from sklearn.linear_model import LogisticRegression
        ### Select k best features
        k = 16 # number of features to choose
        my_features_list = selectKFeatures(my_dataset, features_list, k)
        data = featureFormat(my_dataset, my_features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        
        pipeline_lr = Pipeline(steps = [
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(random_state=42))
        ])
#        scaler = StandardScaler()
#        features = scaler.fit_transform(features)
        
        parameters = {'C':[10**i for i in range(-10,2)]}
        folds = 1
#        clf = LogisticRegression()
        best_param = tune_parameter_values(labels, features, folds, pipeline_lr, 
                      parameters)
        return LogisticRegression(**best_param), my_features_list
    
    if clf_name == "nb":        
        ### Select k best features
        k = 8 # number of features to choose
        my_features_list = selectKFeatures(my_dataset, features_list, k)
        ### Naive Bayes
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(), my_features_list
    
    if clf_name == "rfc":
        ### Random Forest
        from sklearn.ensemble import RandomForestClassifier
        parameters = {'n_estimators': [5, 10, 20, 50, 70, 100, 200, 500, 1000],
                      'max_depth': [2, 4, 8, 10],
                      'min_samples_split': [2, 4, 8, 10]}
        folds = 1
        clf = RandomForestClassifier(random_state = 66)
        best_param = tune_parameter_values(labels, features, folds, clf, 
                      parameters)
        return RandomForestClassifier(**best_param)
    
    if clf_name == "abc":
        ### AdaBoosting
        from sklearn.ensemble import AdaBoostClassifier
        
        parameters = {'n_estimators':[30, 40, 50, 60, 80, 100, 200, 500, 1000],
                      'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1]}
        folds = 1
        clf = AdaBoostClassifier(random_state = 66)
        best_param = tune_parameter_values(labels, features, folds, clf, 
                      parameters)
        return AdaBoostClassifier(**best_param)
    
    if clf_name == "knc":
        ### K Neighbors
        from sklearn.neighbors import KNeighborsClassifier
        parameters = {'n_neighbors':list(range(1,20,1))}
        folds = 1
        clf = KNeighborsClassifier()
        best_param = tune_parameter_values(labels, features, folds, clf, 
                      parameters)
        return KNeighborsClassifier(**best_param)
    
    if clf_name == "gbc":   
        ### Gradient Boosting
        from sklearn.ensemble import GradientBoostingClassifier
        parameters = {'n_estimators':[80, 100, 200, 500, 1000],
                      'max_depth':[1,2,3,4,5,6],
                      'learning_rate':[0.01, 0.05, 0.1],
                      'min_samples_split':[25, 30, 32, 35, 38, 40, 50, 60],
                      'max_leaf_nodes':[2,5,8,9,10,11,20,30]}
        folds = 1
        clf = GradientBoostingClassifier(random_state = 66)
        best_param = tune_parameter_values(labels, features, folds, clf, 
                      parameters)
        best_param['random_state'] = 66
        return GradientBoostingClassifier(**best_param)

#clf, my_features_list = classifiers("lr")
#clf, my_features_list = classifiers("nb")
#clf, my_features_list = classifiers("rfc")
#clf, my_features_list = classifiers("abc")
#clf, my_features_list = classifiers("knc")
#clf, my_features_list = classifiers("gbc")

print "============================="
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html





test_classifier(clf, my_dataset, my_features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, my_features_list)