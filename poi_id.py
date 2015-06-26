#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
#from tester import test_classifier, dump_classifier_and_data
from pprint import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from feature_engi import addFeatures, selectKFeatures
from remove_outlier import removeOutliers
from tune_parameter import tune_parameter_values
from my_tester import mytest_classifier, dump_classifier_and_data

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
        k = 8 # number of features to choose
        my_features_list = selectKFeatures(my_dataset, features_list, k)
        data = featureFormat(my_dataset, my_features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        ### make pipeline
        pipeline_lr = Pipeline(steps = [
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(random_state=66))
        ])
#        scaler = StandardScaler()
#        features = scaler.fit_transform(features)
        ### list parameters to be trained
        parameters = {'classifier__C':[10**i for i in range(-10,2)],
                      'classifier__penalty':['l1','l2']}
        folds = 1
        clf = tune_parameter_values(labels, features, folds, pipeline_lr, 
                      parameters)
                      
        clf = Pipeline(steps = [
                    ('scaler', StandardScaler()),
                    ('classifier', LogisticRegression(C=10**(-10),
                                                      penalty = 'l2',
                                                      random_state=66))
        ])

        return clf, my_features_list
    
    elif clf_name == "nb":        
        ### Select k best features
        k = 8 # number of features to choose
        my_features_list = selectKFeatures(my_dataset, features_list, k)
        ### Naive Bayes
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(), my_features_list
    
    elif clf_name == "rfc":
        ### Random Forest
        from sklearn.ensemble import RandomForestClassifier
        ### Select k best features
        k = 8 # number of features to choose
        my_features_list = selectKFeatures(my_dataset, features_list, k)
        data = featureFormat(my_dataset, my_features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        ### make pipeline
        pipeline_rfc = Pipeline(steps = [
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(random_state=66))
        ])
#        scaler = StandardScaler()
#        features = scaler.fit_transform(features)
        ### list parameters to be trained
        parameters = {'classifier__n_estimators': list(range(5,21)),
                      'classifier__max_features': list(range(2,7,1)),
                      'classifier__max_depth': list(range(2,9,1)),
                      'classifier__min_samples_split': list(range(2,9,1))}
        folds = 1
        clf = tune_parameter_values(labels, features, folds, 
                                           pipeline_rfc, parameters)
        clf = Pipeline(steps = [
                    ('scaler', StandardScaler()),
                    ('classifier', RandomForestClassifier(n_estimators = 5,
                                                          max_features = 2,
                                                          max_depth = 5,
                                                          min_samples_split = 3,
                                                          random_state=66))
        ])                                         
        return clf, my_features_list
    
    elif clf_name == "abc":
        ### AdaBoosting
        from sklearn.ensemble import AdaBoostClassifier
        ### Select k best features
        k = 12 # number of features to choose
        my_features_list = selectKFeatures(my_dataset, features_list, k)
        data = featureFormat(my_dataset, my_features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        ### make pipeline
        pipeline_abc = Pipeline(steps = [
#                    ('scaler', StandardScaler()),
                    ('classifier', AdaBoostClassifier(random_state=66))
        ])
#        scaler = StandardScaler()
#        features = scaler.fit_transform(features)        
        ### list parameters to be trained
        parameters = {'classifier__n_estimators':[100, 120, 140, 160, 200, 300, 500, 1000,2000],
                      'classifier__learning_rate': [0.01, 0.05, 0.08, 0.09, 
                                                    0.1, 0.11, 0.12, 0.2, 0.5]}
        folds = 1
        clf = tune_parameter_values(labels, features, folds, 
                                    pipeline_abc, parameters)
        clf = Pipeline(steps = [
#                    ('scaler', StandardScaler()),
                    ('classifier', AdaBoostClassifier(n_estimators = 1000,
                                                      learning_rate = 0.05,
                                                      random_state = 66))
        ])                             
        return clf, my_features_list
    
    elif clf_name == "knc":
        ### K Neighbors
        from sklearn.neighbors import KNeighborsClassifier
        ### Select k best features
        k = 8 # number of features to choose
        my_features_list = selectKFeatures(my_dataset, features_list, k)
        data = featureFormat(my_dataset, my_features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        ### make pipeline
        pipeline_knc = Pipeline(steps = [
                    ('scaler', StandardScaler()),
                    ('classifier', KNeighborsClassifier())
        ])
#        scaler = StandardScaler()
#        features = scaler.fit_transform(features)        
        ### list parameters to be trained
        parameters = {'classifier__n_neighbors':list(range(1,6,1))}
        folds = 1
        clf = tune_parameter_values(labels, features, folds, 
                                           pipeline_knc, parameters)
        clf = Pipeline(steps = [
                    ('scaler', StandardScaler()),
                    ('classifier', KNeighborsClassifier(n_neighbors = 3))
        ])                                         
        return clf, my_features_list
    elif clf_name == "svc":        
        ### support vector classifier
        from sklearn.svm import SVC
        ### select k best features
        k = 16
        my_features_list = selectKFeatures(my_dataset, features_list, k)
        data = featureFormat(my_dataset, my_features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        ### make pipeline
        pipeline_svc = Pipeline(steps = [
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(random_state=66))
        ])
#        scaler = StandardScaler()
#        features = scaler.fit_transform(features)         
        ### list parameters to be trained
        parameters = {'classifier__C':[1,10,100],
                      'classifier__kernel':['linear','poly','rbf','sigmoid'],
                      'classifier__gamma':[0.05,0.06,0.07,0.08,0.09,0.1,0.11]}
        folds = 1
        clf = tune_parameter_values(labels, features, folds, 
                                           pipeline_svc, parameters)
        clf = Pipeline(steps = [
                    ('scaler', StandardScaler()),
                    ('classifier', SVC(C=100, kernel='poly',gamma=0.1,
                                       random_state=66))
        ])                                         
        return clf, my_features_list
    
    
#    elif clf_name == "gbc":   
#        ### Gradient Boosting
#        ### training time could be more than 4 hours
#        from sklearn.ensemble import GradientBoostingClassifier
#        parameters = {'n_estimators':[80, 100, 200, 500, 1000],
#                      'max_depth':[1,2,3,4,5,6],
#                      'learning_rate':[0.01, 0.05, 0.1],
#                      'min_samples_split':[25, 30, 32, 35, 38, 40, 50, 60],
#                      'max_leaf_nodes':[2,5,8,9,10,11,20,30]}
#        folds = 1
#        clf = GradientBoostingClassifier(random_state = 66)
#        best_param = tune_parameter_values(labels, features, folds, clf, 
#                      parameters)
#        best_param['random_state'] = 66
#        return GradientBoostingClassifier(**best_param)
    else:
        raise ValueError("Invalid Classifier")


### uncomment the classifier you want to try
### and comment out the others

clf, my_features_list = classifiers("lr")
#clf, my_features_list = classifiers("nb")
#clf, my_features_list = classifiers("rfc")
#clf, my_features_list = classifiers("abc")
#clf, my_features_list = classifiers("knc")
#clf, my_features_list = classifiers("svc")

print "============================="
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html




mytest_classifier(clf, my_dataset, my_features_list)
#test_classifier(clf, my_dataset, my_features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, my_features_list)