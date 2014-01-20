'''
Created on 20/01/2014

@author: olena
'''
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import linear_model

def perform_svm(train_data,train_target,test_data,kernel,polydeg=3):
    if kernel == 'poly':
        svc = svm.SVC(kernel=kernel,degree=polydeg)
    else:
        svc = svm.SVC(kernel=kernel)
    svc.fit(train_data,train_target)
    return svc.predict(test_data)

def perform_logisticReg(train_data,train_target,test_data):
    log = linear_model.LogisticRegression(C=1e5)
    log.fit(train_data, train_target)
    return log.predict(test_data)

