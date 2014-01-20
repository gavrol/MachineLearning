'''
Created on 19/01/2014

@author: olena
'''

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def normalize(vec):
    min_ = np.min(vec)
    max_ = np.max(vec)
    mean_ = np.mean(vec)
    #print min_,max_,mean_
    n_vec = [(v-min_)/(max_-min_) for v in vec]
    #print np.max(n_vec),np.min(n_vec)
    return n_vec

def determine_train_test_sets(data,target,proportion=10):
    #split data into train and test sets
    np.random.seed(0)
    indices =np.random.permutation(len(data))
    train_data = data[indices[:-proportion]]
    train_target = target[indices[:-proportion]]
    test_data = data[indices[-proportion:]]
    test_target = target[indices[-proportion:]]
    
#     print 'size of training set:',len(train_data),'?=',len(train_target)
#     print 'size of testing set:',len(test_data),'?=',len(test_target)
    return train_data,train_target,test_data,test_target

def fitting_knn(train_data,train_target,test_data,num_neighbors=5):

    knn = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn.fit(train_data,train_target)
    return knn.predict(test_data)

def accuracy_eval(pred_target,test_target):
    return np.mean(pred_target == test_target)
        