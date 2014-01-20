'''
Created on 19/01/2014

@author: olena
'''
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import functions_knn
import PLOTS
import functions_SupervisedLearning
from sklearn import linear_model

def load_dataset(fn): 
    data = []
    labels = []
    with open(fn,'r') as ifile:
        for line in ifile:
            tokens = line.strip().split('\t')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def plot1(data,labels,feature_names,label_names):
    
    fig = plt.figure(facecolor='w', edgecolor='b', frameon=True)
    ax = fig.add_subplot(111)
    legend = []
    for t, marker,color in zip(xrange(len(label_names)),'>ox','rgb'):
        l, = ax.plot(data[:,feature_names.index('area')][labels==label_names[t]], 
                   data[:,feature_names.index('compactness')][labels==label_names[t]],
                   marker=marker,c=color,linewidth=0)
        ax.set_ylabel('compactness')
        ax.set_xlabel('area')
        legend.append((l,label_names[t]))
    fig.legend([l for (l,d) in legend],[d for (l,d) in legend],'lower center')
    plt.savefig("wheat_expl.jpg")

if __name__=="__main__":
    DATA_DIR = '..'+ os.sep + 'data'+ os.sep
    fn = 'seeds.tsv'
    data,labels = load_dataset(DATA_DIR+fn)
    print "number of observations",data.shape
    feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficient',
    'length of kernel groove',
    ]
    
    # exploratory analysis
    label_names = np.unique(labels)
    print label_names
    #plot1(data,labels,feature_names,label_names)
    
    print "how many nans are in the dataset?",np.sum(np.isnan(data))
    
    """make a normalized dataset"""
    ndata = data.copy()
    for c in xrange(data.shape[1]):   ndata[:,c] = functions_knn.normalize(data[:,c])
    
    """make numeric target vector"""
    t = []
    for label in labels:
        if label.lower() == 'canadian':
            t.append(0)
        elif label.lower() == 'kama':
            t.append(1)
        else: t.append(2)
    target = np.array(t)
    
    """invoke Knn method"""
    print'\napplying K-nearest-neighbor method'
    pred_accuracy = []
    train_data,train_target,test_data,test_target = functions_knn.determine_train_test_sets(ndata, target,proportion=20)
    for k in xrange(2,15,1): 
        predicted = functions_knn.fitting_knn(train_data, train_target, test_data, num_neighbors = k)
        print predicted
        print test_target
        accuracy = functions_knn.accuracy_eval(predicted,test_target)
        pred_accuracy.append(accuracy)
        print k,':',accuracy
    PLOTS.simple_scatter_plot(np.array([k for k in xrange(2,15,1)]), np.array(pred_accuracy), "number of neighbors (k)",
                              "accuracy","Accuracy with kNN method for different k", "accurKNN")
        
    
    """invoke SVM"""
    print '\napplying SVM'
    pred_accuracy = []
    for kernel in['linear','poly','rbf']:
        predicted = functions_SupervisedLearning.perform_svm(train_data, train_target, test_data, kernel, polydeg=3)
        print predicted
        print test_target
        accuracy = functions_knn.accuracy_eval(predicted,test_target)

        pred_accuracy.append(accuracy)
        print kernel,':',accuracy
    PLOTS.simple_scatter_plot(np.array([k for k in xrange(3)]), np.array(pred_accuracy), "kernels",
                              "accuracy","Accuracy with SVM for different kernels", "accurSVM")
  
    """apply logistic regression .... apply only if there are just two outcomes"""
    t = []
    for label in labels:
        if label.lower() == 'canadian':
            t.append(0)
#         elif label.lower() == 'kama':
#             t.append(1)
        else: t.append(1)
    target = np.array(t)
    print '\napplying Logistic Regression'
    pred_accuracy = []
    predicted = functions_SupervisedLearning.perform_logisticReg(train_data, train_target, test_data)
    print predicted
    print test_target
    accuracy = functions_knn.accuracy_eval(predicted,test_target)
    print accuracy
    #pred_accuracy.append(accuracy)
    