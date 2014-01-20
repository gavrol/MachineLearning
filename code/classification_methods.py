'''
Created on 19/01/2014

@author: olena
'''
import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


COLORS =["#8533D6","#5C5C8A","#a36e81","#7ba29a","#6600FF","#5C85D6","#006600","#1963D1","#0066FF","#5C5C8A","#6666FF",]


def plots1(data):    
    legend = []
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    fig = plt.figure(facecolor='w', edgecolor='b', frameon=True)#,axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False)
    fig.subplots_adjust(hspace=0.25,wspace=0.35,bottom=0.2)  
    for p in xrange(len(pairs)): 
            ax = fig.add_subplot(2,3,p)
            legend = []
            for t, marker,color in zip(xrange(3),'>ox','rgb'):
                l, = ax.plot(data['data'][data['target']==t,pairs[p][0]],
                        data['data'][data['target']==t,pairs[p][1]],marker=marker,c=color,linewidth=0)
                ax.set_ylabel(data['feature_names'][pairs[p][1]])
                ax.set_xlabel(data['feature_names'][pairs[p][0]])
                legend.append((l,data['target_names'][t]))
    fig.legend([l for (l,d) in legend],[d for (l,d) in legend],'lower center')
    plt.savefig("iris_expl.jpg")
    
     

if __name__ == '__main__':
    data = load_iris()
    print data.keys()
    print data['target_names']
    print data['feature_names']

    #plots1(data)
    
    #explorations for separation exercise
    plength = data['data'][:,2] #petal length is the third on 
    is_setosa = (data['target_names'] == 'setosa')
    
    ns_features = data['data'][~is_setosa]
    ns_labels = data['target_names'][~is_setosa]
    is_virginica = (data['target_names'] == 'virginica')
        
