# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:33:59 2014

@author: olena
"""
import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

COLORS =["#8533D6","#5C5C8A","#a36e81","#7ba29a","#6600FF","#5C85D6","#006600","#1963D1","#0066FF","#5C5C8A","#6666FF",]


def error(f,x,y):
    return sp.sum((f(x)-y)**2)
    
def plotting1(x,y,y1=None,y2=None,y3=None):
    fig,ax = plt.subplots(1,1)
    ax.plot(x,y,'bo')
    ax.set_xlabel('time')
    ax.set_ylabel('hits/hour')
#     plt.ylabel('hits/hour')
#     plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)])
#     plt.autoscale(tight=True)
#     plt.grid()

    if y1 != None:
        ax.plot(x,y1,color=COLORS[1],linewidth=2)
    if y2 != None:
        ax.plot(x,y2,color=COLORS[2],linewidth=2)  
    if y3 != None:
        ax.plot(x,y3,color=COLORS[4],linewidth=2)  
 
    plt.savefig('plt.jpg')   
    
    
if __name__=='__main__':
    DATA_DIR = '..'+ os.sep + 'data'+ os.sep
    #print os.getcwd()
    df = sp.genfromtxt(DATA_DIR+'web_traffic.tsv',delimiter='\t')
    print df[0:10]
    col1 = df[:,0]
    col2 = df[:,1]
    print sp.sum(sp.isnan(col2))
    print col1.sum()
    print col2.sum()
    print np.mean(col2)
    print np.sum(col2)
    print np.mean(col2[~np.isnan(col2)])
    col1= col1[~np.isnan(col2)]
    col2= col2[~np.isnan(col2)]
    np.sum(np.isnan(col2))
    

    fs=[]
    for i in range(3):
        print "Degree=",i+1
    
        fp = sp.polyfit(col1,col2, i+1)
        print "Model parameters:",fp
        f = sp.poly1d(fp)
        fs.append(f)
        print "Error of the model: %.2g" %error(f,col1,col2)       
    plotting1(col1,col2,y1=fs[0](col1),y2=fs[1](col1),y3=fs[2](col1))
