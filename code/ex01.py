import os
import numpy as np
import pandas
import urllib 
import matplotlib.pyplot as plt
from pylab import *
import statsmodels.api as sm
#from statsmodels.graphics.api import interaction_plot, abline_plot

import statsmodels
from scipy import stats

COLORS =["#8533D6","#5C5C8A","#a36e81","#7ba29a","#6600FF","#5C85D6","#006600","#1963D1","#0066FF","#5C5C8A","#6666FF",]

def read_data(url,fn,DATA_DIR):
    try:
        df = pandas.read_csv(DATA_DIR+fn)
    except:
        url = urllib.urlopen(url+fn)
        df = pandas.read_table(url, delimiter=",")
    return df
    
def format_histogram(ax,title=None,xlab="",ylab="",index=0):
    if title != None:
        ax.set_title(title,color=COLORS[index],fontsize=9,fontweight='bold')
    ax.set_xlabel(xlab,fontsize=9)
    ax.set_ylabel(ylab,fontsize=9)
    ax.tick_params(labelsize=9)                 
  
def categorization_example():
    url = "http://stat.columbia.edu/~rachel/datasets/"
    fn = "nyt1.csv"
    DATA_DIR = '.'+os.sep+"data01"+os.sep
    df = read_data(url,fn,DATA_DIR)
    print df.columns
    ages = [0,18,25,35,45,55,65,1000]
    df["age_group"] = pandas.Series(['' for x in df.index], index = df.index)
    newind = []
    for g in range(1,len(ages)):
        if g ==1:
            res_str = "<"+str(ages[g])
        elif g == len(ages)-1:
            res_str = str(ages[g-1])+'>'
        else:
            res_str = str(ages[g-1])+"-"+str(ages[g]-1)
        newind.append(res_str)
        df['age_group'][(df['Age']>=ages[g-1] ) & ( df['Age']<ages[g])] = res_str

    ser = {}
    ser['countAll'] = pandas.Series([0 for x in newind],index = newind)
    ser['count_G0'] = pandas.Series([0 for x in newind],index = newind)
    ser['count_G1'] = pandas.Series([0 for x in newind],index = newind)
    
    for i in range(len(newind)):
        ser['countAll'][i] = df['age_group'][df['age_group']==newind[i]].count()
        for G in [0,1]:
            ser['count_G'+str(G)][i] = df['age_group'][(df['age_group']==newind[i]) & (df['Gender']==G)].count() 

    for col in ['Clicks','Impressions','Signed_In']:
        ser[col] = pandas.Series([0 for x in newind],index = newind)
        ser[col+'_G0']= pandas.Series([0 for x in newind],index = newind)
        ser[col+'_G1']= pandas.Series([0 for x in newind],index = newind)
        for i in range(len(newind)):
            ser[col][i] = df[col][df['age_group']==newind[i]].sum()
            for G in [0,1]:
                ser[col+'_G'+str(G)][i] = df[col][(df['age_group']==newind[i]) & (df['Gender']==G)].sum()    
        
        
    newdf = pandas.DataFrame(ser,index=newind)
    newdf['CTR'] = pandas.Series([0.0 for x in newdf.index],index=newdf.index)
    newdf['CTR_G0'] = pandas.Series([0.0 for x in newdf.index],index=newdf.index)
    newdf['CTR_G1'] = pandas.Series([0.0 for x in newdf.index],index=newdf.index)
    for i in range(len(newdf.index)):
        newdf['CTR'][i] = float(newdf['Clicks'][i])/float(newdf['Impressions'][i])
        for G in [0,1]:
            newdf['CTR_G'+str(G)][i] = float(newdf['Clicks_G'+str(G)][i])/float(newdf['Impressions_G'+str(G)][i])
        
    fig_fn ='Hist.jpg'    
    VARS = ['count_G1','count_G0','CTR_G0','CTR_G1']
    fig,axes = plt.subplots(nrows=len(VARS),ncols=1,sharex=True,sharey=False)
    fig.subplots_adjust(hspace=0.15)  
    for v in range(len(VARS)):
        ax = axes[v]
        newdf[VARS[v]].plot(kind='bar',ax=ax,color=COLORS[v])
        format_histogram(ax,ylab=VARS[v], index=v)
   
    if fig_fn != None:
        plt.savefig(fig_fn)  
    else:
        plt.show()     
     
def categorical_var_plots(DATA_DIR): 
    fn = 'lgtrans.csv'
    df = pandas.read_csv(DATA_DIR+fn) 
    
#     #plotting  
#     fig,axes = plt.subplots(nrows=1,ncols=1,sharex=False,sharey=False)
#     fig.subplots_adjust(hspace=0.15) 
#     fig_fn = fn.split('.')[0]+".jpg"
#     #axes.plot(df['read'],df['write'],'ro') 
#     ax = axes#[0]
#     ax.plot(df['read'][df['female']=='female'],df['write'][df['female']=='female'],'ro') 
#     ax.plot(df['read'][df['female']=='male'],df['write'][df['female']=='male'],'bo') 
#     ax.set_xlabel('read')
#     ax.set_ylabel('write')
# 
# #     ax = axes[1]
# #     ax.plot(np.log(df['read']),np.log(df['write']),'ro') 
# #     ax.set_xlabel('lgread')
# #     ax.set_ylabel('lgwrite')
# 
#     if fig_fn != None:
#         plt.savefig(fig_fn)  
#     else:
#         plt.show()     
 
    #attempt logistic model   
    do_logit(df,DFname=fn.split('.')[0]+"asLog")   
#     ndf = df[(df['read']<45) | (df['read']> 55)] 
#     do_logit(ndf,DFname=fn.split('.')[0]+"asLogonlyOnSubset")   
    
    #OLS
#     print "OLS on the original"
#     do_ols(df,DFname=fn.split('.')[0]+'ALL')
#     print "OLS for Females only"
#     fdf = df[df['female']=='female']
#     do_ols(fdf,DFname=fn.split('.')[0]+'Female')
#     print "OLS for Males only"
#     fdf = df[df['female']=='male']
#     do_ols(fdf,DFname=fn.split('.')[0]+'Male')
#     print "OLS after log transformation"
#     fdf = df.copy(deep=True) 
#     fdf['read'] = np.log(fdf['read'])
#     fdf['write'] = np.log(fdf['write'])
#     do_ols(fdf,DFname=fn.split('.')[0]+'log')

def do_logit(df,DFname=''):
    df['write>50'] = 0
    df['write>50'][df['write']>50] = 1
    df['const'] = 1
    
    yName = 'write>50'
    x_vars = ['const','read','math']
    mod = sm.Logit(df[yName],df[x_vars]).fit()  
 
    print "\n",mod.summary()    
    if mod.llr_pvalue < 0.05 : print " model holds water" 
    else: print "the model is not predicting well"
    prob_plot(mod,df,yName,x_vars,Gname=DFname,out_dir='')
    
    tprA = []
    fprA = []
    
    for th in range(20,90,5):#[0.4,0.45,0.5,0.6, 0.7]:
        thR = float(th)/100.0
        print 'Threshold:',thR
        tpr,fpr = do_ROC(df,thR,mod)
        tprA.append(tpr)
        fprA.append(fpr)
    y = np.array(tprA)
    x = np.array(fprA)
    print y
    print x
    fig,ax1 = plt.subplots(1,1)
    ax1.plot(x,y,'go')
    plt.show()
    
    
def do_ROC(df,th,model):
    df['pred'] = model.predict()  
    TP = df['pred'][(df['pred']>= th) & (df['write>50'] == 1)].count()
    RP = df['write>50'][df['write>50'] ==1].count()
    tpr = float(TP)/float(RP)  
    print 'TP =',TP, 'RP=',RP,'tpr=',tpr
    
    FP = df['pred'][(df['pred']>= th) & (df['write>50'] == 0)].count()
    RN = df['write>50'][df['write>50'] ==0].count()
    fpr = float(FP)/float(RN)  
    print 'FP =',FP, 'RN=',RN,'fpr=',fpr
    return tpr,fpr


def prob_plot(model, data,yName,xNames,Gname='',out_dir=''):
    fig, ax = plt.subplots()
    ax.plot(data[yName],model.predict(),'o')
    ax.set_xlabel(yName)
    ax.set_ylabel('Predicted '+yName)
    ax.set_title(Gname+" Predicted vs true "+yName+"\n "+",  ".join(xNames),fontsize=9)
    plt.savefig(out_dir+"LogReg_ProbPlot"+Gname+".jpg")   
     

def do_ols(df,DFname=''):
    df['const'] = 1
    x_vars = ['const','read','math']#,'female']
    mod = sm.OLS(df['write'],df[x_vars]).fit()
    print mod.summary()
    plot_resid_vs_true(mod,df,'write',Gname=DFname)
    plot_predicted_vs_true(mod,df,'write',Gname=DFname)
    
    
def plot_resid_vs_true(model,data,yName,out_dir='',Gname=''):
    fig, ax = plt.subplots()
    ax.plot(data[yName],model.resid,'g.')
    ax.set_title(Gname+": Residuals vs True "+yName)
    ax.set_ylabel("residual size")
    ax.set_xlabel(yName)
    plt.savefig(out_dir+"Resid_vs_True_"+Gname+".jpg")

def plot_predicted_vs_true(model,data,yName,out_dir='',Gname=''):
    fig, ax = plt.subplots()  
    ax.plot(data[yName],model.predict(),'o')
    ax.set_xlabel(yName)
    ax.set_ylabel('Predicted '+yName)
    ax.set_title(Gname+": Predicted vs true "+yName)
    plt.savefig(out_dir+"pred_vs_true_"+Gname+".jpg")        
    
            
if __name__=="__main__":
    DATA_DIR = '.'+os.sep+'data01'+os.sep
    categorical_var_plots(DATA_DIR)
    #categorization_example()
       
 