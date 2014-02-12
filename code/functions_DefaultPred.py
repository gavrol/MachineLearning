'''
Created on 02/02/2014

@author: olena

Purpose:  contains various functions necessary for cleaning data for Default predictions.

'''
import os
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import linear_model
import datetime
import NN
from sklearn import feature_selection #import RFE
from sklearn import datasets #import make_classification

def select_allDefaulted_n_someNONdefaulted(df,prop4NonDefault=10):
    """allows to select a all Defauted and 10% nondefaulted"""
    df['tmp'] = 0
    df['tmp'][df['loss'] > 0] = 1
    count = 1
    for i in df[df['tmp'] == 0].index:
        count += 1
        if count%prop4NonDefault == 0:
            df['tmp'][i] = 1
    
    df = df[df['tmp']==1]
    df = df.drop(['tmp'],1) 
    return df    
    
def make_train_test_subsets(df,proportion=80):
    print "\n making a test and train subsets"
    loss_counts = {}
    df['train'] = 0
    for i in df.index:
        if str(df['loss'][i]) not in loss_counts:
            loss_counts.setdefault(str(df['loss'][i]),{'count': 0,'indices':[]})
        loss_counts[str(df['loss'][i])]['count'] +=1
        loss_counts[str(df['loss'][i])]['indices'].append(i)

    for loss in loss_counts.keys():
        sample_size = float(loss_counts[loss]['count'])*float(proportion)/100.0
        count = 0
        for ind in loss_counts[loss]['indices']:
            if count <= sample_size:
                df['train'][ind] = 1
                count += 1
            else:
                break
    print "number of observations in train is:",df['train'].sum()
    return df
    


def normalize(vec):
    """i think there is an sp.linalg.norm function, but for some reason it's not working for me"""
    min_ = np.min(vec)
    max_ = np.max(vec)
    #print min_,max_,mean_
    n_vec = [(v-min_)/(max_-min_) for v in vec]
    #print np.max(n_vec),np.min(n_vec)
    return n_vec


def response_options(df):
    """this function does a few conversions"""
    print "\n \tPlaying with Response variables"
    df["hasdefaulted"] = 1
    df['hasdefaulted'][df['loss']== 0] = 0   
    print "number of loans that did not default",df['loss'][df['loss'] == 0].count() 
    print "number of loans that defaulted",df['loss'][df['loss'] > 0].count()   
    #print "now differently: num of defaults =", df["hasdefaulted"].sum()
    #print "now differently: num of no defaults =", len(df.index) - df["hasdefaulted"].sum()   
    print "added hasdefaulted 0/1 var for logisticReg and other purposes"
    
    df["lossCategory"] = 0
    df["lossCategory"][(df["loss"]>0) &(df["loss"]<= 5)] = 3
    df["lossCategory"][(df["loss"]>0) &(df["loss"]<= 5)] = 3
    df["lossCategory"][(df["loss"]>5) &(df["loss"]<= 8)] = 7
    df["lossCategory"][(df["loss"]>8) &(df["loss"]<= 12)] = 10
    df["lossCategory"][(df["loss"]>12) &(df["loss"]<= 17)] = 15
    df["lossCategory"][(df["loss"]>17) &(df["loss"]<= 25)] = 20
    df["lossCategory"][(df["loss"]>25) &(df["loss"]<= 40)] = 30
    df["lossCategory"][(df["loss"]>41) &(df["loss"]<= 65)] = 50 
    df["lossCategory"][(df["loss"]>65)] = 80 
    return df  

def print_cols(columns):
    for i in range(len(columns)):
        print columns[i],
        if i % 20 == 0.0:
            print'\n'
            
######################### auxiliary functions for predictions #########            
def make_data_4scikit_functions(columns,train_df,test_df,target_name,normalizeInput=True):
    nps = np.array([])
 
    for c in range(len(columns)):
        if normalizeInput:
            norm_vec = normalize(train_df[columns[c]])
        else:
            norm_vec = train_df[columns[c]]
        nps = np.append(nps,norm_vec) #train_df[columns[c]])
    nps = nps.reshape((len(columns),len(train_df.index)))
    train_data = nps.transpose()
    train_target = np.array(train_df[target_name])
    #print train_data.shape
    #print train_target.shape
    
    nps = np.array([])
    for c in range(len(columns)):
        if normalizeInput:
            norm_vec = normalize(test_df[columns[c]])
        else:
            norm_vec = test_df[columns[c]]
        nps = np.append(nps,norm_vec) #test_df[columns[c]])
    nps = nps.reshape((len(columns),len(test_df.index)))
    test_data = nps.transpose()   
    try: 
        test_target = np.array(test_df[target_name])     
    except:
        print "!!! CAUTION:",target_name,"does not exist for test data...okay for final prediction"
        test_target = None
        
    return train_data,train_target,test_data,test_target   

               
##########  predictive functions     ###########################
def apply_LinearRegression(train_df,test_df,loss_bound=20,dep_variable="loss"):
    """the loss_bound allows to determine for which loss values LinReg is to be applied.
    this is because the loss has a very uneven distribution"""
    
    print '\n applying Linear Regression method on response var:',dep_variable
    trainLR_df = train_df[train_df[dep_variable] >=loss_bound]
    testLR_df = test_df[test_df[dep_variable] >=loss_bound]

    columns = valid_columns(train_df)
    active_var = columns
    train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,trainLR_df,testLR_df,dep_variable,normalizeInput=True)
    mse_test_data,fit_score,mse_train_data =  perform_LinearReg(train_data,train_target,test_data,test_target,normalizeInput=False)                                                                                              
    print "\n Linear Regression: MSE(testdata) =",mse_test_data,"R2=",fit_score,"MSE(traindata)=",mse_train_data

    ofn = open("LinReg_log_"+datetime.datetime.now().strftime("%d%m-%H-%M")+".csv","w")  
    ofn.write("Variables,MSE(TestData),FitScore,MSE(TrainData)\n")
    print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
    ResultsD = {}
    ResultsD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
    vars_mse = "+".join(active_var)
    vars_score = "+".join(active_var)
    min_mse = 100000000 #mse_test_data
    max_score = 0#fit_score 0 

    for step in [1,3,5,10,15]:
        for i in range(0,len(columns),step):
            active_var = columns[i:i+step]
            train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,trainLR_df,testLR_df,dep_variable,normalizeInput=True)
            mse_test_data,fit_score,mse_train_data =  perform_LinearReg(train_data,train_target,test_data,test_target,normalizeInput=False)                                                                                              
            ResultsD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
            print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
            if min_mse > mse_test_data:
                    min_mse = mse_test_data
                    vars_mse = "+".join(active_var)
            if fit_score > max_score:
                    max_score = fit_score
                    vars_score = "+".join(active_var)     


    active_var = most_occurring_vars(ResultsD)    
    train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,trainLR_df,testLR_df,dep_variable,normalizeInput=True)
    mse_test_data,fit_score,mse_train_data =  perform_LinearReg(train_data,train_target,test_data,test_target,normalizeInput=False)                                                                                              
    ResultsD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
    print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
    if min_mse > mse_test_data:
            min_mse = mse_test_data
            vars_mse = "+".join(active_var)
    if fit_score > max_score:
            max_score = fit_score
            vars_score = "+".join(active_var)     
    
#     for c1 in range(len(columns)):
#         for c2 in range(c1+1,len(columns)):                       
#             for c3 in range(c2+1,len(columns)):
#                 for c4 in range(c3+1,len(columns)):
#                     for c5 in range(c4+1,len(columns)):
#                         active_var = [columns[c1],columns[c2],columns[c3],columns[c4],columns[c5]]
#                         train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,trainLR_df,testLR_df,dep_variable,normalizeInput=True)
#                         mse_test_data,fit_score,mse_train_data =  perform_LinearReg(train_data,train_target,test_data,test_target,normalizeInput=False)                                                                                              
# #                        ResultsD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
#                         print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
#                         if min_mse > mse_test_data:
#                             min_mse = mse_test_data
#                             vars_mse = "+".join(active_var)
#                         if fit_score > max_score:
#                             max_score = fit_score
#                             vars_score = "+".join(active_var)
    ofn.close()                          
    print "best accuracy achieved with",vars_mse
    print "accuracy=",min_mse
     
    print "best model fit achieved with",vars_score
    print "model fit =", max_score


def MSE(predicted_vals,true_vals):
    a = predicted_vals - true_vals
    mse = np.dot(a,a)/float(len(a))
    return mse

def AbsError(predicted_vals,true_vals):
    v = np.abs(predicted_vals - true_vals)
    return np.mean(v)
    
def perform_LinearReg(train_data,train_target,test_data,test_target,normalizeInput=True,plot=False):
    mod = linear_model.LinearRegression(fit_intercept=True, normalize=normalizeInput)           
    mod.fit(train_data,train_target,n_jobs=-1)
    predicted_values_train = mod.predict(train_data)
    mse_train_data = AbsError(predicted_values_train,train_target)
    predicted_values_test = mod.predict(test_data)
    mse_test_data = AbsError(predicted_values_test,test_target)
    fit_score = mod.score(train_data,train_target)
    if plot:
        make_simple_scatter(test_target,predicted_values_test,"True vs Predicted for TestData") 
        make_simple_scatter(train_target,predicted_values_train,"True vs Predicted for TrainData")         
    return mse_test_data,fit_score,mse_train_data
    

### Logistic Regression
def most_occurring_vars(RegResultsD,toupleDictKey=False):
    #{"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
    #ResultsD[(k,"+".join(active_var))] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
    
    scores = [(RegResultsD[key]['score'],key) for key in RegResultsD.keys()]
    scores.sort()
    scores.reverse()

    cols = {}
    single_var = []
    for (score,var) in scores[0: 15]:
        if toupleDictKey:
            columns = var[1].split("+")
        else:  columns = var.split("+")
        for key in columns:
            if key not in cols.keys():
                cols[key] = 1
            else:
                cols[key] += 1
        
        if toupleDictKey:  
            if var[1].find("+") < 0: #i.e., the var is significant by itself
                single_var.append(var[1])       
        else:
            if var.find("+") < 0: #i.e., the var is significant by itself
                single_var.append(var)      
             
    
    best_scores = [(cols[key],key) for key in cols.keys()]
    best_scores.sort()
    best_scores.reverse()
    keyVar = []
    for (score,var) in best_scores[0:7]:
        keyVar.append(var)
    for var in single_var:
        if var not in keyVar:
            keyVar.append(var)
    print best_scores[:10]
    return keyVar
    
    
def MSE_log(predicted_vals,true_vals):
    return np.mean(predicted_vals == true_vals)


def apply_LogisticReg_4prediction(trainLogR_df,testLogR_df,loss_bound=1,rfe=True):
    """the loss_bound allows to determine for which loss values are to be 0 and which 1;
    this is because the loss has a very uneven distribution;
    E.g., if loss_bound is 1, then this is a straight forward log reg where no-defaulters have 0 and defaulters =1"""
    print '\n applying Logistic Reg method with loss bound=',loss_bound
    trainLogR_df["hasdefaulted20"] = 1
    trainLogR_df["hasdefaulted20"][trainLogR_df["loss"] <loss_bound] = 0
    testLogR_df["hasdefaulted20"] = 1    
    testLogR_df["hasdefaulted20"][testLogR_df["loss"] <loss_bound] = 0 
     
    columns = valid_columns(trainLogR_df)   
    active_var = columns
    train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,trainLogR_df,testLogR_df, "hasdefaulted20",normalizeInput=True)
    print train_data.shape    
    num_features_to_select = 30
    if len(columns) != train_data.shape[1]:
        print "!!! CAUTION: number of columns and dim of train_data do not agree", len(columns),"!=",train_data.shape[1]
    if rfe:
        train_data,train_target = datasets.make_classification(n_samples=train_data.shape[0], n_features=train_data.shape[1],
                                                               n_informative=num_features_to_select, random_state=0)
    mod = linear_model.LogisticRegression()
    mod.fit(train_data,train_target)
    selector = feature_selection.RFE(mod, n_features_to_select=num_features_to_select)
    selector = selector.fit(train_data,train_target)
    #print(selector.support_)
    print selector.ranking_
    col_ranking = zip(selector.ranking_,columns)
    print columns
    col_ranking.sort()
    #col_ranking.reversed()
    #print col_ranking
    print col_ranking[0:num_features_to_select]

    s = "["    
    for (rank,col) in col_ranking[0:num_features_to_select]:
        s+='"'+col+'",'
    s += "]"
    t_Fn= open("col_names_from_RFE_LogReg.txt",'w')
    t_Fn.write(s)
    t_Fn.close()    
    
    
def perform_logisticReg(train_data,train_target,test_data,test_target,plot=False):
    mod = linear_model.LogisticRegression()
    mod.fit(train_data, train_target)
    predicted_values_train = mod.predict(train_data)
    mse_train_data = AbsError(predicted_values_train,train_target)
    predicted_values_test = mod.predict(test_data)
    mse_test_data = AbsError(predicted_values_test,test_target)
    fit_score = mod.score(train_data,train_target)

    if plot:
        make_simple_scatter(test_target,mod.predict_proba(test_data)[:,1],"True vs Predicted for TestData") 
        make_simple_scatter(train_target,mod.predict_proba(train_data)[:,0],"True vs Predicted for TrainData")  
    return mse_test_data,fit_score,mse_train_data

def apply_LogisticRegression(trainLogR_df,testLogR_df,loss_bound=20):
    """the loss_bound allows to determine for which loss values are to be 0 and which 1;
    this is because the loss has a very uneven distribution;
    E.g., if loss_bound is 1, then this is a straight forward log reg where no-defaulters have 0 and defaulters =1"""
    print '\n applying Logistic Reg method with loss bound=',loss_bound
    trainLogR_df["hasdefaulted20"] = 1
    trainLogR_df["hasdefaulted20"][trainLogR_df["loss"] <loss_bound] = 0
    testLogR_df["hasdefaulted20"] = 1    
    testLogR_df["hasdefaulted20"][testLogR_df["loss"] <loss_bound] = 0 
     
    columns = valid_columns(trainLogR_df)   
    active_var = columns
    train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,trainLogR_df,testLogR_df, "hasdefaulted20",normalizeInput=True)
    print train_data.shape
    mse_test_data,fit_score,mse_train_data = perform_logisticReg(train_data,train_target,test_data,test_target,plot=False)
    print"\n Logistic Regression: MSE(testdata) =",mse_test_data,"goodnes of fit=",fit_score,"MSE(traindata)=",mse_train_data
    ofn = open("LogReg_log_"+datetime.datetime.now().strftime("%d%m-%H-%M")+".csv","w")  
    ofn.write("Variables,MSE(TestData),FitScore,MSE(TrainData)\n")
    print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
    LogRegD= {}
    LogRegD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
    vars_mse = "+".join(active_var)
    vars_score = "+".join(active_var)
    min_mse = 100000000 #mse_test_data
    max_score = 0#fit_score 0 

    for step in [1,3,5,10,15]:
        for i in range(0,len(columns),step):
            active_var = columns[i:i+step]
            train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,trainLogR_df,testLogR_df, "hasdefaulted20",normalizeInput=False)
            mse_test_data,fit_score,mse_train_data = perform_logisticReg(train_data,train_target,test_data,test_target,plot=False)
            LogRegD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}

            print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
            if min_mse > mse_test_data:
                    min_mse = mse_test_data
                    vars_mse = "+".join(active_var)
            if fit_score > max_score:
                    max_score = fit_score
                    vars_score = "+".join(active_var)     


    active_var = most_occurring_vars(LogRegD)    
    train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,trainLogR_df,testLogR_df, "hasdefaulted20",normalizeInput=False)
    mse_test_data,fit_score,mse_train_data = perform_logisticReg(train_data,train_target,test_data,test_target,plot=False)
    LogRegD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}

    print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
    if min_mse > mse_test_data:
            min_mse = mse_test_data
            vars_mse = "+".join(active_var)
    if fit_score > max_score:
            max_score = fit_score
            vars_score = "+".join(active_var)     
    
#     for c1 in range(len(columns)):
#         for c2 in range(c1+1,len(columns)):                       
#             for c3 in range(c2+1,len(columns)):
#                 for c4 in range(c3+1,len(columns)):
#                     for c5 in range(c4+1,len(columns)):
#                         active_var = [columns[c1],columns[c2],columns[c3],columns[c4],columns[c5]]
#                         train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,trainLogR_df,testLogR_df, "hasdefaulted20",normalizeInput=False)
#                         mse_test_data,fit_score,mse_train_data = perform_logisticReg(train_data,train_target,test_data,test_target,plot=False)
#                         LogRegD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
#                         print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
#                         if min_mse > mse_test_data:
#                             min_mse = mse_test_data
#                             vars_mse = "+".join(active_var)
#                         if fit_score > max_score:
#                             max_score = fit_score
#                             vars_score = "+".join(active_var)
    ofn.close()                          
    print "best accuracy achieved with",vars_mse
    print "accuracy=",min_mse
      
    print "best model fit achieved with",vars_score
    print "model fit =", max_score


### applying Knn ########

def apply_Knn(train_df, test_df,dep_variable="loss"):
    """invoke Knn method"""
    print'\napplying K-nearest-neighbor method on response var:',dep_variable
    columns = valid_columns(train_df)  
    active_var = columns
    train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,train_df,test_df,dep_variable,normalizeInput=True)

    ResultsD= {}
    ofn = open("Knn_log_"+datetime.datetime.now().strftime("%d%m-%H-%M")+".csv","w")  
    ofn.write("k,Variables,MSE(TestData),FitScore,MSE(TrainData)\n")
    vars_mse = "+".join(active_var)
    vars_score = "+".join(active_var)
    min_mse = 100000000 #mse_test_data
    max_score = 0#fit_score 0 

    knn = [2,3,5,8,13]
    
    for k in  knn:
        mse_test_data,fit_score,mse_train_data = perform_Knn(train_data, train_target, test_data, test_target,num_neighbors = k)
        print"\n Knn:",k,"MSE(testdata) =",mse_test_data,"goodnes of fit=",fit_score,"MSE(traindata)=",mse_train_data
        ResultsD[(k,"+".join(active_var))] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
        print>>ofn,str(k),',',"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
        if min_mse > mse_test_data:
            min_mse = mse_test_data
            vars_mse = "+".join(active_var)
        if fit_score > max_score:
            max_score = fit_score
            vars_score = "+".join(active_var)     
 
    
    for step in [1,3,5,10,15]:
        for i in range(0,len(columns),step):
            active_var = columns[i:i+step]
            train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,train_df,test_df,dep_variable,normalizeInput=True)
            for k in  knn:
                mse_test_data,fit_score,mse_train_data = perform_Knn(train_data, train_target, test_data, test_target,num_neighbors = k)
                #print"\n Knn:",k,"MSE(testdata) =",mse_test_data,"goodnes of fit=",fit_score,"MSE(traindata)=",mse_train_data
                ResultsD[(k,"+".join(active_var))] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
                print>>ofn,str(k),',',"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
            
                if min_mse > mse_test_data:
                    min_mse = mse_test_data
                    vars_mse = "+".join(active_var)
                if fit_score > max_score:
                    max_score = fit_score
                    vars_score = "+".join(active_var)     

    active_var = most_occurring_vars(ResultsD,toupleDictKey=True) 
    train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,train_df,test_df,dep_variable,normalizeInput=True)

    for k in  knn:
        mse_test_data,fit_score,mse_train_data = perform_Knn(train_data, train_target, test_data, test_target,num_neighbors = k)
        print"\n Knn:",k,"MSE(testdata) =",mse_test_data,"goodnes of fit=",fit_score,"MSE(traindata)=",mse_train_data
        ResultsD[(k,"+".join(active_var))] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
        print>>ofn,str(k),',',"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
        if min_mse > mse_test_data:
            min_mse = mse_test_data
            vars_mse = "+".join(active_var)
        if fit_score > max_score:
            max_score = fit_score
            vars_score = "+".join(active_var)     


    ofn.close()                          
    print "best accuracy achieved with",vars_mse
    print "accuracy=",min_mse
      
    print "best model fit achieved with",vars_score
    print "model fit =", max_score    
    
       
from sklearn import neighbors
def perform_Knn(train_data,train_target,test_data,test_target,num_neighbors=5):

    mod = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    mod.fit(train_data,train_target)
    predicted_values_train = mod.predict(train_data)
    mse_train_data = AbsError(predicted_values_train,train_target)
    predicted_values_test = mod.predict(test_data)
    mse_test_data = AbsError(predicted_values_test,test_target)
    fit_score = mod.score(train_data,train_target)
    return mse_test_data,fit_score,mse_train_data
   
    
### SVM   
def apply_SVM(train_df,test_df,dep_variable="loss"):
    print "\n applying SVM method on response var:",dep_variable
    columns = valid_columns(train_df)  
    ofn = open("SVM_log_"+datetime.datetime.now().strftime("%d%m-%H-%M")+".csv","w")  
    ofn.write("Variables,MSE(TestData),FitScore,MSE(TrainData)\n")
    ResultsD= {}
    
    active_var = columns    
    train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,train_df,test_df,dep_variable,normalizeInput=True)
#     print train_data.shape
#     print train_target.shape
#     print train_data
#     print train_target
    mse_test_data,fit_score,mse_train_data = perform_svm(train_data,train_target,test_data,test_target,'rbf',polydeg=3)
    print"\n SVM: MSE(testdata) =",mse_test_data,"goodnes of fit=",fit_score,"MSE(traindata)=",mse_train_data
    print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
    ResultsD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
    vars_mse = "+".join(active_var)
    vars_score = "+".join(active_var)
    min_mse = 100000000 #mse_test_data
    max_score = 0#fit_score 0 



    for step in [1,3,5,10,15]:
        for i in range(0,len(columns),step):
            active_var = columns[i:i+step]
            train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,train_df,test_df,dep_variable,normalizeInput=True)
            mse_test_data,fit_score,mse_train_data = perform_svm(train_data,train_target,test_data,test_target,'rbf',polydeg=3)
            ResultsD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}
            #print"\n SVM:","+".join(active_var),"MSE(testdata) =",mse_test_data,"goodnes of fit=",fit_score,"MSE(traindata)=",mse_train_data

            print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
            if min_mse > mse_test_data:
                    min_mse = mse_test_data
                    vars_mse = "+".join(active_var)
            if fit_score > max_score:
                    max_score = fit_score
                    vars_score = "+".join(active_var)     


    active_var = most_occurring_vars(ResultsD)    
    train_data,train_target,test_data,test_target = make_data_4scikit_functions(active_var,train_df,test_df,dep_variable,normalizeInput=True)
    mse_test_data,fit_score,mse_train_data = perform_svm(train_data,train_target,test_data,test_target,'rbf',polydeg=3)
    ResultsD["+".join(active_var)] = {"mse_test_data":mse_test_data,"score":fit_score,"mse_train_data":mse_train_data}

    print>>ofn,"+".join(active_var),',',mse_test_data,',',fit_score,',',mse_train_data
    if min_mse > mse_test_data:
            min_mse = mse_test_data
            vars_mse = "+".join(active_var)
    if fit_score > max_score:
            max_score = fit_score
            vars_score = "+".join(active_var)     

    ofn.close()                          
    print "best accuracy achieved with",vars_mse
    print "accuracy=",min_mse
      
    print "best model fit achieved with",vars_score
    print "model fit =", max_score    

        
from sklearn import svm
def perform_svm(train_data,train_target,test_data,test_target,kernel,polydeg=3,plot=False):
    if kernel == 'poly':
        mod = svm.SVC(kernel=kernel,degree=polydeg)
    else:
        mod = svm.SVC(kernel=kernel)
    mod.fit(train_data,train_target)
    predicted_values_train = mod.predict(train_data)
    mse_train_data = AbsError(predicted_values_train,train_target)
    predicted_values_test = mod.predict(test_data)
    mse_test_data = AbsError(predicted_values_test,test_target)
    fit_score = mod.score(train_data,train_target)

    if plot:
        make_simple_scatter(test_target,predicted_values_test,"True vs Predicted for TestData SVM") 
        make_simple_scatter(train_target,predicted_values_train,"True vs Predicted for TrainData SVM")        
    return mse_test_data,fit_score,mse_train_data

#### neural networks 
def apply_ANN(train_df,test_df,dep_variable="loss"):
    print '\n applying ANN method on response var:',dep_variable
    nps = np.array([])
    columns = ["f142","f25","f144","f145","f213","f281","f283",] #valid_columns(train_df)   #["f281","f283","f144","f142","f145","f25","f213"]
    print columns

    for c in range(len(columns)):
        norm_vec = normalize(train_df[columns[c]])
        nps = np.append(nps,norm_vec)
    nps = nps.reshape((len(columns),len(train_df.index)))
    train_data = nps.transpose()
    t_target = np.array(train_df[dep_variable]).reshape((1,len(train_df.index)))
    train_target = t_target.transpose()
    print "size of TrainData:",train_data.shape
    print "size of TrainTarget:",train_target.shape

    nps = np.array([])

    for c in range(len(columns)):
        norm_vec = normalize(test_df[columns[c]])
        nps = np.append(nps,norm_vec)
    nps = nps.reshape((len(columns),len(test_df.index)))
    test_data = nps.transpose()
    t_target = np.array(test_df[dep_variable]).reshape((1,len(test_df.index)))
    test_target = t_target.transpose()
    print "size of TestData:",test_data.shape
    print "size of TestTarget:",test_target.shape
                
    mse_test_data,fit_score,mse_train_data = solve_w_ANN(train_data,train_target,test_data,test_target,len(columns),dep_variable)
    print"\n ANN: MSE(testdata) =",mse_test_data,"FitError=",fit_score,"MSE(traindata)=",mse_train_data
    
#import neurolab as nl
def solve_w_ANN(train_data,train_target,test_data,test_target,num_variables,dep_variable):
    print "\n applying ANN"
    
    minmax_vectors = [[0,1] for i in xrange(num_variables)]
    #create network    
    nn = NN.NN_1HL()
    
    nn.fit(train_data.T, train_target.T)
    predicted_values_test = nn.predict(test_data)
    fit_score = AbsError(test_target, predicted_values_test)
    predicted_values_train = nn.predict(train_data)     
    print fit_score
#    net = nl.net.newff(minmax_vectors,[num_variables+3,1])
#    #train
#    fit_score = net.train(train_data,train_target,epochs=1500, goal=0.001)
#    print "ANN FitScore=",fit_score
#    net.save("NN_"+dep_variable+'.net')
#    
#    predicted_values_train = net.sim(train_data)
#    predicted_values_test = net.sim(test_data)
    mse_train_data = AbsError(predicted_values_train,train_target)
    mse_test_data = AbsError(predicted_values_test,test_target)
    print "incorrect number of predictions for train data (valid only for 0/1 vars)", np.sum(np.abs(predicted_values_train -train_target))
    print "incorrect number of predictions for test data (valid only for 0/1 vars)", np.sum(np.abs(predicted_values_test -test_target))    
    return mse_test_data,fit_score,mse_train_data

 
#### functions for cleaning columns #############     

def valid_columns(df):
    """all columns that are not variables used for predicting MUST be at the end of the dataframe"""
    cols = []
    
    for col in df.columns:
        if col not in ["loss","selection","random",'hasdefaulted','hasdefaulted20','train','lossCategory']:
            cols.append(col)
    return cols

def drop_correlated_columns_fast(df,CorrCoefLimit=0.76):
    #logf = open("log.txt",'w')
    ndf = df.dropna()
    cols2remove = []
    columns = valid_columns(ndf)    
          
    mat = np.array(ndf['f1'])
    for c in range(1,len(columns)):
        mat = np.vstack((mat,ndf[columns[c]]))
    corrmat = np.corrcoef(mat)
    
    for c in range(1,len(columns)-1):
        if columns[c] not in cols2remove:
            for cn in range(c+1,len(columns)):
                if columns[cn] not in cols2remove:
                    if abs(corrmat[c][cn]) > CorrCoefLimit:
                        cols2remove.append(columns[cn])
    print "correlated columns are:",cols2remove
    print "number of correlated columns is",len(cols2remove)
    s = "["
    for col in cols2remove:
        s+='"'+col+'",'
    s += "]"
    t_Fn= open("col_names_correlated.txt",'w')
    t_Fn.write(s)
    t_Fn.close()    


    df = df.drop(cols2remove,1)
    return df
                    
def drop_correlated_columns(df):
    """determine columns that are correlated, if corr coef >=.8, remove the one with more NaNs"""
    
    cols2remove = []
    columns = valid_columns(df)                    
    for c in range(len(columns)):
        if columns[c] not in cols2remove:
            v1 = np.array(df[columns[c]])
            v1numNANS = v1[np.isnan(v1)].shape[0]
            for cn in range(c+1,len(columns)):
                if columns[cn] not in cols2remove:
                    v2 = np.array(df[columns[cn]])
                    v2numNANS = v2[np.isnan(v2)].shape[0]
                    v1 = v1[~np.isnan(v1)]
                    v2 = v2[~np.isnan(v1)]
                    v2 = v2[~np.isnan(v2)]
                    v1 = v1[~np.isnan(v2)]
                    mat = np.vstack((v1,v2))
                    corrmat = np.corrcoef(mat)
                    if abs(corrmat[0][1])>.79:
                        if v1numNANS > v2numNANS:
                            cols2remove.append(v1)
                        else:
                            cols2remove.append(v2)
                    
            
                                
    print "correlated columns are:",cols2remove
    print "number of correlated columns is",len(cols2remove)
    s = "["
    for col in cols2remove:
        s+='"'+col+'",'
    s += "]"
    t_Fn= open("col_names_correlated.txt",'w')
    t_Fn.write(s)
    t_Fn.close()    
    
    df = df.drop(cols2remove,1)
    return df

def check_correlation(df,corrlimit=0.6):
    print "\n checking for columns correlated with response vars"
    columns = valid_columns(df)   
    for resp_name in ["loss","lossCategory","hasdefaulted"]:
        if resp_name in df.columns:
            corrColumns = []       
            for col in columns:
                resp_col = np.array(df[resp_name])
                v1 = np.array(df[col])
                v1 = v1[~np.isnan(v1)]
                resp_col = resp_col[~np.isnan(v1)]
                mat = np.vstack((v1,resp_col))
                corrmat = np.corrcoef(mat)
                if abs(corrmat[0][1])>corrlimit:
                    corrColumns.append(col)
            print "VERY correlated columns with",resp_name,"are:\n"
            print_cols(corrColumns)

def drop_homogeneous_columns(df):
    """columns that have min=max MUST be all filled with the same value,
       drop such columns 
       No need to worry about NaNs
    """
    cols2remove = []
    for col in df.columns:
        v = np.array(df[col])
        v = v[~np.isnan(v)]
        if np.min(v) == np.max(v):#df[col].min() == df[col].max():
            cols2remove.append(col)
            #df[col].to_csv(col+".csv")
 
    print "\nnumber of columns that will be removed because they are filled with only one value",len(cols2remove)
    print "columns that are filled with the same values (not counting NaNs) are:",cols2remove
    s = "["    
    for col in cols2remove:
        s+='"'+col+'",'
    s += "]"
    t_Fn= open("col_names_homogeneous.txt",'w')
    t_Fn.write(s)
    t_Fn.close()
    df = df.drop(cols2remove,1)
    return df

def drop_duplicated_columns(df):
    """there are columns that are duplicates of each other, i.e., a repeat of a column. drop the repeats """
    columns = df.columns
    cols2remove = []
    for c in range(len(columns)):
        if columns[c] not in cols2remove:
            for cn in range(c+1,len(columns)):
                if cn not in cols2remove:
                    v =  np.array(df[columns[c]] - df[columns[cn]])
                    v = v[~np.isnan(v)]
                    if np.min(v) == np.max(v):#  sum(df[tmpName].min() == df[tmpName].max()):
                        #print columns[c],'=',columns[cn],'dropping col',columns[cn]
                        cols2remove.append(columns[cn])

    print "\nnumber of columns that will be removed because they are duplicates of other columns", len(cols2remove)
    print "the following columns will be dropped due to being duplicates of other columns",cols2remove
    s = "["    
    for col in cols2remove:
        s+='"'+col+'",'
    s += "]"
    t_Fn= open("col_names_duplicated.txt",'w')
    t_Fn.write(s)
    t_Fn.close()    
    df = df.drop(cols2remove,1)
    return df

def clean_strlike_var(df):
    """some columns have entries like 123000000000000. for some reason python reads them in as strings.
    we have a few choices: 1) delete such columns, 2) convert them to int/float
    """
    print "\n checking for str-like columns "
    count = 0
    df['tmp'] = 0.0
    for col in df.columns:
        if df[col].dtype == 'object':
            count += 1
            print "str-like variables in column", col
            for i in df.index:
                df['tmp'][i] = float(df[col][i])
            if df['tmp'].dtype == "object":
                print "!!! AHHHH, conversion did not work"   
                df = df.drop(col,1)
            else:
                df[col] = df['tmp'].copy()
    df = df.drop('tmp',1)
    print "number of str-like columns detected",count
    return df


def determine_possibly_useful_variables(fn):
    #fn = "train_selection_subset.csv"
    DATA_DIR = '..'+os.sep+'data'+os.sep

    df = pd.DataFrame.from_csv(DATA_DIR+fn)
    print "number of columns",len(df.columns)
    print df.columns
    print "number of original observations",len(df.index)    
    """taking only a subset, the one Bear selected thru his probability selection"""
    df = df[df['selection']==1] #taking only those that Bear selected

    """calling some cleaning functions"""
    df.replace(['N/A','NA','na','Na','N/a','',None],np.NaN,inplace=True)
    
    
    print "number of observations:",len(df.index)
    print "number of loans that did not default",df['loss'][df['loss'] == 0].count() 
    print "number of loans that defaulted",df['loss'][df['loss'] > 0].count()   
   
    plot_loss_distribution(df,figname="selecSample")
    #now drop columns that do not contribute to anything
    df = drop_homogeneous_columns(df)
    df = drop_duplicated_columns(df)
    df = drop_correlated_columns_fast(df,CorrCoefLimit=0.76)
    print "\n number of columns after cleaning column contents, the number of cols is:",len(df.columns)
    s = "["    
    for col in df.columns:
        s+='"'+col+'",'
    s += "]"
    t_Fn= open("col_names.txt",'w')
    t_Fn.write(s)
    t_Fn.close()

    df = df.dropna()
    print "\n NANs dropped"
    df = response_options(df)
    plot_loss_distribution(df,figname="selecSample_noNANs")
    
    df.to_csv(DATA_DIR+"train_small_noNANs.csv")    




###### plotting #######
def make_simple_scatter(x,y,name):
    x = np.array(x)
    y = np.array(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(x)]
    fig = plt.figure(facecolor='w')
    fig.suptitle(name)
    ax = fig.add_subplot(1,1,1)    
    ax.scatter(x,y)
    plt.savefig(name+'.jpg')                       


def plot_loss_distribution(df,var="loss",figname=None):
    fig,ax = plt.subplots(1,1)
    df[df['loss']>0].hist(column= var,ax=ax,bins = 100)
    fig.suptitle("Distribution of "+var,ha='center', va='center',fontsize=11,color='red')  
    if figname == None:
        figname = "LossDistributionHist"
    
    plt.savefig(figname+".jpg") 

#     fig,ax = plt.subplots(1,1)
#     df[df['loss']>0].hist(column='loss',ax=ax)
#     plt.savefig('fullTrainingSet_LossDistr_afterDroppingNANs.jpg')
  
            #plt.plot(df['loss'], fitted.fittedvalues, 'bd',label='fitted vals')


###### old functions... keeping them only for educational sake    ####

def checking_for_NANs(df, logf=None):
    """allows to check for NaNs in each column;
     could be called at any time, before and after cleaning"""
    for col in df.columns:
        try:
            print col, 'has',sp.sum(sp.isnan(df[col])),'NaNs',df[col].dtype
        except:
            print 'problems with NAN for',col, df[col].dtype 
#         if col in ['f137','f138']:
#             for i in range(len(df.index)):
#                 print >>logf,df[col].iloc[i], 
#                 print >>logf,type(df[col].iloc[i])
#             

##### to delete #######################
def dropnas(df):
    df = df.dropna()
    print "NANs have been dropped"
    print "number of records after removing NaNs", len(df)
    print "number of loans that did not default",df['loss'][df['loss'] == 0].count() 
    print "number of loans that defaulted",df['loss'][df['loss'] > 0].count()   
    print "now differently: num of defaults =", df["hasdefaulted"].sum()
    print "now differently: num of no defaults =", len(df.index) - df["hasdefaulted"].sum()   

    
    


