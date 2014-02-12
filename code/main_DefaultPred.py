'''
Created on 03/02/2014

@author: olena
'''
import os
import pandas as pd
import numpy as np
import functions_DefaultPred


if __name__ == '__main__':
    DATA_DIR = '..'+os.sep+'data'+os.sep
    fn = "train_v2_subsample3_cleanRec.csv"#"train_v2_subsample_cleanRec.csv"#"train_cleanRec.csv"
    df = pd.DataFrame.from_csv(DATA_DIR+fn)
    print "number of columns",len(df.columns)

    print "number of original observations",len(df.index)  
    print "in original set: number of NONdefaulted is",df['loss'][(df['loss'] == 0) ].count() 
    print "in original set: number of defaulted is",df['loss'][(df['loss'] > 0) ].count() 
    proportion = 80 # .8 of the data is to be for training
    df = functions_DefaultPred.make_train_test_subsets(df,proportion)

    
    print "in TRAINING set: number of NONdefaulted is",df['loss'][(df['loss'] == 0) & (df['train']==1)].count() 
    print "in TRAINING set: number of defaulted is",df['loss'][(df['loss'] > 0) & (df['train']==1)].count() 
 

    train_df = df[df['train']==1]
    test_df = df[df['train']==0]
#    functions_DefaultPred.plot_loss_distribution(train_df, var="loss", figname="LossTrainSet")
#    functions_DefaultPred.plot_loss_distribution(train_df, var="lossCategory", figname="LossCategoryTrainSet")
#     functions_DefaultPred.plot_loss_distribution(test_df, var="loss", figname="LossTestSet")
#     functions_DefaultPred.plot_loss_distribution(test_df, var="lossCategory", figname="LossCategoryTestSet")

    """USE trainDefOnly if modelling loss or lossCategory as dependent variables"""
    dep_variable = "hasdefaulted"
    consider_Defaulters_only = False #True
    if consider_Defaulters_only:
        train_df = train_df[df['loss'] >= 1]
        test_df = test_df[df['loss'] >= 1]
        print "!!! CAUTION: only defaults will be considered. train set has",len(train_df.index),"observations"
        print "!!! CAUTION: only defaults will be considered. test set has",len(test_df.index),"observations"    
        dep_variable = "loss"
 
    """ apply kNN  """
#    functions_DefaultPred.apply_Knn(train_df,test_df,dep_variable=dep_variable)


    """apply SVM """
#   functions_DefaultPred.apply_SVM(train_df,test_df,dep_variable=dep_variable)
    
    """ apply ANN  """#try different goal functions
#    functions_DefaultPred.apply_ANN(train_df,test_df,dep_variable =dep_variable)

    """apply LinRegression """
#    functions_DefaultPred.apply_LinearRegression(train_df, test_df,loss_bound = 1,dep_variable="loss")
  
#  
    """apply logistic regression """
    functions_DefaultPred.apply_LogisticRegression(train_df,test_df,loss_bound=1)

        
