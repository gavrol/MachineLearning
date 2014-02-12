import os
import numpy as np
import scipy
import pandas
import matplotlib.pyplot as plt
#import statsmodels.api as sm
import statsmodels as sm


if __name__=="__main__":
    fn = "kidney.csv"
    DATA_DIR = '.'+os.sep+"data01"+os.sep
    kdf = pandas.read_csv(DATA_DIR+fn)
    print kdf.columns
    print kdf
    print kdf.groupby
    
    mod1 = sm.regression.linear_model()
    