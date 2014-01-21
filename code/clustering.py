'''
Created on 20/01/2014

@author: olena
'''
import os
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem    
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


import datetime
tstamp = datetime.datetime.now().strftime('%H-%M')
logf = open('log_'+tstamp+'.txt','w')
    
if __name__ == '__main__':
    DATA_DIR = '..'+ os.sep + 'data'+ os.sep+'20news-18828'+os.sep

    groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
    filenames = []
    for group in groups:
        for fn in os.listdir(DATA_DIR+group):
            if os.path.isfile(DATA_DIR+group+os.sep+fn):
                filenames.append(DATA_DIR+group+os.sep+fn)
                
    posts = [open(fn).read() for fn in filenames]
    print "number of filnames/posts",len(posts)      
    postsA = np.array(posts) 
            
            
    """separate train and test data"""      
  
#     proportion = int(len(posts)*.15) #% of the data to below to test
#     np.random.seed(0)
#     indices =np.random.permutation(len(postsA))
#     train_data = postsA[indices[:-proportion]]
#     test_data = postsA[indices[-proportion:]]
#     print len(train_data),len(test_data)
#     print postsA[0:5]
    
    """invoke language processing tools"""
    min_df =1
    max_df =0.8
    print>> logf,'min_df=',min_df,'max_df=',max_df
    vectorizer = StemmedTfidfVectorizer(min_df=min_df, max_df=max_df,#charset=utf-8,
                                    # max_features=1000,
                                    stop_words='english', decode_error='replace', #charset_error='ignore'
                                    )
    vectorized = vectorizer.fit_transform(postsA)
    #print>>logf, vectorizer.get_feature_names()
    num_samples, num_features = vectorized.shape
    print("#samples: %d, #features: %d" % (num_samples, num_features))

    new_post = """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
    """

    new_post_vec = vectorizer.transform([new_post])
    

 
    """apply clustering method"""
    for num_clusters in xrange(10,150,30): 
        km = KMeans(n_clusters=num_clusters, init='k-means++', n_init=1,verbose=1)
        clustered = km.fit(vectorized)
            
        new_post_label = km.predict(new_post_vec)[0]
    
        similar_indices = (km.labels_ == new_post_label).nonzero()[0]
    
        similar = []
        for i in similar_indices:
            dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
            similar.append((dist, postsA[i]))
        
        similar = sorted(similar)
    
        print>>logf, "\nnfor num_clust=",num_clusters,"number of similar posts:",len(similar)
        print "best match score:",similar[0][0]
        print >>logf,"\n dist_score =",similar[0][0],"TEXT:\n",similar[0][1]
        print >>logf,"\n dist_score =",similar[-1][0],"TEXT:\n",similar[-1][1]
#         for (d,text) in similar:
#             print>>logf,'\n\n',d,':', text.replace('\n',' ').replace('\t', ' ')
        

 
 
 
 
 
 
 
    
#     posts = [open(DATA_DIR+fn).read() for fn in os.listdir(DATA_DIR)]
#     vec = CountVectorizer(min_df=1)
#     X_train = vec.fit_transform(posts)
#     print(X_train)
#     print vec.get_feature_names()
#     print X_train.toarray()
#     print X_train.shape
#     num_samples,num_features = X_train.shape
#     
#     new_post = "imaging databases"
#     new_post_vec = vec.transform([new_post])
#     print new_post_vec
#     print new_post_vec.toarray()
#     
#     
