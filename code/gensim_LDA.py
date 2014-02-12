'''
Created on 22/01/2014

@author: olena
'''
import os
import numpy as np
import scipy as sp
from gensim import corpora, models, similarities
from itertools import izip
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import datetime
tstamp = datetime.datetime.now().strftime('%Y%m%d-%H-%M')
logf = open('log_gensim_'+tstamp+'.txt','w')

documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
              "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
print>>logf, "\ntexts=",texts
# remove words that appear only once
all_tokens = sum(texts, [])
print>>logf, "\nall_tokens=",all_tokens
print>>logf, "number of elements in all_tokens =",len(all_tokens)
print >>logf, "\n what is set(all_tokens)???", set(all_tokens)
print >>logf,"number of elements in set(all_tokens) =",len(set(all_tokens))
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
print>>logf,"\ntokens_once=",tokens_once
texts = [[word for word in text if word not in tokens_once] for text in texts]
print>>logf,"\n after cleaning the text: \n",texts,"\n word count in text is",len(texts)


dictionary = corpora.Dictionary(texts)
words_in_dictionary = set(sum(texts,[]))
print words_in_dictionary,"\n num words in dict=",len(dictionary)
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('../data/corpus.mm', corpus) # store to disk, for later use
#### different types of serialization ####
# corpora.SvmLightCorpus.serialize('/tmp/corpus.svmlight', corpus)
# corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
# corpora.LowCorpus.serialize('/tmp/corpus.low', corpus)
# corpus = corpora.MmCorpus('/tmp/corpus.mm') # a way to load a corpus back
print>>logf, "\n dictionary =",dictionary
print>>logf, "\n corpus based on the dictionary =",
for doc in corpus: print>>logf,doc

#before applying any model, it's better to put it thru TermFrequency*InverseDocFreq
tfidf  = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
print>>logf, "\n corpus after TFIDF transformation=", 
for doc in corpus_tfidf:  print>>logf,doc

#now wrap it in LSI model
lsi = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=2)
corpus_lsi = lsi[corpus_tfidf]
print>>logf,"\n corpus after LSI transformation=",
for doc in corpus_lsi: print >> logf,doc
#lsi.print_topics(2)
print>>logf,"\n printing LSI model"
for i in range(lsi.num_topics): print>>logf, lsi.print_topic(i,12)

topics_lsi = [lsi[c] for c in corpus_tfidf]
print 'number of topics in LSI',len(topics_lsi)
for i in range(len(topics_lsi)): print topics_lsi[i]

#now wrap it in LDA model
mod = models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=2)
corpus_mod = mod[corpus_tfidf]
print>>logf,"\n corpus after LDA transformation=",
for doc in corpus_mod: print >> logf,doc
#mod.print_topics(2)
print>>logf,"\n printing LDA model"
for i in range(mod.num_topics): print>>logf, mod.print_topic(i,12)

topics_lda = [mod[c] for c in corpus_tfidf]
print 'number of topics in LDA',len(topics_lda)
for i in range(len(topics_lda)): print topics_lda[i]


new_doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(new_doc.lower().split())
vec_lsi = lsi[vec_bow]
print "new doc converted to lsi", vec_lsi

vec_bow = dictionary.doc2bow(new_doc.lower().split())
vec_lda = mod[vec_bow]
print "new doc converted to lda", vec_lda


index_lsi = similarities.MatrixSimilarity(corpus_lsi)
index_lda = similarities.MatrixSimilarity(corpus_mod)

sims_lsi = index_lsi[vec_lsi]
sims_lda = index_lda[vec_lda]

sims_lsi = sorted(enumerate(sims_lsi),key=lambda item:-item[1])
sims_lda = sorted(enumerate(sims_lda),key=lambda item:-item[1])
print>>logf,"\n similarities of",new_doc,"to LSI",sims_lsi
print>>logf,"\n similarities of",new_doc,"to LDA",sims_lda





# if __name__ == '__main__':
#     DATA_DIR = '..'+ os.sep + 'data'+ os.sep+'20news-18828'+os.sep
# 
#     groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
#     filenames = []
#     for group in groups:
#         for fn in os.listdir(DATA_DIR+group):
#             if os.path.isfile(DATA_DIR+group+os.sep+fn):
#                 filenames.append(DATA_DIR+group+os.sep+fn)
#                 
#     posts = [open(fn).read() for fn in filenames]
#     print "number of filnames/posts",len(posts)      
