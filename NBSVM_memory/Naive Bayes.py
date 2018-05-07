# -*- coding: utf-8 -*-
"""
Created on Sat May  5 21:16:54 2018

@author: lihan
"""

import pandas as pd
import string, re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
from sklearn.model_selection import train_test_split
import gzip
from sklearn.linear_model import LogisticRegression
import time
from scipy import sparse 
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Electronics_5.json.gz')


#df = pd.read_csv('amazon_cleaned.csv')
df = df.dropna(axis=0,how="any")
df = df.reset_index(drop=True)


# NBSVM
def f():
    begin_time = time.time()
    # a random seed preset for train test splitting
    X_train, X_test = train_test_split(df.loc[:, ['overall', 'reviewText']], train_size=size, test_size=300000,random_state=123)

    label_cols = ['1', '2', '3', '4', '5']

    # reform the stars from 1 column into 5 columns
    for j in label_cols:
        X_train[j] = (X_train['overall'] == int(j))

    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

    def tokenize(s): return re_tok.sub(r' \1 ', s).split()


    #n = X_train.shape[0]
    # parameters are untuned!
    # term frequency–inverse document frequency
    vec = CountVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                          min_df=3, max_df=0.9, strip_accents='unicode')
    # This creates a sparse matrix with only a small number of non-zero elements
    trn_term_doc = vec.fit_transform(X_train['reviewText'])
    test_term_doc = vec.transform(X_test['reviewText'])  # fit a tfid sparse matrix with the same parameters as the train set's
    
    x = trn_term_doc
    test_x = test_term_doc
    # fit the naive bayes model
    
#    # it seems that the Gaussian NB doesn't accept the sparse matrix
##    from sklearn.naive_bayes import GaussianNB
##    clf = GaussianNB()
##    clf.fit(X_train.toarray(), Y_train_reformed.toarray())
#        # get model
#    def get_mdl(y):
#        y = y.values  # array of the values
#        m = MultinomialNB()  # Naive Bayes
#        # return a fitted model of x times its likelihood against y and the likelihood
#        return m.fit(x, y)
#
#    # predict test data
#
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x, X_train['overall'])
    preds = clf.predict(test_x)
#
#    # create a predict matrix (1-5 stars)
#    preds = np.zeros((X_test.shape[0], len(label_cols)))
#
#    for ii, jj in enumerate(label_cols):
#        print('fit', jj)
#        m = get_mdl(X_train[jj])  # feed the column indicating whether it is a 1-5 stars review into the model.
#        print('predict', jj)
#        # predict the probability of 1-5 stars
#        preds[:, ii] = m.predict_proba(test_x)[:, 1]
#    
#     # predictions are the probabilities of stars (1-5)
#    preds = np.asmatrix(preds)
#    stars = np.matrix([1, 2, 3, 4, 5]).transpose()
#
#    # multiply normalized probabilities with stars (Estimation)
#    tmp_preds = preds.sum(1)
#    tmp_preds = preds / tmp_preds
#    tmp = tmp_preds * stars

#    clf = MultinomialNB()
#    clf.fit(x, X_train['overall'])
#    preds = clf.predict(test_x)

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(X_test['overall'], preds)
    print(mse)

    end_time = time.time()
    runtime = end_time-begin_time
    print(runtime)
    return runtime, mse


from memory_profiler import memory_usage

full_mem_usage = []
# full_runtime = []
# full_mse = []

trainsize = [50000]

for size in trainsize:
    print('Current size used:', size)
    mem_usage = memory_usage(f)
    full_mem_usage.append(mem_usage)
    print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
    print('Maximum memory usage: %s' % max(mem_usage))
    


# write results (memory)
for i in range(5):
    thefile = open('SVM'+str(i)+'.txt', 'w')
    for item in full_mem_usage[i]:
        thefile.write("%s\n" % item)
thefile.close()
