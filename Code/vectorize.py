import pandas as pd
import string, re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
from sklearn.model_selection import train_test_split
import gzip


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

df = getDF('./reviews_Electronics_5.json.gz')


# a random seed preset for train test splitting
X_train, X_test = train_test_split(df.loc[:, ['overall', 'reviewText']], test_size=0.3, random_state=123)


label_cols = ['1', '2', '3', '4', '5']

# reform the stars from 1 column into 5 columns
for j in label_cols:
    X_train[j] = (X_train['overall'] == int(j))

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()


# n = df.shape[0]
n = X_train.shape[0]
# parameters are untuned!
# term frequency–inverse document frequency
vec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
# This creates a sparse matrix with only a small number of non-zero elements
trn_term_doc = vec.fit_transform(X_train['reviewText'])
test_term_doc = vec.transform(X_test['reviewText'])  # fit a tfid sparse matrix with the same parameters as the train set's
