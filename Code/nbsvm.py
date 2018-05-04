import pandas as pd
import string, re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy import sparse
from sklearn.model_selection import train_test_split
import gzip
from sklearn.linear_model import LogisticRegression

# Read Data
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

df = getDF('./DATA/reviews_Electronics_5.json.gz')


# NBSVM

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

vocab = vec.get_feature_names()  # Get the feature names of the vectorizer


# the basic naive bayes feature equation, probability
def pr(y_i, y):
    # y_i binary value {0,1}
    # y true value (whether the review is a 1-5 star review)
    p = x[y == y_i].sum(0)  # sum on axis 0 (column wise)
    # return normalized probability with laplace correction
    return (p + 1) / ((y == y_i).sum() + 1)


x = trn_term_doc
test_x = test_term_doc


# get model
def get_mdl(y):
    y = y.values  # array of the values
    r = sparse.csr_matrix(np.log(pr(1, y) / pr(0, y)))  # likelihood
    m = LogisticRegression(C=4, dual=True)  # logistic regression
    x_nb = x.multiply(r)
    # return a fitted model of x times its likelihood against y and the likelihood
    return m.fit(x_nb, y), r


# predict test data

# create a predict matrix (1-5 stars)
preds = np.zeros((X_test.shape[0], len(label_cols)))

for ii, jj in enumerate(label_cols):
    print('fit', jj)
    m, r = get_mdl(X_train[jj])  # feed the column indicating whether it is a 1-5 stars review into the model.
    print('predict', jj)
    # predict the probability of 1-5 stars
    preds[:, ii] = m.predict_proba(test_x.multiply(r))[:, 1]

# predictions are the probabilities of stars (1-5)
preds = np.asmatrix(preds)
stars = np.matrix([1, 2, 3, 4, 5]).transpose()

# multiply normalized probabilities with stars (Estimation)
tmp_preds = preds.sum(1)
tmp_preds = preds / tmp_preds
tmp = tmp_preds * stars


from sklearn.metrics import mean_squared_error
mean_squared_error(X_test['overall'], tmp)
