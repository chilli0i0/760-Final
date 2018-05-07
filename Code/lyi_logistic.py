# -*- coding: utf-8 -*-
"""
@author: Think
"""
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from memory_profiler import memory_usage
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

amazon = pd.read_csv("./Data/amazon_cleaned.csv")
amazon = amazon.dropna(axis=0,how="any")
amazon = amazon.reset_index(drop=True)


def f():
    print("size: " + str(ni))
    time1 = time.time()
    X_train, X_test = train_test_split(amazon.loc[:, ['overall', 'reviewText']], train_size=ni, test_size=300000,
                                       random_state=123)

    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(X_train['reviewText'])
    transformer = TfidfTransformer()
    train_x = transformer.fit_transform(matrix)
    train_y = X_train['overall']

    test_x = transformer.transform(vectorizer.transform(X_test['reviewText']))
    test_y = X_test['overall']

    classifier = LogisticRegression(C=2,penalty="l2",multi_class="multinomial",solver="saga")

    p_logit=classifier.fit(train_x,train_y)

    y_logit = p_logit.predict_proba(test_x)
    prediction = [sum(np.array([1,2,3,4,5])*i) for i in y_logit]
    # mse = np.mean(np.square((prediction-test_y)))

    mse = mean_squared_error(test_y, prediction)
    print('MSE: ' + str(mse))

    time2 = time.time()
    print('Time spent on training model: '+ str(time2-time1))
    return float(time2-time1), mse

n = [50000, 100000, 500000, 1000000, 1300000]
mem = []
for i in range(5):
    ni = n[i]
    mem.append(memory_usage(f))

for i in range(5):
    print('Max memory usage:' + str(max(mem[i])))

for i in range(5):
    print('Max memory usage:' + str(np.mean(mem[i])))

# write results (memory)
for i in range(5):
    thefile = open('logit_summary'+str(i)+'.txt', 'w')
    for item in mem[i]:
        thefile.write("%s\n" % item)
