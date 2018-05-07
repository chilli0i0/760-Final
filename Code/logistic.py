# -*- coding: utf-8 -*-
"""
@author: Think
"""
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression  
from memory_profiler import memory_usage
amazon=pd.read_csv("amazon_cleaned.csv")
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer = CountVectorizer()  
matrix = vectorizer.fit_transform(amazon['summary']) 
from sklearn.feature_extraction.text import TfidfTransformer  
transformer = TfidfTransformer()  
tfidf = transformer.fit_transform(matrix)  
n=[50000, 100000, 500000, 1000000, 1300000]
test_x = tfidf[1300000:]
test_y = amazon['overall'][1300000:]
mem = []
for i in range(5):
    ni=n[i]
    mem.append(memory_usage(f))
def f():
    print("size: " + str(ni))
    train_x = tfidf[:ni]
    train_y = amazon['overall'][:ni]
    classifier = LogisticRegression(C=2,penalty="l2",multi_class="multinomial",solver="saga")
    time1 = time.time()
    p_logit=classifier.fit(train_x,train_y)
    time2 = time.time()
    print('Time spent on training model: '+ str(time2-time1))
    y_logit=p_logit.predict_proba(test_x)
    prediction = [sum(np.array([1,2,3,4,5])*i) for i in y_logit]
    mse = np.mean(np.square((prediction-test_y)))
    print('MSE: ' + str(mse))
    return float(time2-time1), mse

