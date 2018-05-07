#package for data cleaning
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

#read the data and transform them
import os
os.chdir("F:\PythonWorkspace")

max_words = 50000
max_length = 500
X_all = np.load("X_all.npy")
Y_all = np.load("Y_all.npy")

#package for DNN model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

###############################################################################

# create the model
def f():
    
    begin_time = time.time()
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, train_size=size, test_size=30000, random_state=123)
    
    inputs = Input(shape=(X_train.shape[1], ))
    x = Embedding(max_words+1,128)(inputs)
    x = LSTM(128,dropout=0.2, recurrent_dropout=0.2)(x)
    output = Dense(5, activation="sigmoid")(x)
    model = Model(inputs, output)
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=4, batch_size=128)
    
    prediction = model.predict(X_test,verbose=0)
    reuslt2 = prediction[:,0] + 2*prediction[:,1] + 3*prediction[:,2] + 4*prediction[:,3] + 5*prediction[:,4]
    y_test0 = y_test[:,0] + 2*y_test[:,1] + 3*y_test[:,2] + 4*y_test[:,3] + 5*y_test[:,4]
    index1 = reuslt2>5
    reuslt2[index1] = 5
    mse = np.sum(np.square(reuslt2-y_test0))/30000
    print(mse)
    
    end_time = time.time()
    runtime = end_time-begin_time   
    print(runtime)
    
    return runtime,mse

###############################################################################

trainsize = [50000, 100000, 500000, 1000000, 1300000]

from memory_profiler import memory_usage

full_mem_usage = []

for size in trainsize:
    print('Current size used:', size)
    mem_usage = memory_usage(f)
    full_mem_usage.append(mem_usage)
    print('Avager memory usage: %s' % np.mean(mem_usage))
    print('Maximum memory usage: %s' % max(mem_usage))

for i in range(5):
    thefile = open('nbsvm_new'+str(i)+'.txt', 'w')
    for item in full_mem_usage[i]:
        thefile.write("%s\n" % item)


###############################################################################
