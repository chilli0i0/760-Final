# -*- coding: utf-8 -*-
"""
Created on Mon May  7 02:54:48 2018

@author: 12508
"""

amazon = pd.read_csv("amazon_cleaned.csv")
amazon = amazon.dropna(axis=0,how="any")
amazon = amazon.reset_index(drop=True)
amazon_text = amazon.loc[:,"reviewText"]

max_words = 50000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(amazon_text.tolist())
X_all = tokenizer.texts_to_sequences(amazon_text.tolist())

max_length = 500
X_all_new = np.zeros((len(amazon_text),max_length))
for i in range(len(X_all)):
    row = X_all[i]
    for j in range(max_length):
        if j<len(row):
            X_all_new[i,j] = row[j]
        else:
            X_all_new[i,j] = 0
    if (i%100)==0:
        print(i)        
np.save("X_all.npy",X_all_new)
Y = amazon.loc[:,"overall"].astype(int)
Y_all = np.zeros((len(amazon_text),5))
for i in range(len(amazon_text)):
    Y_all[i,Y[i]-1] = 1
np.save("Y_all.npy",Y_all)
