#import packages
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords



#set the system not to print warnings
pd.options.mode.chained_assignment = None



#Read the data
#please use your own directory
import os
os.chdir("F:\PythonWorkspace")

amazon = pd.read_csv("amazon_review.csv")
amazon = amazon.dropna(axis=0,how="any")
amazon = amazon.reset_index(drop=True)
amazon_text = amazon.loc[:,"reviewText"]



#Remove the noise
stops = set(stopwords.words("english"))
keep = {"all","do","don't","her","his","him","should", "shouldn't","not","aren't","couldn't","didn't",'didn','doesn',
        "doesn't","didn't","wouldn't","won't","won","shan't","hasn't","hadn't",
        "weren't","wasn't","mustn't","mightn't","weren't","very","couldn't",
        "isn't","haven't","haven't"}
for word in keep:
    stops.remove(word)

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

def noise_remove(review, stops):
    #only keep characters ! and ?
    review = re.sub("\n", " ", review)
    review = re.sub("!", " !", review)
    review = re.sub("\?", " ?", review)
    review = re.sub("[^a-zA-Z!?]", " ", review)
    review = re.sub("[^a-zA-Z!?']", " ", review)
    #change to lower case
    review = review.lower()
    #split words like isn't to is not
    review = re.sub("n't"," not",review)
    review = re.sub("n'"," not",review)
    #remove the stop words
    review = review.split()
    useful_review = [word for word in review if not word in stops]
    #normalize the verb and noun
    useful_review = [lem.lemmatize(word, "v") for word in useful_review]
    return " ".join(useful_review)

for i in range(len(amazon_text)):
    amazon_text[i] = noise_remove(amazon_text[i],stops)
    if (i%10000)==0:
        print(i)
    
for i in range(len(10)):
    print(i)

#produce the word dictionary
wordcount={}
for i in range(len(amazon_text)):
    text = amazon_text[i].split()
    for j in range(len(text)):
        word = text[j]
        if word in ("not","no","never"):
            if (j+1) < len(text):
                word = text[j] + " " + text[j+1]
                j = j+1
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

print("The length of dictionary is:",len(wordcount.keys()))



#cut down words with counts lower than 2
import operator
wordcount_sort = sorted(wordcount.items(), key=operator.itemgetter(1))

word_count_value = np.asarray(list(wordcount.values()))
word_count_key = np.asarray(list(wordcount.keys()))
word_count_key = word_count_key[word_count_value.astype(int)>=2]

wordcount2 = {}
for word in word_count_key:
    wordcount2[word] = wordcount[word]
wordcount = wordcount2
print("The length of dictionary is:",len(wordcount.keys()))

feature_names = list(wordcount.keys())



#Produce the feature matrix
def initialize_matrix(value,row,col):
    matrix = []                                                      
    for i in range (0, row):     
        new = []                 
        for j in range (0, col):   
            new.append(value)      
        matrix.append(new)
    return matrix

feature_matrix = initialize_matrix(0,len(amazon_text),len(wordcount.keys()))

feature_names1 = pd.DataFrame(feature_names)
feature_names2 = set(feature_names1[0])

feature_names1["num"] = list(range(len(feature_names)))

for i in range(len(amazon_text)):
    text = amazon_text[i].split()
    for j in range(len(text)):
        word = text[j]
        if word in feature_names:
            j = feature_names.index(word)
            feature_matrix[i][j] = feature_matrix[i][j] + 1
    if (i%100)==0:
        print(i)        

feature_matrix = pd.DataFrame(feature_matrix)
feature_matrix.to_csv("feature_matrix.csv",index=False)


amazon.to_csv("amazon_cleaned.csv")











