import pandas as pd
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

import os
#please use your own directory
os.chdir("F:\PythonWorkspace")

df = getDF('reviews_Electronics_5.json.gz')

df.to_csv("amazon_review.csv",index=False)
