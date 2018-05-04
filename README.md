# 760-Final

## Comparing methods in predicting the Amazon reviews. (Sentiment Classification)

Our plan is to compare the execution time and the memory usage of different models.

Models that we want to use:
* LogisticRegression
* Naive Bayes
* SVM
* NBSVM
* Xgboost
* LSTM

The train test size that we are going to test on:
* A fixed test size so that the accuracy won't be affected by the size: 300,000
* Different train sizes:
  * 50,000
  * 100,000
  * 500,000
  * 1,000,000
  * 1,300,000
* One problem that remains: Should we run it multiple times and take the average (this scheme is preferred but would take much longer) or should we set a seed and only run it once?
* To split the train and test sample, use: `train_test_split(df.loc[:, ['overall', 'reviewText']], train_size=100000, test_size=300000, random_state=123)`

A scheme to check memory and execution time is given in `nbsvm.py`. Don't hesitate find better ways.

For example, NBSVM with `train_size=100000, test_size=300000` has an MSE of `0.6426298624704828` and has an execution time of `217.8640398979187` seconds.

Furthermore, the memory usage is:

```
Memory usage (in chunks of .1 seconds): [2290.8203125, ..., 3312.515625, 2727.3203125]
Maximum memory usage: 5572.56640625
```

(By the way, the MSE is `0.48158917592225187` and execution time is `1691.8317730426788` seconds with train_size=0.7 and test_size=0.3)



## How to compare the memory and execution time?
The plan is:
* Regarding execution time, we are going to compare different methods under the same splits
* Both the max memory usage and the memory usage over time is planned to be compared.
  * Memory over time should be compared under the percentage of time since it is for sure that different methods will use different amount of time to execute.
