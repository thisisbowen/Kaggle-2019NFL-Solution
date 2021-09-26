# Kaggle-2019NFL-Top3%-Solution
Team member: [Gavin Chen](https://github.com/WeijunChen) and me.

## Problem to Solve
NFL Big Data Bowl hosted by [Kaggle](https://www.kaggle.com/c/nfl-big-data-bowl-2020)  

Predicting [American Football](https://en.wikipedia.org/wiki/American_football) yardage gained of running play by using features known at the time when the ball is handed off.

### Evaluation metric
Continuous Ranked Probability Score (CRPS) is derived based on the predicted scalar value.

## Model Structure

A Single MLP with Batch Normalization and Dropout

```
model.add(Dense(256,input_shape=[self.input_size],activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(199,activation='softmax'))
```
