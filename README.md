# Kaggle-2019NFL-Solution

## Problem to Solve
NFL Big Data Bowl hosted by [Kaggle](https://www.kaggle.com/c/nfl-big-data-bowl-2020)  

Predicting [American Football](https://en.wikipedia.org/wiki/American_football) yardage gained of running play by using features known at the time when the ball is handed off.

### evaluation me
Continuous Ranked Probability Score (CRPS) is derived based on the predicted scalar value.
The CRPS is computed as follows:
$$
C=\frac{1}{199N}\sum_{m=1}^N\sum_{n=-99}^{99}(P(y\geq n)-H(n-Y_m))^2
$$
$H(x)=1$ if $x\geq 0$ else $0$

## Feature Ideas

## Model Structure

A Single MLP with Batch Normalization and Dropout

```
model.add(Dense(256,input_shape=[self.input_size],activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(199,activation='softmax'))
```


## Things we didn't do well
