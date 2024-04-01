# Stock Price Prediction

## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
We aim to build a RNN model to predict the stock prices of Google using the dataset provided. The dataset has many features, but we will be predicting the "Open" feauture alone. We will be using a sequence of 60 readings to predict the 61st reading.
Note: These parameters can be changed as per requirements.

## Design Steps

### Step 1:
Read the csv file and create the Data frame using pandas.


### Step 2:
Select the " Open " column for prediction. Or select any column of your interest and scale the values using MinMaxScaler.



### Step 3:
Create two lists for X_train and y_train. And append the collection of 60 readings in X_train, for which the 61st reading will be the first output in y_train.

### STEP 4:
Make Predictions and plot the graph with the Actual and Predicted values.



## Program
#### Name: Harish R
#### Register Number: 21222110012

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential
```
```
dataset_train = pd.read_csv('trainset.csv')
```
```
train_set = dataset_train.iloc[:,1:2].values
```
```
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
```
```
X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))
```
```
model = Sequential()
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(64, input_shape=X_train.shape[1:], activation="relu"),
    Dense(32, activation='tanh'),
    Dense(16, activation='relu'),
    Dense(8, activation='tanh'),
    Dense(4, activation='softmax'),
])

model.compile(optimizer='adam', loss='mse')
```

## Output

### True Stock Price, Predicted Stock Price vs time

![image](https://github.com/Harishspice/rnn-stock-price-prediction/assets/117935868/70b0822e-6395-4f4e-a04f-17abcac4cdff)


### Mean Square Error

![image](https://github.com/Harishspice/rnn-stock-price-prediction/assets/117935868/9f07f229-efc2-4907-a805-999f58a08701)

## Result
Hence, we have successfully created a Simple RNN model for Stock Price Prediction.


