# -*- coding: utf-8 -*-
#data processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #for data sets


#importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_train.csv')
training_set = dataset_train.iloc[:,1:2].values#values for numpy array #all the rows and look at the csv 1 to 2 but its only the colomn 1 
# we have to normalized the features
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1)) # the range we want to scare the features.
training_set_scaled  = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps and 1 output 
x_train = []
y_train = []

for i in range(60,1258):   # doesnt include 1258 index  it contains the 60 previous days from when we try to predict
    x_train.append(training_set_scaled[i-60:i,0]) #zero colomn and 60 rows for each day 
    y_train.append(training_set_scaled[i,0]) # we need the price on that day  for y as an ouput 
    
x_train, y_train = np.array(x_train), np.array(y_train)

#reshaping  do it on your own
##reshape ## add a  dimention in the numpy 
#x_train.shape[0] number of rows and x_train.shape[1] num of colomns
x_train =  np.reshape(x_train,(x_train.shape[0],x_train.shape[1], 1)) #batch size total days,  timsteps 60, inputsize #new indicator price of another stock that is dependent























