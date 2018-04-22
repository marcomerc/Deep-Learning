# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the data set
dataset = pd.read_csv('Credit_Card_Applications.csv')
x= dataset.iloc[:,:-1] # the -1 is for the last value 
y= dataset.iloc[:,-1] # the -1 is for the last value 




#fesature scaling  Data preprocessing/
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(x) # NORMALIZATION

# we are going to use a libary.
from minisom import MiniSom
som  = MiniSom(x = 10, y =10, input_len = 15, sigma =1.0, learning_rate = 0.5) 
som.random_weights_init(X)
som.train_random(X,100)









