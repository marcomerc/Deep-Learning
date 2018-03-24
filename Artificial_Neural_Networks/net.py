
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()                          # we encode the country into a 0 1 2 to make it easier for the net one object to encode       
labelencoder_X_2 = LabelEncoder()                                         # we encode the country into a 0 1 2 to make it easier for the net one object to encode
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # the first endtry is all rows and 1 index or second colomn after this line it's encoded
X[:, 2]= labelencoder_X_2.fit_transform(X[:, 2]) # the first endtry is all rows and 2 index or second colomn after this line it's encoded
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 