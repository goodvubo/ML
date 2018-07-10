# Classification template

'''Importing the libraries'''
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


'''Importing the dataset'''
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values


'''Encoding categorical data'''
'''Encoding the Independent Variable'''
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#onehotencoder = OneHotEncoder(categorical_features = [3])
#X = onehotencoder.fit_transform(X).toarray()
'''Encoding the Dependent Variable'''
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)


'''Avoiding Dummy Variable Trap'''
#X = X[:, 1:]


'''Splitting the dataset into the Training set and Test set'''
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


'''Feature Scaling'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


'''Fitting classifier to the Training set'''
#def costFunction():
#    return np.sum(np.square(np.subtract(np.dot(W, X_n_ones_transpose), y_train))) / (2 * X_train.shape[0])


#def prediction(x):
#    tmpPrd = np.dot(W, (np.insert(x, [0], np.ones((x.shape[0], 1)), axis=1)).transpose())
#    return [(1 / (1 + math.exp(-i))) for i in tmpPrd]


alpha = 0.03
W = [0] * (X_train.shape[1] + 1)
X_n_ones = np.insert(X_train, [0], np.ones((X_train.shape[0], 1)), axis=1)
'''Transpose to use in Dot Product operation'''
X_n_ones_transpose = X_n_ones.transpose()

'''Gradient Ascent'''
tempW = W

while(True):
    ''' ŷ '''
    WdotX = np.dot(W, X_n_ones_transpose)
    WdotX = [(1 / (1 + math.exp(-i))) for i in WdotX]
    ''' (y - ŷ) '''
    WdotX_y = np.subtract(y_train, WdotX)
    ''' Updating all θ '''
    tempW = [(tempW[i] + ((alpha / X_train.shape[0]) * sum(np.multiply(WdotX_y, X_n_ones_transpose[i])))) for i in range(0, X_n_ones_transpose.shape[0])]
    if(np.array_equal(W, tempW)):
        break
    W = tempW


'''Prediction'''
tmpPredict = np.dot(W, (np.insert(X_test, [0], np.ones((X_test.shape[0], 1)), axis=1)).transpose())
y_pred_temp = [(1 / (1 + math.exp(-i))) for i in tmpPredict]
y_pred = [int(np.rint(i)) for i in y_pred_temp]


'''Making the Confusion Matrix'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
