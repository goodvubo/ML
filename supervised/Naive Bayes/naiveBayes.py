# Classification template

'''Importing the libraries'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''Importing the dataset'''
dataset = pd.read_csv('playing_tennis.csv')


'''Encoding categorical data'''
from sklearn.preprocessing import LabelEncoder
df = dataset.apply(LabelEncoder().fit_transform)

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values


'''Splitting the dataset into the Training set and Test set'''
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.01, random_state = 0)

'''Feature Scaling'''
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

'''Fitting classifier to the Training set'''
pdf = {}
priorFrequency = {}
for i in range(0, X.shape[0]):
    if(y[i] not in priorFrequency.keys()):
        priorFrequency[y[i]] = 1
    else:
        priorFrequency[y[i]] += 1
        

classConditionalFrequency = {}
for key in priorFrequency:
    classConditionalFrequency[key] = [0] * X.shape[1]
    
    
x_test = X
y_pred = [None] * len(x_test)


CCF = {}    
for i in range(0, len(x_test)):
    CCF = classConditionalFrequency
    for j in range(0, len(X)):
        for k in range(0, X.shape[1]):
            if(x_test[i][k] == X[j][k]):
                CCF[y[j]][k] += 1
    maxPdf = -1
    for key in priorFrequency:
        temp = 1
        for itm in CCF[key]:
            temp *= (itm/priorFrequency[key])
        pdf[key] = (priorFrequency[key]/X.shape[0]) * temp
        if(pdf[key] > maxPdf):
            maxPdf = pdf[key]
            y_pred[i] = key


'''Making the Confusion Matrix'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
