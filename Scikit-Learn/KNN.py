# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:37:36 2020

@author: csevern
"""
#%% Load Data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


df = pd.read_csv("", encoding='utf-8')

# Assign colum names to the dataset
#%% Select Data

# Read dataset to pandas dataframe

print(df.head())
df = df.sample(frac=1).reset_index(drop=True)
X = df.iloc[:10000, 1:-1].values
y = df['CCs'].tolist()[:10000]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

#%%Train

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
score = 0
for i in range(1,100,3):
    p=i+2
    sc2=0
    classifier = KNeighborsClassifier(n_neighbors=i, weights='distance')
    classifier.fit(X_train, y_train)
    
    classifier1 = RadiusNeighborsClassifier(radius=p,n_neighbors=i, weights='distance')
    classifier1.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_pred1 = classifier1.predict(X_test)
    
    sc1 =classifier.score(X_train,y_train)
    sc2 =classifier1.score(X_train,y_train)
    print(i,p,sc1,sc2)
    if sc1 > sc2:
        ml = classifier
        mltype = "KNN"
        scr = sc1
        pred = y_pred
        
    else:
        ml = classifier1
        mltype = "Radius"
        scr = sc2
        pred = y_pred1
    #print("K nearest =", i, "Radius=", i, "Type=", mltype, scr)    
    if scr > score:
        print("K nearest =", i, "Radius=", i, "Type=", mltype, scr)
        score = scr
        saveml = ml
        savepred = pred
            
        


save_class = open("KNN_CCs.pickle", "wb")
print("saving file")
pickle.dump(classifier, save_class)
save_class.close()
from sklearn.metrics import classification_report
if mltype == "KNN":

    print("="*20, "KNN", "="*20)
    #print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, savepred))
    #print(classifier.score(X_train,y_train))
    #print(classifier.score(X_train,y_train))
else:
    print("="*20, "RNN", "="*20)
    #print(confusion_matrix(y_test, y_pred1))
    print(classification_report(y_test, savepred))
   ## print(classifier1.score(X_train,y_train))
    #print(classifier1.score(X_train,y_train))
