# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 16:04:49 2020

@author: csevern
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import random

df = pd.read_csv("")

#WcR	AgC   AccC
list9 = []
ylist = df['AccC'].tolist()
xlist = df['WcR'].tolist()  
train_x=[]
train_y=[]
test_x = []
test_y = []
for v,x in enumerate(xlist):
    num = random.randint(1,10)
    if num == 1:
        test_x.append(x)
        test_y.append(ylist[v])
    else:
        train_x.append(x)
        train_y.append(ylist[v])
y_values = np.array(train_y).reshape(len(train_y),1)
x_values = np.array(train_x).reshape(len(train_x),1)

# Create linear regression object
#regr = RANSACRegressor(min_samples=2, max_trials=1000, max_skips=10000, stop_probability = 0.99,residual_threshold=5)

#regr = LinearRegression(fit_intercept=True, normalize=False, n_jobs=10, copy_X=True)

 
regr = TheilSenRegressor(max_iter=1000)
#regr = HuberRegressor(max_iter=10000, fit_intercept=False)

# Train the model using the training sets
#prev = 100000000000
#for x in range(1, 10):
    #degrees = x
    #regr.fit(np.vander(x_values, degrees), y_values)

regr.fit(x_values, y_values.ravel())
for v,x in enumerate(test_x):
    
    value = np.array([x]).reshape(1)
    predicted = regr.predict(np.vander(value))
    
    #predicted = regr.predict(np.vander(value,degrees),return_std=True)
    #standard = np.sqrt(predicted[1].item())
        #if standard < prev:
            #predictedval = predicted[0].item()
            #prev = standard
    predictedval = predicted[0].item()
    print(x, predictedval)
    list9.append(int(predictedval))
    

CombinedList = pd.DataFrame(list(zip(test_x, test_y, list9)), columns=['Word Count', 'Real CC', 'Predicted CC'])

savefile = ""
CombinedList.to_csv(savefile, encoding='utf-8')
"""
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('Mean squared error: %.2f'
      % mean_squared_error(diabetes_X_test, diabetes_x_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(diabetes_y_test, diabetes_y_pred))
"""
# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
"""