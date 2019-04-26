#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:31:54 2019

@author: geraldigunawan
"""
from Analytics import *
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#%%
train_original.isnull().sum()
#%%
test_original.isnull().sum()
#%%
"""drop loan ID column since it doesn't impact predition"""
train = train_original.drop('Loan_ID', axis = 1)
test = test_original.drop('Loan_ID', axis = 1)
#%%
train.head()
#%%
test.head()
#%%
X = train.drop('Loan_Status',1) 
y = train.Loan_Status
#%%
X.head()
#%%
y.head()
#%%
X = pd.get_dummies(X)
#%%
X.head()
#%%
"""split train and validation data"""
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)
#%%
"""predict train data and measure model accuracy"""
model = LogisticRegression()
model.fit(x_train, y_train)
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)
#%%
"""predict test data"""
test = pd.get_dummies(test)
test
#%%
pred_test = model.predict(test)
#%%
submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']
submission
#%%
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv',, index = False)