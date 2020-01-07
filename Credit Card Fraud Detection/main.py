#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 16:00:39 2019

@author: geraldigunawan
"""

import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime as dt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

#%%
pd.set_option('display.max_columns', 100)
#%%
cc_data = pd.read_csv('creditcard.csv')
#%%
cc_data.head()
cc_data.columns
cc_data.shape
cc_data.isnull().sum()
#%%
cc_data['Class'].unique()
cc_data.columns
#%%
cc_data[cc_data['Class'] == 1]['Class'].count()
cc_data[cc_data['Class'] == 0]['Class'].count()
#%%
cc_fraud_time_and_amount = cc_data[cc_data['Class'] == 1][['Time','Amount']]
cc_fraud_time_and_amount.describe()
#%%
cc_fraud_time_and_amount['Time_in_hour'] = pd.to_datetime(cc_fraud_time_and_amount.Time, unit='s').dt.strftime('%H:%M:%S')
cc_fraud_time_and_amount
#%%
##swap column
columnsTitles=["Time","Time_in_hour","Amount"]
cc_fraud_time_and_amount=cc_fraud_time_and_amount.reindex(columns=columnsTitles)
cc_fraud_time_and_amount
#%%
cc_genuine_time_and_amount = cc_data[cc_data['Class'] == 0][['Time','Amount']]
cc_genuine_time_and_amount.max()
#%%
X = cc_data.iloc[:,0:29]  #independent columns
X
#%%
y = cc_data.iloc[:,-1]    #target column i.e price range
y
#%%
X_numerical_columns = list(X._get_numeric_data().columns)
X_numerical_columns
#%
#%%
matrix = X.corr()
sns.heatmap(matrix, square=True, cmap="BuPu");
#%%
models = [RandomForestClassifier(),ExtraTreesClassifier()]
for model in models:
    model.fit(X,y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()
    #%%