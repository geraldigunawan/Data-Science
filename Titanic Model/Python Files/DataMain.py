#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:12:11 2018

@author: geraldigunawan

Purpose: predict the survival of passengers abroad Titanic ship
"""

import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.ensemble import RandomForestRegressor
#%%
print('The scikit-learn version is {}.'.format(sklearn.__version__))

#set console view
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', 10)
pd.set_option('display.max_columns', 100)

titanic_train_data = pd.read_csv('/Applications/MAMP/htdocs/Data-Science/Titanic Model/Datasets/train.csv')
train_data_copy = titanic_train_data.copy()
#%% add additional columns to indicate columns with missing data
def add_extra_column_to_indicate_missing_value(titanic_data):
    cols_with_missing = (col for col in titanic_data.columns
                                 if titanic_data[col].isnull().any())

    for col in cols_with_missing:
        titanic_data[col + '_was_missing'] = 1
    return titanic_data
#%%
def list_columns_with_missing_value(titanic_data):
    result = []
    cols_with_missing = (col for col in titanic_data.columns
                                 if titanic_data[col].isnull().any())
    for col in cols_with_missing:
        result.append(col + '_was_missing')    
    return result
#%% Fill miussing age values with mean
def populate_missing_age(titanic_data):
    titanic_data['Age'].fillna(round(titanic_data['Age'].mean()), inplace=True)
    return titanic_data
#%% Fill missing embarkation port based on C = Cherbourg, Q = Queenstown, S = Southampton
def populate_missing_embarkation_port(titanic_data):
    for index, row in titanic_data.iterrows():
        port_list = ['C'] * 15 + ['Q'] * 30 + ['S'] * 55
        titanic_data['Embarked'].fillna((random.choice(port_list)), inplace=True)
    return titanic_data
#print(titanic_port_populated['Embarked'].loc[(result['PassengerId'] == 62)]);