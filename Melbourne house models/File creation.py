#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:26:47 2018

@author: geraldigunawan
"""
#%%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv('Datasets/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)

# Read the test data
test = pd.read_csv('Datasets/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

"""my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices}, index = ['first','sec','third','fourth','fifth'])
print(my_submission)"""

my_submission = pd.DataFrame({'Id': test.Id, 'LotArea': test.LotArea,'SalePrice': predicted_prices}, columns=['Id','SalePrice'])
print(my_submission)

# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
