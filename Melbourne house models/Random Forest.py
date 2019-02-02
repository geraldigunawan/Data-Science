#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:31:06 2018

@author: geraldigunawan
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

main_file_path = 'Datasets/train.csv'
melbourne_data = pd.read_csv(main_file_path)

y = melbourne_data.SalePrice
predictors = ['LotArea', 'YearBuilt', 'BedroomAbvGr', 'GarageCars']
X = melbourne_data[predictors]

#splitting training data and validation data
training_data_X, validation_data_X, training_data_y, validation_data_y = train_test_split(X, y,random_state = 0)

print("Actual data to reconciliate:")
print(melbourne_data['SalePrice'].loc[(melbourne_data['Id']==530) | (melbourne_data['Id']==492) | (melbourne_data['Id']==460) |(melbourne_data['Id']==280) |(melbourne_data['Id']==656)])

#print("Validation data head:")
#print(str(validation_data_X.head()[predictors]) + "\n")

#%%
#using random forest model to predict melbourne_data
forest_model = RandomForestRegressor()
forest_model.fit(training_data_X, training_data_y)
forest_model_predictions = forest_model.predict(validation_data_X.head())
#print("forest model prediction is" + "\n" + str(forest_model_predictions))
print("forest model prediction")
forest_model_data = pd.DataFrame({'LotArea': validation_data_X.head().LotArea, 'YearBuilt': validation_data_X.head().YearBuilt, 'BedroomAbvGr': validation_data_X.head().BedroomAbvGr, 'GarageCars': validation_data_X.head().GarageCars, 'SalePricePrediction': forest_model_predictions})
print(str(forest_model_data) + "\n")

#%%
#using decision tree model to predict melbourne_data
tree_model = DecisionTreeRegressor()
tree_model.fit(training_data_X, training_data_y)
tree_model_predictions = tree_model.predict(validation_data_X.head())
#print("tree model prediction is" + "\n" + str(tree_model_predictions))
print("tree model prediction")
tree_model_data = pd.DataFrame({'LotArea': validation_data_X.head().LotArea, 'YearBuilt': validation_data_X.head().YearBuilt, 'BedroomAbvGr': validation_data_X.head().BedroomAbvGr, 'GarageCars': validation_data_X.head().GarageCars, 'SalePricePredicttion': tree_model_predictions})
print(tree_model_data)
