#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:32:52 2018

@author: geraldigunawan
"""
#%%
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

main_file_path = 'Datasets/train.csv'
melbourne_data = pd.read_csv(main_file_path)

y = melbourne_data.SalePrice
predictors = ['LotArea', 'YearBuilt', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars']
X = melbourne_data[predictors]

training_data_X, validation_data_X, training_data_y, validation_data_y = train_test_split(X, y,random_state = 0)

def get_mae(max_leaf_nodes, trainingdata_X, validationdata_X, trainingdata_y, validationdata_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(trainingdata_X, trainingdata_y)
    validation_data_predictions = model.predict(validationdata_X)
    mae = mean_absolute_error(validationdata_y, validation_data_predictions)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 55, 60, 70, 80]:
    my_mae = get_mae(max_leaf_nodes, training_data_X, validation_data_X, training_data_y, validation_data_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
