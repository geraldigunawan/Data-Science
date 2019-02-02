#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 12:16:26 2018

@author: geraldigunawan
"""
#%%
import pandas as pd

#load data
melb_data = pd.read_csv('Melbourne house models/Datasets/melb_data.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing.imputation import Imputer


melb_target = melb_data.Price
#setting predictors to include every factors except house price.
melb_predictors = melb_data.drop(['Price'], axis = 1)

#For the sake of keeping the example simple, we'll use only numeric predictors.
melb_numeric_predictors = melb_predictors.select_dtypes(include=['number'])#or exclude=['object']
print ("numeric predictors are:")
print (list(melb_numeric_predictors), "\n")

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_predictors, melb_target, train_size = 0.7, test_size = 0.3, random_state = 0)

def score_dataset(x_training, x_validate, y_training, y_validate):
    my_model = RandomForestRegressor(n_estimators = 20) #n_estimators is number of trees in the forest
    my_model.fit(x_training, y_training)
    validation_data_predictions = my_model.predict(x_validate)
    return mean_absolute_error(y_validate, validation_data_predictions)

#%%
#drop columns with missing values
cols_with_missing = [col for col in X_train.columns
                                 if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)

print("Mean Absolute Error from dropping columns with Missing Values: ")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test),"\n")

#%%
#impute missing values with some numbers
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test), "\n")

#%%
#get score from imputation with extra columns shwoing what was imputed, this means adding extra predictors for machine to learn pattern
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns
                                 if X_train[col].isnull().any())

for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

testingaja = pd.DataFrame({'Rooms': imputed_X_test_plus.head().Rooms, 'YearBuilt_was_missing': imputed_X_test_plus.head().YearBuilt_was_missing})
print(testingaja)

my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
