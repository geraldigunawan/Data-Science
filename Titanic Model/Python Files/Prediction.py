#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 13:29:41 2018

@author: geraldigunawan
"""

import pandas as pd
import numpy as np
from DataMain import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

titanic_train_data = pd.read_csv('/Applications/MAMP/htdocs/Data-science/Data-Science-2/Titanic Model/Datasets/train.csv')
titanic_test_data = pd.read_csv('/Applications/MAMP/htdocs/Data-science/Data-Science-2/Titanic Model/Datasets/test.csv')
#%%
traindata_with_extra_columns = add_extra_column_to_indicate_missing_value(titanic_train_data)
missing_columns_in_train_data = list_columns_with_missing_value(titanic_train_data)
traindata_with_age_filled_with_mean = populate_missing_age(traindata_with_extra_columns)
traindata_after_modification = populate_missing_embarkation_port(traindata_with_age_filled_with_mean)
#%% relabel Sex
le = LabelEncoder()
le.fit(traindata_after_modification.Sex)
list(le.classes_)
#%%relabel Embarkartion port
le.fit(traindata_after_modification.Embarked)
list(le.classes_)
#%% transform labeling
traindata_after_modification.Sex = le.transform(traindata_after_modification.Sex)
#%% transform labeling
traindata_after_modification.Embarked = le.transform(traindata_after_modification.Embarked)
#%% transform inverse labeling
le.inverse_transform(traindata_after_modification.Sex)
#%%
traindata_after_modification.Sex
#%%
traindata_after_modification.Embarked
#%%
traindata_after_modification.columns
#%%
print(missing_columns_in_train_data)
#%%
testdata_with_extra_columns = add_extra_column_to_indicate_missing_value(titanic_test_data)
missing_columns_in_test_data = list_columns_with_missing_value(titanic_test_data)
testdata_with_age_filled_with_mean = populate_missing_age(testdata_with_extra_columns)
testdata_after_modification = populate_missing_embarkation_port(testdata_with_age_filled_with_mean)
#%%relabel Sex
le = LabelEncoder()
le.fit(testdata_after_modification.Sex)
list(le.classes_)
#%%relabel Embarkartion port
le.fit(testdata_after_modification.Embarked)
list(le.classes_)
#%% transform labeling
testdata_after_modification.Sex = le.transform(testdata_after_modification.Sex)
#%% transform labeling
testdata_after_modification.Embarked = le.transform(testdata_after_modification.Embarked)
#%% transform inverse labeling
le.inverse_transform(testdata_after_modification.Sex)
#%%
testdata_after_modification.Sex
#%%
testdata_after_modification.Embarked
#%%
testdata_after_modification.columns
#%%
print(missing_columns_in_test_data)
#%%
target_prediction = traindata_after_modification.Survived
total_missing_columns_in_train_test = list(set(missing_columns_in_train_data) & set(missing_columns_in_test_data))
predictors = ['PassengerId','Sex','Pclass','Age','SibSp','Embarked'] + total_missing_columns_in_train_test
#%%
print(predictors)
#%%
traindata_X = traindata_after_modification[predictors]
#%%
testdata_X = testdata_after_modification[predictors]
#%%
#using random forest model to predict melbourne_data
forest_model = RandomForestRegressor()
forest_model.fit(traindata_X, target_prediction)
forest_model_predictions = np.round(forest_model.predict(testdata_X)).astype(int)
#print("forest model prediction is" + "\n" + str(forest_model_predictions))
print("forest model prediction")
forest_model_data = pd.DataFrame({'PassengerId': testdata_X.PassengerId,'Survived': forest_model_predictions})
print(str(forest_model_data) + "\n")
#%%
sample_train = pd.DataFrame(traindata_after_modification)
sample_train.to_csv('traindata_after_modification.csv', index = False)
#%%
sample_test = pd.DataFrame(testdata_after_modification)
sample_test.to_csv('testdata_after_modification.csv', index = False)
#%%
prediction = pd.DataFrame(forest_model_data)
prediction.to_csv('submission.csv', index = False)