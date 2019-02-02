#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 12:44:36 2018

@author: geraldigunawan
"""

from DataMain import *
from scipy import stats

#%%
titanic_train_data.describe()
#%%w
#number of survivors
titanic_train_data[titanic_train_data['Survived'] == 1]['PassengerId'].count()
#%%
#number of casualties
titanic_train_data[titanic_train_data['Survived'] == 0]['PassengerId'].count()
#%%
titanic_train_data[titanic_train_data['Survived'] == 1]['Pclass'].value_counts().sort_index().plot.bar()
#%% class and cabin relation
titanic_train_data[titanic_train_data['Cabin'].notna()]['PassengerId'].count()
#%%
titanic_train_data[titanic_train_data['Cabin'].isna()]['PassengerId'].count()
#%%
list(titanic_train_data[titanic_train_data['Fare'] > 500].values)
#%%
#numer of rows with age missing
titanic_train_data[titanic_train_data['Age'].isna()]['PassengerId'].count()
#%%
#number of survivors based on age
titanic_train_data[titanic_train_data['Survived'] == 1]['Age'].value_counts().sort_index().plot().line()
#%% 
#number of casualties based on age
titanic_train_data[titanic_train_data['Survived'] == 0]['Age'].value_counts().sort_index().plot().line()
#%%
titanic_train_data['Pclass'].unique()
#%%
passengers_without_siblings = titanic_train_data[titanic_train_data['SibSp'] == 0]['PassengerId'].count()
passengers_with_siblings = titanic_train_data[titanic_train_data['SibSp'] > 0]['PassengerId'].count()
survived_without_siblings = titanic_train_data[(titanic_train_data['SibSp'] == 0) & (titanic_train_data['Survived'] == 1)]['PassengerId'].count()
survived_with_siblings = titanic_train_data[(titanic_train_data['SibSp'] > 0) & (titanic_train_data['Survived'] == 1)]['PassengerId'].count()
print ("total no of passenegers without siblings:", passengers_without_siblings)
print ("total no of passengers with siblings:", passengers_with_siblings)
print ("no of survivors without siblings:", survived_without_siblings)
print ("no of survivors with siblings:", survived_with_siblings)
#%%
titanic_train_data[titanic_train_data['Embarked'] == 'S']['PassengerId'].count()
#%%
titanic_train_data[titanic_train_data['Embarked'] == 'C']['PassengerId'].count()
#%%
titanic_train_data[titanic_train_data['Embarked'] == 'Q']['PassengerId'].count()