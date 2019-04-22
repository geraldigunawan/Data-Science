#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:21:16 2019

@author: geraldigunawan
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
pd.set_option('display.max_columns', 100)


#%%
loan_train_data = pd.read_csv('/Applications/MAMP/htdocs/Data-Science/Loan Prediction Model/train_u6lujuX_CVtuZ9i.csv')
loan_test_data = pd.read_csv('/Applications/MAMP/htdocs/Data-Science/Loan Prediction Model/test_Y3wMUE5_7gLdaTN.csv')
#%%
train_original=loan_train_data.copy() 
test_original=loan_test_data.copy()
#%%
list(train_original)
#%%
train_original.head()
#%% check tuple representing the dimensionality of the DataFrame
train_original.shape
#%% check missing values
train_original.isnull().sum()
#%%
train_original[train_original['Gender'].isnull()]['Loan_ID']
#%%
train_original[train_original['Gender'].isnull()]['Gender'].count()
#%%
gender_married_table = pd.crosstab(train_original['Gender'], train_original['Married'])
#%%
gender_married_table
#%%
gender_married_table.dropna().plot.bar(title = 'Marriage based gender', figsize = (4,4))
#%%
train_original[train_original['Gender'] == 'Male'].count()
#%%
train_original[train_original['Gender'] == 'Female'].count()
#%%
train_original['Gender'].fillna('Male', inplace = True)
#%%
train_original['Loan_Status'].value_counts()
train_original['Loan_Status'].value_counts(normalize=True)
#%%
train_original['Loan_Status'].value_counts().plot.bar()
#%%
train_original.dtypes
#%%
train_original['Gender'].value_counts(normalize=True).plot.bar(title= 'Gender')
#%%
train_original['Married'].value_counts(normalize=True).plot.bar(title= 'Married') 
#%%
train_original['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')
#%%
train_original['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')
#%%
train_original['Dependents'].value_counts(normalize=True).plot.bar(title= 'Dependents')
#%%
train_original['Education'].value_counts(normalize=True).plot.bar(title= 'Education')
#%% 
train_original['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')
#%%
sns.distplot(train_original['ApplicantIncome'])
#%%
train_original['ApplicantIncome'].plot.box() 
#%%
train_original.boxplot(column='ApplicantIncome', by = ('Education','Married'))
#%%
sns.distplot(train_original['CoapplicantIncome'])
#%%
train_original['CoapplicantIncome'].plot.box()
#%%
train_original.boxplot(column='CoapplicantIncome', by = 'Education')
#%%
df=train_original.dropna() 
#%%
df.isnull().any()
#%%
df.shape
#%%
sns.distplot(df['LoanAmount']);
#%%
sns.distplot(df['LoanAmount'], kde = False ,hist = True); 
#%%
sns.distplot(df['LoanAmount'], kde = True ,hist = False); 
#%%
Gender = pd.crosstab(train_original['Loan_Status'],train_original['Gender']) 
#%%
Gender
#%%
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))