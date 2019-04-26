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
"""Read train and test data"""
loan_train_data = pd.read_csv('/Applications/MAMP/htdocs/Data-Science/Loan Prediction Model/train_u6lujuX_CVtuZ9i.csv')
loan_test_data = pd.read_csv('/Applications/MAMP/htdocs/Data-Science/Loan Prediction Model/test_Y3wMUE5_7gLdaTN.csv')
#%%
"""Make a local copy to ensure original data is not modified"""
train_original=loan_train_data.copy() 
test_original=loan_test_data.copy()
#%%
"""Briefly describe datasets"""
list(train_original)
train_original.columns
train_original.head()
train_original.shape
train_original.dtypes
#%%
train_original['CoapplicantIncome'].count()
#%%
train_original[train_original['CoapplicantIncome'] == 0.0]['CoapplicantIncome'].count()
#%%
gender_married_table = pd.crosstab(train_original['Gender'], train_original['Married'])
gender_married_table
#%%
gender_married_table.dropna().plot.bar(title = 'Marriage based gender', stacked = True, figsize = (4,4))
#%%
train_original[train_original['Gender'] == 'Male']['Gender'].count()
train_original[train_original['Gender'] == 'Female']['Gender'].count()
train_original['Gender'].fillna('Male', inplace = True)
#%%
train_original['Loan_Status'].value_counts()
train_original['Loan_Status'].value_counts(normalize=True)
#%%
train_original['Loan_Status'].value_counts().plot.bar()
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
train_original.boxplot(column='ApplicantIncome', by = ('Education'))
plt.title('Applicant Income by Education')
plt.suptitle("")
#%%
sns.distplot(train_original['CoapplicantIncome'])
#%%
train_original['CoapplicantIncome'].plot.box()
#%%
train_original.boxplot(column='CoapplicantIncome', by = 'Education')
#%%
df=train_original.dropna() 
df.isnull().any()
df.shape
sns.distplot(df['LoanAmount']);
sns.distplot(df['LoanAmount'], kde = False ,hist = True); 
sns.distplot(df['LoanAmount'], kde = True ,hist = False); 
#%%
"""
Hypothesis:
    
- Applicants with high income should have more chances of loan approval.
- Applicants who have repaid their previous debts should have higher chances of loan approval.
- Loan approval should also depend on the loan amount. If the loan amount is less, chances of loan approval should be high.
- Lesser the amount to be paid monthly to repay the loan, higher the chances of loan approval.
"""
#%%
"""Check if loan approval has direct correlation over gender"""
LoanApproval_by_gender = pd.crosstab(train_original['Gender'],train_original['Loan_Status'])
LoanApproval_by_gender
LoanApproval_by_gender.div(LoanApproval_by_gender.sum(1).astype(float), axis=0)
LoanApproval_by_gender.div(LoanApproval_by_gender.sum(1).astype(float), axis=0).plot.bar(stacked= True, title = 'Loan Approval based on gender')
#%%
"""Check if loan approval has direct correlation over marriage"""
LoanApproval_by_marriage = pd.crosstab(train_original['Married'],train_original['Loan_Status'])
LoanApproval_by_marriage.div(LoanApproval_by_marriage.sum(1).astype(float), axis=0).plot.bar(stacked= True, title = 'Loan Approval based on marriage')
#%%
"""Check if loan approval has direct correlation over marriage"""
LoanApproval_by_dependents = pd.crosstab(train_original['Dependents'],train_original['Loan_Status'])
LoanApproval_by_dependents.div(LoanApproval_by_dependents.sum(1).astype(float), axis=0).plot.bar(stacked= True, title = 'Loan Approval based on dependents')
#%%
"""Check if loan approval has direct correlation over education"""
LoanApproval_by_education = pd.crosstab(train_original['Education'],train_original['Loan_Status'])
LoanApproval_by_education.div(LoanApproval_by_education.sum(1).astype(float), axis=0).plot.bar(stacked= True, title = 'Loan Approval based on education')
#%%
"""Check if loan approval has direct correlation over employment"""
LoanApproval_by_employment = pd.crosstab(train_original['Self_Employed'],train_original['Loan_Status'])
LoanApproval_by_employment.div(LoanApproval_by_employment.sum(1).astype(float), axis=0).plot.bar(stacked= True, title = 'Loan Approval based on employment')
#%%
"""Check if loan approval has direct correlation over credit history"""
LoanApproval_by_credit_history = pd.crosstab(train_original['Credite_History'],train_original['Loan_Status'])
LoanApproval_by_credit_history.div(LoanApproval_by_credit_history.sum(1).astype(float), axis=0).plot.bar(stacked= True, title = 'Loan Approval based on credit history')
#%%
"""Check if loan approval has direct correlation over property area"""
LoanApproval_by_property_area = pd.crosstab(train_original['Property_Area'],train_original['Loan_Status'])
LoanApproval_by_property_area.div(LoanApproval_by_property_area.sum(1).astype(float), axis=0).plot.bar(stacked= True, title = 'Loan Approval based on property area')
txt = 'It can be inferred that applicants who live in semiurban area has higher chance to get their loan approved'
plt.text(-0.5,-0.5, txt)
#%%
"""Find the mean income of people for which the loan has been approved vs the mean income of people for which the loan has not been approved"""
train_original.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()

bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train_original['Income_bin'] = pd.cut(df['ApplicantIncome'],bins,labels=group)
Income_bin = pd.crosstab(train_original['Income_bin'],train_original['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
Income_bin
plt.xlabel('ApplicantIncome')
plt.ylabel('Percentage')
plt.title('Loan approval based on applicant income categories')
txt= 'It can be inferred that Applicant income does not affect the chances of loan approval which contradicts our hypothesis in which we assumed that if the applicant income is high the chances of loan approval will also be high.'
plt.text(-0.5, -0.5, txt, ha='center')
#%%
"""Find the mean income of people for which the loan has been approved vs the mean income of people for which the loan has not been approved"""
train_original.groupby('Loan_Status')['CoapplicantIncome'].mean().plot.bar()
#%%
bins = [0,1000,3000,42000]
group=['Low','Average','High']
train_original['Coapplicant_income_bin'] = pd.cut(df['CoapplicantIncome'],bins,labels=group)
Coapplicant_income_bin = pd.crosstab(train_original['Coapplicant_income_bin'],train_original['Loan_Status'])
Coapplicant_income_bin.div(Coapplicant_income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
Coapplicant_income_bin
plt.xlabel('CoapplicantIncome')
plt.ylabel('Percentage')
plt.title('Loan approval based on coapplicant income categories')
txt= 'It can be inferred that lower coapplicant income has higher chance of loan approval, which is a bit odd. This could be because some coapplicants do not have income.'
plt.text(-0.5, -0.5, txt, ha='center')
#%% Combining applicant and coapplicant income
train_original['CombinedIncome'] = train_original['ApplicantIncome'] + train_original['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
train_original['Total_Income_bin'] = pd.cut(train_original['CombinedIncome'],bins,labels=group)
Total_income_bin = pd.crosstab(train_original['Total_Income_bin'], train_original['Loan_Status'])
Total_income_bin.div(Total_income_bin.sum(1).astype(float), axis = 0).plot(kind = "bar", stacked = True)
plt.title('Loan approval based on total income of both applicant and coapplicant')
plt.xlabel('Applicant + coapplicant income')
plt.ylabel('Percentage')
txt= 'After combining applicant and coapplicant income, it makes more sense now that lower income results to lower chance of loan approval'
plt.text(-0.5, -0.5, txt, ha='center')
#%%
train_original.columns
#%%
train_original = train_original.drop(['Income_bin', 'Coapplicant_income_bin', 'CombinedIncome',
       'Total_Income_bin'], axis = 1)
#%%
train_original['Dependents'].replace('3+',3, inplace= True)
train_original['Loan_Status'].replace('N', 0,inplace=True)
train_original['Loan_Status'].replace('Y', 1,inplace=True)
train_original['Loan_Status'].unique()
#%%
matrix = train_original.corr()
matrix
plt.subplots(figsize=(5.0, 5.0))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
#%%
"""check missing values"""
train_original.isnull().sum()
#%%
train_original['Married'].loc[(train_original['Loan_ID'] == 'LP001357') | (train_original['Loan_ID'] == 'LP001760') | (train_original['Loan_ID'] == 'LP002393')]
#%%
train_original['Married'].mode()
#%%
"""impute train data with mode"""
train_original['Gender'].fillna(train_original['Gender'].mode()[0], inplace=True) 
train_original['Married'].fillna(train_original['Married'].mode()[0], inplace=True) 
train_original['Dependents'].fillna(train_original['Dependents'].mode()[0], inplace=True)
train_original['Self_Employed'].fillna(train_original['Self_Employed'].mode()[0], inplace=True)
train_original['Credit_History'].fillna(train_original['Credit_History'].mode()[0], inplace=True)
#%%
train_original.head()
#%%
train_original['Loan_Amount_Term'].value_counts()
train_original['Loan_Amount_Term'].fillna(train_original['Loan_Amount_Term'].mode()[0], inplace=True)
#%%
train_original['LoanAmount'].value_counts()
#%%
train_original['LoanAmount'].fillna(train_original['LoanAmount'].median(), inplace=True)
#%%
test_original.head()
test_original.isnull().sum()
#%%
"""impute test data with mode or median"""
test_original['Gender'].fillna(test_original['Gender'].mode()[0], inplace=True) 
test_original['Dependents'].fillna(test_original['Dependents'].mode()[0], inplace=True)
test_original['Self_Employed'].fillna(test_original['Self_Employed'].mode()[0], inplace=True)
test_original['Credit_History'].fillna(test_original['Credit_History'].mode()[0], inplace=True)
test_original['Loan_Amount_Term'].fillna(test_original['Loan_Amount_Term'].mode()[0], inplace=True)
test_original['LoanAmount'].fillna(test_original['LoanAmount'].median(), inplace=True)
#%%
train_original['LoanAmount'].value_counts()
#%%
train_original['LoanAmount'].plot(kind ='hist', bins = 20, grid = True)
#%%
"""treating outlier"""
train_original['LoanAmount_log'] = np.log(train_original['LoanAmount'])
#%%
train_original['LoanAmount_log'].hist(bins=20)
#%%
test_original['LoanAmount'].plot(kind ='hist', bins = 20, grid = True)
#%%
test_original['LoanAmount_log'] = np.log(test_original['LoanAmount'])
test_original['LoanAmount_log'].plot(kind ='hist', bins = 20, grid = True)
#%%
train_original.isnull().sum()
#%%
test_original.isnull().sum()