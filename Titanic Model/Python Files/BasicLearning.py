#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:36:28 2019

@author: geraldigunawan
"""
import numpy as np
import pandas as pd

np.version.version
#%% create dictionary
d = {'name':'Geraldi','age':23,'uni':'University of Sydney'}
#%%
d
#%%
pd.Series(d)
#%% create Series
s = pd.Series([1,2,3,4.0,'ala',6,7], index=['a','b','c','d','e','f','g'])
s
#%%
'g' in s
#%%
s.dtype
#%%
s2 = pd.Series(np.random.randint(5, size = 5))
s2
#%%
s2[1]
#%%
s3 = pd.Series(np.array(['a','b','c']))
s3
#%%
s4 = pd.Series(['Geraldi',23,'University of Sydney'])
s4
#%%
dff = pd.DataFrame({'id':s3, 'value':s4})
dff
#%%
df = pd.DataFrame([s3,s4]).transpose()
df
#%%
df.columns = ['Id','Info']
df
#%%
df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
df
#%%
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
df.append(df2,ignore_index=True)
#%%
df3 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]],columns=list('ABC'))
df3
#%%
df4 = pd.DataFrame([[2, 2, 3], [4, 5, 6],[np.nan,np.nan,np.nan]], columns=list('ABC'))
df4
#%%
result = df3.append(df4, ignore_index=True)
result
#%%
result.isnull()
#%%
s1 = pd.Series([np.nan, 2,np.nan], index =list('ABC'))
result.append(s1, ignore_index=True)
#%%
df = pd.DataFrame(columns=['A'])
df
#%%
df2[0:3]
# select the first, second and third rows from the surveys variable
#%%
df2[-1:]
#%%
#%%select subset from rows and columns
df2.iloc[0:3,1:2]
#%%
df2.iloc[0:2,2:3]

#%%Converting dict to series of dataframe
data_dict = {'Name' : 'Andy', 'Age' : 24, 'Uni' : 'University of Sydney'}
data_dict
#%%
data_series = pd.Series(data_dict)
data_series
#%%
df = pd.DataFrame.from_dict(data_dict)
df
#%%
df = pd.DataFrame(data_series).transpose()
df