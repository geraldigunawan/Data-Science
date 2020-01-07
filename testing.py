#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 09:59:13 2019

@author: geraldigunawan
"""

import pandas as pd
import numpy as np

#%%
dictionary_1 = {'Name': 'Geraldi', 'Age':10, 'Employment':'CBA'}
dictionary_1
#%%
series_1 = pd.Series(dictionary_1)
series_1
#%%
dataframe_1 = pd.DataFrame(series_1).transpose()
dataframe_1
#%%
series_2 = pd.Series(['Abdul',20,'Usyd'], index=['Name', 'Age','Employment'])
series_2
#%%
dataframe_2 = pd.DataFrame(series_2).transpose()
dataframe_2
#%%
pd.concat([dataframe_1,dataframe_2], ignore_index = True)
#%%
dataframe_3 = pd.DataFrame([['Tommy',20,'Rassure'],['Hugh',20,'KodraMentha'],['Geraldi',21,'CBA']], columns=['Name', 'Age','Employment'])
dataframe_3
#%%
dataframe_5 = pd.DataFrame([['Hazard',40,'Hays']], columns = ['Name', 'Age','Employment'])
dataframe_5
#%%
dataframe_3 = dataframe_3.append(dataframe_5, ignore_index = True)
#%%
dataframe_3 = dataframe_3.drop([4,5])
#%%
dataframe_3.iloc[0:5]