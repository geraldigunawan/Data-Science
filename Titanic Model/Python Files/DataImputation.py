#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:53:34 2018

@author: geraldigunawan
"""

from DataMain import *

#%%
titanic_with_extra_columns = add_extra_column_to_indicate_missing_value(train_data_copy)
#%%
titanic_with_age_filled_with_mean = populate_missing_age(titanic_with_extra_columns)
#%%
titanic_with_embarkation_filled = populate_missing_embarkation_port(titanic_with_age_filled_with_mean)
#%%
result = titanic_with_embarkation_filled