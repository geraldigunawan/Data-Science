#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 11:52:30 2018

@author: geraldigunawan
"""

#%%
import pandas as pd

main_file_path = 'Melbourne house models/Datasets/train.csv' # this is the path to the Iowa data that you will use
data = pd.read_csv(main_file_path)

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left
home_sales_price = data.SalePrice
print(home_sales_price.head())

columns_of_interest = ['Fireplaces','GarageArea','WoodDeckSF','SalePrice']
house_price_factors = data[columns_of_interest]

print(house_price_factors.describe())
