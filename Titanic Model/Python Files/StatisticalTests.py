#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Nov 12 11:20:02 2018

@author: geraldigunawan
"""
from DataImputation import *
from scipy import *
from scipy.stats import chisquare
#%%
#no of survivors based on age after missing age value is populated with mean
result[result['Survived'] == 1]['Age'].value_counts().sort_index().plot().line()
#%%
#no of casulaties based on age after missing age value is populated with mean
result[result['Survived'] == 0]['Age'].value_counts().sort_index().plot().line()
#%%
"""
 check normal distribution of survivor based on age after missing value is filled with mean
 H0: data do not follow normal distribution
 H1: Cannot conclude the data do not follow a normal distribution
"""
k2, p = stats.normaltest(result[result['Survived'] == 1]['Age'])
print('p-value =', p)

if p <= 0.05:
    print ("Rejects the hypothesis of normality. The data do not follow normal distribution")
else:
    print ("there is not enough evidence to concude that the data do not follow normal distribution")
#%%
"""
Statistical test 1:  Is there a significant difference in terms of survival/casualties between gender? 

Hypothesis:          H0: There is no significant difference of survival/casualties rate between gender
                     H1: There is significant difference of survival/casualties rate between gender

Statistical method:  Pearson Chi Square

facts:               - 35% of passengers are female
                     - 65% of passengers are male
                     - total passengers count are 891
"""
male_survived = result[(result['Survived'] == 1) & (result['Sex'] == 'male')]['PassengerId'].count()
male_casualties = result[(result['Survived'] == 0) & (result['Sex'] == 'male')]['PassengerId'].count()
female_survived = result[(result['Survived'] == 1) & (result['Sex'] == 'female')]['PassengerId'].count()
female_casualties = result[(result['Survived'] == 0) & (result['Sex'] == 'female')]['PassengerId'].count()
print ('Observed_male survived: %d, Observed_male casualties: %d' % (male_survived, male_casualties))
print ('Observed_female survived: %d, Observed_female casualties: %d' % (female_survived, female_casualties) + "\n")

chi2, p = chisquare(f_obs = [male_survived,male_casualties,female_survived,female_casualties], f_exp = [222,356,120,193], ddof=1)
print ("Chi2 is", chi2)
print ("p-value is",p)
if p <= 0.05:
    print ("Rejects H0: There is significant difference of survival/casualties rate between gender")
else:
    print ("Retain H0: There is no significant difference of survival/casualties rate between gender")
#%%
"""
Statistical test 2:  Is there a significant difference in terms of survival/casualties for different ticket class?

Hypothesis:          H0: There is no significant difference of survival/casualties rate between ticket class
                     H1: There is significant difference of survival/casualties rate between ticket class

Statistical method:  Pearson Chi Square

facts:               - class 1 contains 216 passengers: 24.2%
                     - class 2 contains 184 passengers: 20.65%
                     - class 3 contains 491 passengers: 55.1%
"""
class1_survived = result[(result['Survived'] == 1) & (result['Pclass'] == 1)]['PassengerId'].count()
class2_survived = result[(result['Survived'] == 1) & (result['Pclass'] == 2)]['PassengerId'].count()
class3_survived = result[(result['Survived'] == 1) & (result['Pclass'] == 3)]['PassengerId'].count()
class1_casualties = result[(result['Survived'] == 0) & (result['Pclass'] == 1)]['PassengerId'].count()
class2_casualties = result[(result['Survived'] == 0) & (result['Pclass'] == 2)]['PassengerId'].count()
class3_casualties = result[(result['Survived'] == 0) & (result['Pclass'] == 3)]['PassengerId'].count()
print ('Class 1 survived: %d, Class 1 casualties: %d' % (class1_survived, class1_casualties))
print ('Class 2 survived: %d, Class 2 casualties: %d' % (class2_survived, class2_casualties))
print ('Class 3 survived: %d, Class 3 casualties: %d' % (class3_survived, class3_casualties))

chi2, p = chisquare(f_obs = [class1_survived,class2_survived,class3_survived,class1_casualties,class2_casualties,class3_casualties], f_exp = [83,71,188,133,113,303], ddof=2)
print ("Chi2 is", chi2)
print ("p-value is",p)
if p <= 0.05:
    print ("Rejects H0: There is significant difference of survival/casualties rate between ticket class")
else:
    print ("Retain H0: There is no significant difference of survival/casualties rate between ticket class")
#%%
"""
Statistical test 3:  Is there a significant difference in terms of survival/casualties for passengers with/without siblings?

Hypothesis:          H0: There is no significant difference of survival/casualties rate in terms of sibling relations
                     H1: There is significant difference of survival/casualties rate in terms of sibling relations

Statistical method:  Pearson Chi Square

facts:               - total no of passengers wihout siblings: 608
                     - total no of passengers with siblings: 283
""" 
survived_without_siblings = result[(result['SibSp'] == 0) & (result['Survived'] == 1)]['PassengerId'].count()
survived_with_siblings = result[(result['SibSp'] > 0) & (result['Survived'] == 1)]['PassengerId'].count()
not_survived_without_siblings = result[(result['SibSp'] == 0) & (result['Survived'] == 0)]['PassengerId'].count()
not_survived_with_siblings = result[(result['SibSp'] > 0) & (result['Survived'] == 0)]['PassengerId'].count()
print ("With siblings survived : %d, with siblings casualties: %d" % (survived_with_siblings,not_survived_with_siblings))
print ("Without siblings survived : %d, without siblings casualties: %d" % (survived_without_siblings,not_survived_without_siblings))

chi2, p = chisquare(f_obs = [survived_with_siblings,survived_without_siblings,not_survived_with_siblings,not_survived_without_siblings], f_exp = [109,233,174,375], ddof=1)
print ("Chi2 is", chi2)
print ("p-value is",p)
if p <= 0.05:
    print ("Rejects H0: There is significant difference of survival/casualties rate in terms of sibling relations")
else:
    print ("Retain H0: There is no significant difference of survival/casualties rate in terms of sibling relations")
#%%
"""
Statistical test 4:  Is there a significant difference in terms of survival/casualties for passenger who embarked from different places?

Hypothesis:          H0: There is no significant difference of survival/casualties rate in terms of embarkation
                     H1: There is significant difference of survival/casualties rate in terms of embarkation

Statistical method:  Pearson Chi Square

facts:               - 72.5% of passengers embarked from Souththampton (S)
                     - 18.85% of passengers embarked from Cherbourg (C)
                     - 8.64% of passengers embarked from Queenstown (Q)

""" 
S_survived = result[(result['Embarked'] == 'S') & (result['Survived'] == 1)]['PassengerId'].count()
C_survived = result[(result['Embarked'] == 'C') & (result['Survived'] == 1)]['PassengerId'].count()
Q_survived = result[(result['Embarked'] == 'Q') & (result['Survived'] == 1)]['PassengerId'].count()
S_casualties = result[(result['Embarked'] == 'S') & (result['Survived'] == 0)]['PassengerId'].count()
C_casualties = result[(result['Embarked'] == 'C') & (result['Survived'] == 0)]['PassengerId'].count()
Q_casualties = result[(result['Embarked'] == 'Q') & (result['Survived'] == 0)]['PassengerId'].count()
print ('S survived: %d, S casualties: %d' % (S_survived, S_casualties))
print ('C survived: %d, C casualties: %d' % (C_survived, C_casualties))
print ('Q survived: %d, Q casualties: %d' % (Q_survived, Q_casualties))

chi2, p = chisquare(f_obs = [S_survived,C_survived,Q_survived,S_casualties,C_casualties,Q_casualties], f_exp = [248,65,29,398,103,48], ddof=2)
print ("Chi2 is", chi2)
print ("p-value is",p)
if p <= 0.05:
    print ("Rejects H0: There is significant difference of survival/casualties rate in terms of embarkation")
else:
    print ("Retain H0: There is no significant difference of survival/casualties rate in terms of embarkation")