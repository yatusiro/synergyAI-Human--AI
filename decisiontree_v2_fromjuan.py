# -*- coding: utf-8 -*-
"""DecisionTree_v2_fromJuAn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qsB314F3m3TfUQeiddeEwikMx5fOYL3A
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import train_test_split

# import data
data = pd.read_csv("v2_chocolateData.csv")
data.head()

data.shape
# data.info

#  feature = data[['Drying_days','roaster_time','roaster_temperature','coca_melting_temperature','coca_melting_time','coca_melting_rotate_speed','Sugar_Mixer_temperature','Sugar_Mixer_rotate_speed','Sugar_Mixer_time','Conching_time','conching_temperature','conching_rotate_speed','pump1_temperature','pump1_pressure','pump1_Flow_rate','melting_tank_temperature','melting_tank_time','pump2_temperature','pump2_pressure','pump2_Flow_rate','tempering_machine_temperature','tempering_machine_time','tempering_machine_rotate_speed','pump3_temperature','pump3_pressure','pump3_Flow_rate','moulding_machine_temperature','moulding_machine_time','sweetness','color','hardness','glossiness','shape','bitterness']]
feature = data[['Conching_time','conching_temperature','coca_melting_temperature','coca_melting_rotate_speed','pump1_Flow_rate','tempering_machine_time','pump2_temperature','pump3_pressure','tempering_machine_temperature','moulding_machine_time','Sugar_Mixer_temperature','moulding_machine_temperature','melting_tank_time','pump2_Flow_rate','pump3_Flow_rate','color','hardness','shape','glossiness']]
target = data[['passed']]

# split data for train and test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(feature, target, test_size=0.3, random_state=420)

Xtrain

# decision tree
model_DTC = DTC(random_state=25)
model_DTC = model_DTC.fit(Xtrain, Ytrain)
accuracy = model_DTC.score(Xtest, Ytest)
accuracy

from sklearn import svm
model_svm = svm.SVR()
model_svm.fit(Xtrain, Ytrain) # train model

accuracy = model_svm.score(Xtest, Ytest)
accuracy

from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()

model_ExtraTreeRegressor.fit(Xtrain, Ytrain) # train model

accuracy = model_ExtraTreeRegressor.score(Xtest, Ytest)
accuracy

####3.1 decision tree####·
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
####3.2 Linear Model####·
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
####3.3 SVM####·
from sklearn import svm
model_SVR = svm.SVR()
####3.4 KNN####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
####3.5 random forest####·
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#20 trees
####3.6 Adaboost####
from sklearn import ensemble
model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#50 trees
####3.7 GBRT####·
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#100 trees
####3.8 Bagging####
from sklearn.ensemble import BaggingRegressor
model_BaggingRegressor = BaggingRegressor()
####3.9 ExtraTree####
from sklearn.tree import ExtraTreeRegressor
model_ExtraTreeRegressor = ExtraTreeRegressor()
####3.10 ARD
model_ARDRegression = linear_model.ARDRegression()
####3.11 BayesianRidge
model_BayesianRidge = linear_model.BayesianRidge()
####3.12 TheilSen
model_TheilSenRegressor = linear_model.TheilSenRegressor()
####3.13 RANSAC
model_RANSACRegressor = linear_model.RANSACRegressor()

model_BaggingRegressor.fit(Xtrain, Ytrain) # train model

accuracy = model_BaggingRegressor.score(Xtest, Ytest)
accuracy