#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:48:59 2017

@author: pgrimshaw
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from operator import itemgetter
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

import utils.grid_search as gs

#---------------------------
# This script  tunes performance of a gradient boosted regression model
#---------------------------

def run_gbr(X, y):
    
    clf = GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators=50,subsample=1.0)

    #Try a small number of parameters

    gs_params = {'max_features': [0.5,0.75], 'max_depth': [2,3,4], 'min_samples_split': [2,4,8], \
             'min_samples_leaf': [1,2,4]}

    gs.run_eval_gs(X,y,clf,cv=5,gs_params=gs_params)
    
    #It's clear that the values of max_features=0.75 and max_depth=4 are optimal, fix these
    #and test other parameters


    clf = GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators=50,subsample=1.0, \
                                max_features=0.75, max_depth=4)
    gs_params = {'min_samples_split': [2,4,8], 'min_samples_leaf': [1,2,4]}

    gs.run_eval_gs(X,y,clf,cv=5,gs_params=gs_params)

    #There is clearly interaction between min_samples_leaf and min_samples_split. The smaller the value, the
    #better the performance should be and overfitting is addressed by the cross validation here, so choose small
    #values here, min_samples_leaf=2 and min_samples_split=4

    clf = GradientBoostingRegressor(loss='ls',learning_rate=0.1,n_estimators=50,subsample=1.0, \
                                max_features=0.75, max_depth=4,min_samples_leaf=2, min_samples_split=4)
    clf.fit(X,y)

    #Report performance and features of model
    print('Mean squared error %6.4f' % mse(y,clf.predict(X)))
    print('R^2 score %6.4f' % clf.score(X,y))
    scores = cross_validate(clf,X,y,cv=10,scoring=make_scorer(mse))
    print(np.mean(scores['test_score']))
    
    for coef, col in sorted(zip(clf.feature_importances_,X.columns),key=itemgetter(0)):
        print(col, round(coef,3))
    
