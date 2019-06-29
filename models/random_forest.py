#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:48:59 2017

@author: pgrimshaw
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from operator import itemgetter
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

import utils.grid_search as gs

#---------------------------
# This script  tunes performance of a random forest model and applies it to the dataset
#---------------------------

  
def run_RF(X, y):

    clf = RandomForestRegressor(n_estimators=50)
   
    gs.run_eval_gs(X,y,clf,cv=5,
                   gs_params={'max_features': [0.5,0.75], 'min_samples_leaf': [10,50,200]})
    
    #Based on observations refine parameter set
    gs.run_eval_gs(X,y,clf,cv=5,
                   gs_params={'max_features': np.arange(0.3,0.9,0.1) , 'min_samples_leaf': [5,10,20,30]})
    
    
    #Select optimal parameters and apply the tuned model to the dataset
    clf = RandomForestRegressor(max_features=0.5,min_samples_leaf=10,n_estimators=50)
    clf.fit(X,y)
    
    #Report performance and features of model
    print('Mean squared error %6.4f' % mse(y,clf.predict(X)))
    print('R^2 score %6.4f' % clf.score(X,y))
    scores = cross_validate(clf,X,y,cv=5,scoring=make_scorer(mse))
    print(np.mean(scores['test_score']))
    
    print('Feature Importances')
    for coef, col in sorted(zip(clf.feature_importances_,X.columns),key=itemgetter(0)):
        print(col, round(coef,3))
        
