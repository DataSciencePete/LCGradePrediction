#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 18:54:30 2017

@author: pgrimshaw
"""

import numpy as np

from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer

#---------------------------
# This section creates helper functions for reporting model performance
#---------------------------

#Function for reporting on linear model performance. Prints coefficients, intercept and R^2 for linear model

def lin_mdl_perf__(model,X,y,cv):
    print('Model features and coefficients:')
    get_coef = lambda x: x[0]
    for coef, col in sorted(zip(model.coef_,X.columns),key=get_coef):
        print(col, round(coef,3))
    print('Mean squared error %6.4f' % mse(y,model.predict(X)))
    print('R^2 score %6.4f' % model.score(X,y))
    scores = cross_validate(model,X,y,cv=cv)
    mean_test_score_r2 = np.mean(scores['test_score'])
    print('Cross validation mean R^2 score %6.4f, standard deviation %6.4f and \
          scores:' % (mean_test_score_r2, np.std(scores['test_score'])))
    print(scores)
    scores = cross_validate(model,X,y,cv=cv,scoring=make_scorer(mse))
    mean_test_score_mse = np.mean(scores['test_score'])
    print('Cross validation MSE score %6.4f,standard deviation %6.4f and \
          scores:' % (mean_test_score_mse, np.std(scores['test_score'])))
    print(scores)
    print('\n')
    return mean_test_score_r2, mean_test_score_mse, model.coef_
    
#---------------------------
# This section creates a linear regression model and reports performance of the model
#---------------------------

def run_eval_lin_mdl__(X,y,cv):
    print('------------Linear Regression Model-------------')
    
    mdl = LinearRegression(normalize=True)
    print('Linear regression model fitted on complete data:')
    mdl.fit(X,y)
    return lin_mdl_perf__(mdl,X,y,cv)

#---------------------------
# This section creates a regularided regression models and reports performance of the models
#---------------------------

def run_eval_ridge_mdl__(X,y,alphas,cv):
    print('------------Ridge Regression Model-------------')
    
    #Variable to store model performance for each alpha
    ridge_models = []
    
    #Evaluate performance for each alpha
    for al in alphas:
        mdl = Ridge(alpha=al,normalize=True)
        mdl.fit(X,y)
        print('Model performance for alpha=%8.6f' % al)
        ridge_models.append(lin_mdl_perf__(mdl,X,y,cv))
    return ridge_models
        
def run_eval_lasso_mdl__(X,y,alphas,cv):
    print('------------Lasso Regression Model-------------')

    #Variable to store model performance for each alpha
    lasso_models = []
    
    #Evaluate performance for each alpha
    for al in alphas:
        mdl = Lasso(alpha=al,normalize=True)
        mdl.fit(X,y)
        print('Model performance for alpha=%8.6f' % al)
        lasso_models.append(lin_mdl_perf__(mdl,X,y,cv))
    return lasso_models

#---------------------------
# This section plots model performance against regularisation parameter
#---------------------------

def run_eval_linear_models(X,y,alphas,cv):

    #Permute x and y rows to avoid bias in certain groups of rows, affecting cross validation
    X, y = shuffle(X,y,random_state=0)

    
    linear_model = run_eval_lin_mdl__(X,y,cv)
    ridge_models = run_eval_ridge_mdl__(X,y,alphas,cv)
    lasso_models = run_eval_lasso_mdl__(X,y,alphas,cv)
    
    return linear_model, ridge_models, lasso_models



    