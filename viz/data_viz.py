#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:52:36 2017

@author: pgrimshaw
"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

#---------------------------
# This section plots a heatmap of correlation between all credit-related variables
#---------------------------


def gen_credit_corr_heatmap(df):
     
    #Plot the heatmap visualising correlation 
    plt.figure(figsize=(15,15))
    corr_df = df.select_dtypes(['int64','float64']).astype(float).corr()
    for col in corr_df:
        corr_df[col] = corr_df[col].apply(lambda x: round(x,2))
    sns.heatmap(corr_df,robust=True,\
                annot=True,cbar=True,square=True,cmap='PuBuGn')
    plt.title('Credit Variables Correlation Heatmap')
    plt.savefig('viz/credit_heatmap.jpg')
    plt.clf()

#---------------------------
# This section plots each of the independent variables against the dependent variable loan subgrade
# Different charts are used for each variable to present the data clearly
#---------------------------

def gen_sub_grade_plots(df):
    
    fig, ax = plt.subplots(3,2,figsize=(16,24))
    
    #Loan Subgrade vs Loan Amount
    hb = ax[0,0].hexbin(df.sub_grade_num,df.loan_amnt,gridsize=15,cmap='Greens')
    cb = plt.colorbar(hb,ax=ax[0,0])
    cb.set_label('Number of Loans in Hexbin')
    ax[0,0].set(xlabel='Loan Subgrade',ylabel='Loan Amount')
    
    #Loan Subgrade vs Annual Income
    std_by_subgrade = df[['sub_grade_num','annual_inc']].groupby('sub_grade_num')['annual_inc'].std()
    mean_by_subgrade = df[['sub_grade_num','annual_inc']].groupby('sub_grade_num')['annual_inc'].mean()
    ax[0,1].fill_between(range(max(df.sub_grade_num)+1), \
                     mean_by_subgrade+std_by_subgrade, \
                     mean_by_subgrade-std_by_subgrade, \
                     color='#a4c9ef')
    ax[0,1].plot(mean_by_subgrade,c='b')
    ax[0,1].set(xlabel='Loan Subgrade',ylabel='Annual Income')
    blue_patch = mpatches.Patch(color='#a4c9ef', label='One standard deviation')
    ax[0,1].legend(handles=[blue_patch])
    
    
    #Loan Subgrade vs Debt to income ratio
    std_by_subgrade = df[['sub_grade_num','dti']].groupby('sub_grade_num')['dti'].std()
    mean_by_subgrade = df[['sub_grade_num','dti']].groupby('sub_grade_num')['dti'].mean()
    ax[1,0].fill_between(range(max(df.sub_grade_num)+1), \
                     mean_by_subgrade+std_by_subgrade, \
                     mean_by_subgrade-std_by_subgrade, \
                     color='#ed9797')
    ax[1,0].plot(df[['sub_grade_num','dti']].groupby('sub_grade_num')['dti'].mean(),c='r')
    ax[1,0].set(xlabel='Loan Subgrade',ylabel='Debt to Income Ratio')
    red_patch = mpatches.Patch(color='#ed9797', label='One standard deviation')
    ax[1,0].legend(handles=[red_patch])
    
    
    #Loan Subgrade vs Employment Length
    hb2 = ax[1,1].hexbin(df.sub_grade_num,df.emp_length_rank,gridsize=15,cmap='Blues')
    cb = plt.colorbar(hb,ax=ax[1,1])
    cb.set_label('Number of Loans in Hexbin')
    ax[1,1].set(xlabel='Loan Subgrade',ylabel='Employment Length')
    
    #Loan Subgrade vs Home Ownership
    for val in set(df.home_ownership):
            df_counts = df[['sub_grade_num','home_ownership']][df.home_ownership==val].groupby('sub_grade_num').size()
            ax[2,0].plot(df_counts/df[['sub_grade_num']].groupby('sub_grade_num').size(),label=val)
    ax[2,0].legend(loc='upper right')
    ax[2,0].set(xlabel='Loan Subgrade',ylabel='Count of Loans')
    
    #Loan Subgrade vs FICO Score
    df_filter = (df.index%5==0)
    ax[2,1].scatter(df.sub_grade_num[df_filter], df.fico_range_low[df_filter])
    ax[2,1].set(xlabel='Loan Subgrade',ylabel='FICO Score')
    
    for i in range(2):
        for j in range(1):
            ax[i,j].set_xlabel('Loan Subgrade')
            
    plt.savefig('viz/subgrade plots.jpg')
    plt.clf()

#---------------------------
# Create a header chart for the paper visualising loan amount and annual income
#---------------------------

def gen_loan_income_scatter(df):
    plt.figure(figsize=(12,5))
    df_filter = (df.index%1==0)
    plt.scatter(df.loan_amnt[df_filter], df.annual_inc[df_filter],c=df.sub_grade_num[df_filter] \
                ,cmap='summer')
    cb = plt.colorbar()
    cb.set_label('Loan Subgrade (Decreasing Quality)')
    ax = plt.gca()
    ax.set_ylim((0,300000))
    plt.xlabel('Loan Amount')
    plt.ylabel('Annual Income')
    plt.title('Loan Amount versus Annual Income and Loan Grade')
    plt.savefig('viz/loan_income_scatter.jpg')
    plt.clf()
    
def plot_linear_models(model_set,X,alphas):
    
    linear_model, ridge_models, lasso_models = model_set
    
    plot_lin_mdl_coefs__(X,alphas,[mdl[2] for mdl in ridge_models],'viz/ridg reg perf.jpg', 
                       'Regularisation vs Coefficient Values, Ridge Regression')
    plot_lin_mdl_coefs__(X,alphas,[mdl[2] for mdl in lasso_models],'viz/las reg perf.jpg',
                       'Regularisation vs Coefficient Values, Lasso Regression')
    
    plot_lin_mdl_metric__([mdl[0] for mdl in ridge_models],[mdl[0] for mdl in lasso_models],
                        'Coefficient of Determination',alphas,'viz/Regularisation r2 performance.jpg')
    plot_lin_mdl_metric__([mdl[1] for mdl in ridge_models],[mdl[1] for mdl in lasso_models],
                        'Mean Squared Error',alphas,'viz/Regularisation mse performance.jpg')

#Function to plot coefficients against alpha parameter for lasso and ridge models
def plot_lin_mdl_coefs__(X,alphas,coeffs,img_name, plot_title):
    plt.figure(figsize=(10,10))
    for i, col in enumerate(X.columns):
        feature_coefficients = [coeffs[j][i] for j, al in enumerate(alphas)]
        plt.xscale('log')
        plt.plot(alphas, feature_coefficients,label=X.columns[i],linestyle='-',marker='o')
    plt.xlabel('Lambda')
    plt.ylabel('Coefficient value')
    plt.title(plot_title)
    plt.legend()
    plt.savefig(img_name)
    plt.clf()

#Function to plot performance of lasso and ridge models
def plot_lin_mdl_metric__(lasso_metric,ridge_metric,metric_label,alphas,img_name):
    plt.plot(alphas,lasso_metric,label='Lasso')
    plt.plot(alphas,ridge_metric,label='Ridge')
    plt.xlabel('Regularisation Parameter')
    plt.ylabel(metric_label)
    plt.xscale('log')
    plt.legend()
    plt.savefig(img_name)
    plt.clf()
    
