#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:33:23 2018

@author: pgrimshaw
"""


def get_fields_complete():
    # Due to the size of the dataset refine columns to only useful ones which will be used further
    field_complete = ['loan_amnt', 'term', 'int_rate', 'grade', 'sub_grade', 'emp_length', 'home_ownership', \
                      'annual_inc', 'loan_status', 'pymnt_plan', 'purpose', 'title', 'zip_code', 'dti', \
                      'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', \
                      'acc_now_delinq', 'last_pymnt_d', 'fico_range_low', 'fico_range_high']
    return field_complete


# All fields relevant to creditworthiness
def get_fields_credit():
    fields_credit = ['sub_grade_num', 'loan_amnt', 'term', 'annual_inc', 'dti', \
                     'emp_length_rank', 'delinq_2yrs', 'inq_last_6mths', \
                     'open_acc', 'revol_bal', 'revol_util', 'total_acc', \
                     'acc_now_delinq', 'home_ownership', 'fico_range_low', 'int_rate']
    return fields_credit


# All fields relevant to the loan subgrade assigned
def get_fields_loan_subgrade():
    fields_loan_subgrade = ['grade', 'sub_grade_num', 'loan_amnt', 'term', 'annual_inc' \
        , 'dti', 'emp_length_rank', 'home_ownership', 'fico_range_low']
    return fields_loan_subgrade


# Fields that may be predictors for loan_subgrade

def get_features_loan_subgrade():
    features_loan_subgrade = ['loan_amnt', 'term', 'annual_inc', 'dti', 'emp_length_rank', \
                              'home_ownership', 'fico_range_low']
    return features_loan_subgrade
