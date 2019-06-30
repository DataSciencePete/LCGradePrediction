#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 08:26:45 2018

@author: pgrimshaw
"""
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import datetime as dt


def load(filepath):
    # Import the data and enforce data types on mixed columns
    df = pd.read_csv(filepath, dtype={'desc': str, 'verification_status_joint': str, 'next_pymnt_d': str}, \
                     encoding='latin-1', low_memory=True, skiprows=1)
    return df


def load_dict(filepath):
    data_dict = pd.read_excel(filepath, 'LoanStats').set_index('LoanStatNew').to_dict()['Description']
    return data_dict


# Helper function to print dataframe completeness information
def print_completeness_summary(df, df_description, completeness_threshold, data_dictionary):
    df_completeness = []
    for col in df.columns:
        # Identify columns which are mostly complete
        col_completeness = df[col].count() / float(len(df.index))
        df_completeness.append(col_completeness)
        if col_completeness >= completeness_threshold:
            try:
                col_definition = data_dictionary[col]
            except KeyError as e:
                col_definition = 'None provided'
            print('{0:20s} {1:6.4f} {2:80s}'.format(col, col_completeness, col_definition))
    print('Overall proportion of missing values is %6.4f for %s' % (np.mean(df_completeness), df_description))


# ---------------------------
# This section performs feature engineering and outputs the dataset in a file for further analysis
# ---------------------------

def engineer_features(df):
    # Disable warning about chained pandas operations. This is not relevant since we are
    # writing back to the dataframe here anyway
    pd.options.mode.chained_assignment = None

    # Remove any rows which still have null values
    print('Length of dataframe before dropping rows is %d' % len(df.index))
    df = df.dropna(axis=0, how='any')
    print('Length of dataframe after dropping rows is %d' % len(df.index))

    # Create calculated fields to create other useful features for possible analysis
    def diff_month(d1, d2):
        return 12 * (d1.year - d2.year) + d1.month - d2.month

    # Convert string data type to numerical where needed in calculations
    df['last_pymnt_d'] = df['last_pymnt_d'].apply(lambda date_string: dt.datetime.strptime(date_string, '%b-%Y'))
    df['term'] = df['term'].apply(lambda x: int(x[1:3]))
    emp_rank = {'n/a': 0, '< 1 year': 1, '1 year': 2, '2 years': 3, '3 years': 4, '4 years': 5, '5 years': 6, \
                '6 years': 7, '7 years': 8, '8 years': 9, '9 years': 10, '10+ years': 11}
    df['emp_length_rank'] = df['emp_length'].replace(emp_rank)
    perc_string_to_float = lambda x: float(x.replace('%', ''))
    df.int_rate = df.int_rate.apply(perc_string_to_float)

    # Define new column of months since payment
    max_payment_date = max(df['last_pymnt_d'])
    df['mths_since_pymnt'] = df['last_pymnt_d'].apply(lambda lpd: diff_month(max_payment_date, lpd))

    # Add in numerical column for loan subgrade
    df['sub_grade_num'] = LabelEncoder().fit_transform(df['sub_grade'])

    # The home ownership categories 'None' and 'Other' affect the analysis later so were
    # removed as they are small in number but affect the analysis significantly (138 rows)
    df = df[(df.home_ownership != 'NONE') & (df.home_ownership != 'OTHER')]

    return df
