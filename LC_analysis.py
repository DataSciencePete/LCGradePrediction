import argparse
import pandas as pd

import data.data_load as dl
import viz.data_viz as dv
import utils.field_lists as fl
import analysis.linear_models as lm
import analysis.random_forest as rf
import analysis.gradient_boosted_regression as gbr

#Set a threshold for a column's completeness to be useful
completeness_threshold = 0.7

parser = argparse.ArgumentParser(description='Run lending circle data analysis')
parser.add_argument('filepath',type=str)
parser.add_argument('filepath_dict',type=str)

if __name__ == '__main__':
    
    args = parser.parse_args()
    filepath = args.filepath
    filepath_dict = args.filepath_dict

    #---------------------------
    # This section loads the data, summarises basic data quality and creates features
    #---------------------------
    
    df = dl.load(filepath)
    data_dict = dl.load_dict(filepath_dict)

    #Check completeness of the raw data
    dl.print_completeness_summary(df,'Raw dataframe',completeness_threshold,data_dict)
    
    df = df[fl.get_fields_complete()]
    
    #Reevaluate completeness
    dl.print_completeness_summary(df,'Subset of more useful columns',completeness_threshold,data_dict)
    
    #Do feature engineering
    df = dl.engineer_features(df)
    
    #Check dataframe is now complete
    dl.print_completeness_summary(df,'Dataframe with features and null rows dropped',completeness_threshold,data_dict)
    
    #---------------------------
    # This section creates visualisations for EDA
    #---------------------------
    
    dv.gen_credit_corr_heatmap(df[fl.get_fields_credit()])
    dv.gen_sub_grade_plots(df[fl.get_fields_loan_subgrade()])
    dv.gen_loan_income_scatter(df)
    
    #---------------------------
    # This section runs linear, ridge and lasso regression models and plots metrics
    #---------------------------     
    
    
    y = df.sub_grade_num
    X = pd.get_dummies(df[fl.get_features_loan_subgrade()])
    
    
    #Regularisation parameter for ridge and lasso models, called alpha in sklearn
    alphas = [float(1e-6),float(1e-5),float(1e-4),0.001,0.01,0.1,0.5,1]
    
    mdls = lm.run_eval_linear_models(X,y,alphas,cv=10)
    dv.plot_linear_models(mdls,X,alphas)

    #---------------------------
    # This section runs and tunes random forest models models on the data
    #---------------------------      
        
    rf.run_RF_analysis(X,y)

    
    #---------------------------
    # This section runs and tunes gradient boosted regression models
    #---------------------------
    
    gbr.run_gbr_analysis(X,y)
    
    print('-'*20)
    print('Charts have been generated in /viz directory for this analysis')
    print('-'*20)    
    
    
    
    
    

    
