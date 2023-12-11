import flask
import argparse
import pandas as pd
import numpy as  np
import pickle
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
import io
import boto3
from sklearn.metrics import roc_curve
#define a flask app
app = flask.Flask(__name__)

from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

import os
aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")



def calculate_fin_ratios(df):
    #liquidity ratios
    df['current_ratio']=df['asst_current']/df['current_liab'].apply(lambda x :x+1 if x!=-1 else x+0)
    df['cash_ratio']=df['cash_and_equiv']/df['debt_st'].apply(lambda x :x+1 if x!=-1 else x+0)
    df["CFO_ratio"]=df["cf_operations"]/df["current_liab"].apply(lambda x :x+1 if x!=-1 else x+0)

    #debt coverage ratios
    df["CFO_to_total_liab"]=df["cf_operations"]/df["total_liab"].apply(lambda x :x+1 if x!=-1 else x+0)
    df['leverage']=df['total_liab']/df['asst_tot'].apply(lambda x :x+1 if x!=-1 else x+0)
    df['dscr']=df['ebitda']/df['current_liab'].apply(lambda x :x+1 if x!=-1 else x+0)
    df['interest_coverage_ratio']=df['ebitda']/df['exp_financing'].apply(lambda x :x+1 if x!=-1 else x+0)
    df['asset_cov_ratio']=(df['asst_tang_fixed']+df['asst_fixed_fin']-(df['current_liab']-df['debt_st']))/df['total_liab'].apply(lambda x :x+1 if x!=-1 else x+0)

    #profitability ratios
    df["gross_profit_margin"]=df["prof_operations"]/(df["rev_operating"].apply(lambda x :x+1 if x!=-1 else x+0))
    df['fixed_assets']=df['asst_intang_fixed']+df['asst_tang_fixed']+df['asst_fixed_fin']
    df['roa']=df['net_income']/df['asst_tot']
    df['roe'] = df['roe'].fillna(df['net_income'] / df['eqty_tot'])

    return df

def apply_calibration_bins(new_scores):
    bin_limits = []
    calibrated_factors = []

    with open('calibration_data.txt', "r") as file:
        for line in file:
            # Split the line into bin limit and calibrated factor
            limit_str, factor_str = line.split(':')
            # Convert limit string to tuple of floats and factor string to float
            lower_bound, upper_bound = map(float, limit_str.strip('()').split(', '))
            calibrated_factor = float(factor_str.strip())
            # Append to lists
            bin_limits.append((lower_bound, upper_bound))
            calibrated_factors.append(calibrated_factor)


    # Initialize calibrated_scores array with NaN
    calibrated_scores = np.full_like(new_scores, np.nan, dtype=np.float64)

    # Loop through the bin_limits and calibrated_factors to apply calibration
    for i, (lower_bound, upper_bound) in enumerate(bin_limits):
        # For the first bin, include scores less than the lower_bound
        if i == 0:
            bin_mask = (new_scores >= lower_bound) & (new_scores < upper_bound)
            lower_edge_mask = (new_scores < lower_bound)
            calibrated_scores[bin_mask] = calibrated_factors[i]
            calibrated_scores[lower_edge_mask] = calibrated_factors[i]
        # For the last bin, include scores greater than the upper_bound
        elif i == len(bin_limits) - 1:
            bin_mask = (new_scores >= lower_bound) & (new_scores <= upper_bound)
            upper_edge_mask = (new_scores > upper_bound)
            calibrated_scores[bin_mask] = calibrated_factors[i]
            calibrated_scores[upper_edge_mask] = calibrated_factors[i]
        else:
            bin_mask = (new_scores >= lower_bound) & (new_scores < upper_bound)
            calibrated_scores[bin_mask] = calibrated_factors[i]

    return calibrated_scores



def preprocessor(df, preproc_params = {}, new = True):
    """
    This function preprocesses the data for the model
    """
    #copy the dataframe
    new_df=df.copy()

    new_df["rev_operating"]=new_df["rev_operating"].fillna(new_df["COGS"]+ new_df["prof_operations"])

    #operation on df using roa and asst_tot variable
    new_df['net_income']=new_df['roa']*new_df['asst_tot']
    new_df['roe'] = new_df['roe'].fillna(new_df['net_income'] / new_df['eqty_tot'])
    
    new_df['current_liab']=new_df['asst_current']-new_df['wc_net']
    new_df['non_current_assets']=new_df['asst_tot']-new_df['asst_current']
    new_df["total_liab"]=new_df["asst_tot"]-new_df["eqty_tot"]
    new_df['non_current_liab']=new_df['total_liab']-new_df['current_liab']

    new_df.drop(["HQ_city","fs_year",'eqty_corp_family_tot'],axis=1,inplace=True)

    new_df['fixed_assets']=new_df['asst_intang_fixed']+new_df['asst_tang_fixed']+new_df['asst_fixed_fin']

    
    new_df['debt_lt'] = new_df['debt_lt'].fillna(new_df['total_liab']-new_df['current_liab'])
    new_df['margin_fin']=new_df['margin_fin'].fillna(new_df['eqty_tot']-new_df['fixed_assets'])

    new_df['lt_liab']=new_df['total_liab']-new_df['current_liab']

    new_df=calculate_fin_ratios(new_df)

    new_df=new_df[['stmt_date','legal_struct','asst_current', 'AR','cash_and_equiv', 'asst_tot', 'non_current_assets','eqty_tot',
              'debt_st', 'lt_liab', 'current_liab', 'total_liab','non_current_liab',
              'rev_operating', 'COGS', 'prof_operations', 
              'goodwill', 'taxes', 'profit','exp_financing',
              'ebitda', 'wc_net', 'margin_fin', 'cf_operations','net_income',
              'current_ratio','cash_ratio', 'CFO_ratio', 
              'CFO_to_total_liab','leverage', 'dscr', 'interest_coverage_ratio', 'asset_cov_ratio',
              'gross_profit_margin','fixed_assets','roa','roe',
              ]]
    
    

   
    new_df['legal_struct'] = new_df['legal_struct'].map({'SAU':1,'SPA':2,'SAA':3,'SRS':4,'SRL':5,'SRU':6})
    
    target_columns=['legal_struct','current_ratio','CFO_ratio', 'leverage', 'roa','fixed_assets']
    #transform the variables
    for i in target_columns:
        if i=="legal_struct":
            continue
        if i=="fixed_assets":
            continue
        temp=abs(new_df[i])**(1/3)
        #assign sign to the variable
        temp=temp*new_df[i]/abs(new_df[i])
        new_df[i]=temp

    return new_df



def predict(df):
    # Process the data
    features=['legal_struct', 'CFO_ratio', 'current_ratio','roa', 'leverage', 'fixed_assets']

    processed_data = preprocessor(df, new = False)
    X=processed_data[features]

    predictions=model.predict_proba(X)[:,-1]
    predictions=pd.Series(predictions)
    predictions=predictions.apply(lambda x: x if 0<=x<=1 else 0)

    calibrated_scores = apply_calibration_bins(predictions)
    X["scores"]=calibrated_scores

    return X

@app.route('/')
def index():
    return render_template('index.html')
from datetime import datetime

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    This function returns the prediction for the given data
    """
    #get the csv file from the request
    file = flask.request.files['file']
    #read the csv file
    df = pd.read_csv(file,index_col=0)
    #get the predictions
    predictions = predict(df)
    #place on IO buffer and senf to s3 bucket
    # Create an in-memory buffer
    buffer = io.StringIO()
    # Write the predictions to the buffer
    predictions.to_csv(buffer,index=False)  
    # Create an S3 client
    # Upload to S3
    # s3_client = boto3.client('s3')
    bucket_name = 'finance3'
    #change the object name based on current date and time
    object_name = f'predictions_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.csv'
    region_name = 'us-east-1'
    s3 = boto3.client('s3')
    # Write the buffer to an S3 object
    try:
        s3.put_object(Body=buffer.getvalue(), Bucket=bucket_name, Key=object_name)
        file_url = f'https://{bucket_name}.s3.{region_name}.amazonaws.com/{object_name}'
        return flask.jsonify({'message': 'Successfully uploaded predictions to S3', 'file_url': file_url})
    except Exception as e:
        return flask.jsonify({'error': str(e)})


def calculate_cost(default_rate, non_default_rate,fpr,tpr, cost_benefit_matrix):
    fp=fpr
    tn=1-fpr
    fn=1-tpr
    tp=tpr

    # Calculate the total cost using the cost-benefit matrix

    # here we assume that avg transction amount is $1 
    total_cost = (non_default_rate*fp * cost_benefit_matrix[0, 1] +
                  default_rate*fn* cost_benefit_matrix[1, 0] +
                  default_rate* tp * cost_benefit_matrix[0, 0] +
                  non_default_rate* tn * cost_benefit_matrix[1, 1])
    return total_cost 

def find_fixed_cutoff(y_true, y_pred_prob, cost_benefit_matrix,default_rate,non_default_rate,target_cost=0):
    # Initialize lists to store the costs and cutoffs
    costs = []
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    cut_offs = []
    # Calculate the cost for each threshold
    
    for i in range(0, len(thresholds)):
        costs.append(calculate_cost(default_rate, non_default_rate,fpr[i],tpr[i], cost_benefit_matrix))
        cut_offs.append(fpr[i])

    # Find the cutoff where the cost is closest to the target cost
    optimal_idx = np.argmin(np.abs(np.array(costs) - target_cost))
    return fpr[optimal_idx],tpr[optimal_idx],thresholds[optimal_idx]


import json

@app.route('/cut_off', methods=['POST'])
def cut_off():
    """
    This function returns the cut-off point for the model
    """
    #get the cost of false positive from the request
    cost_fp = flask.request.form['cost_fp']
    #get the cost of false negative from the request
    cost_fn = flask.request.form['cost_fn']
    #get the cost of true positive from the request
    cost_tp = flask.request.form['cost_tp']
    #get the cost of true negative from the request
    cost_tn = flask.request.form['cost_tn']

    #create the cost benefit matrix
    cost_benefit_matrix = np.array([
                                    [-float(cost_tp),float(cost_fp)],  # TP, FP
                                    [float(cost_fn), -float(cost_tn)]    # FN, TN
                                ])


    #get the cost from the request
    cost = flask.request.form['cost']
    #calculate the cut-off point
    fpr,tpr,cut_off=cut_off_analysis(cost_benefit_matrix,float(cost))
    #send float values to json
    return json.dumps({'fpr': float(fpr),'tpr':float(tpr)})



def cut_off_analysis(cost_benefit_matrix,target_cost):
    """
    This function calculates the cut-off point for the model
    """
    df=pd.read_csv("test.csv",index_col=0)
    default_rate = df['target'].mean()
    non_default_rate = 1 - default_rate
    y_true=df['target']
    features=['legal_struct', 'CFO_ratio', 'current_ratio','roa', 'leverage', 'fixed_assets']

    processed_data = preprocessor(df, new = False)
    X=processed_data[features]

    y_pred_prob=model.predict_proba(X)[:,-1]
    cut_off_fpr,cut_off_tpr,cut_off=find_fixed_cutoff(y_true, y_pred_prob, cost_benefit_matrix,default_rate,non_default_rate,target_cost=target_cost)
    return cut_off_fpr,cut_off_tpr,cut_off







        
    

# Run the server
if __name__ == '__main__':
    model=xgb.XGBClassifier()
    model.load_model("turquoise_model.bin")
    app.run(port = 8000, debug=True)


    








