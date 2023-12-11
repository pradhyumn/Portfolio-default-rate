import argparse
import pandas as pd
import numpy as  np
import pickle
import xgboost as xgb

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

def main():
    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Process input and output CSV files.")
    parser.add_argument("--input_csv", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path")
    
    args = parser.parse_args()

    # Process the data
    model=xgb.XGBClassifier()
    model.load_model("turquoise_model.bin")
    features=['legal_struct', 'CFO_ratio', 'current_ratio','roa', 'leverage', 'fixed_assets']

    processed_data = preprocessor(pd.read_csv(args.input_csv,index_col=0), new = False)
    X=processed_data[features]

    predictions=model.predict_proba(X)[:,-1]
    predictions=pd.Series(predictions)
    predictions=predictions.apply(lambda x: x if 0<=x<=1 else 0)

    calibrated_scores = apply_calibration_bins(predictions)
    calibrated_scores = pd.Series(calibrated_scores)

    # Write the processed data to the output CSV file
    calibrated_scores.to_csv(args.output_csv, index=False,header=False)

if __name__ == "__main__":
    main()
