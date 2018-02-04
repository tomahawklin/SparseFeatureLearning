import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn import preprocessing
import time
from collections import Counter
import random

def display_cols(cols, df, col_dict):
    for c in cols:
        print("%30s, %10s, %10d, %100s" % (c, df[c].dtype, df[c].isnull().sum(), col_dict[c]))

def clean_df(df):
    df.drop(df[df.loan_status.str.contains('meet') | df.loan_status.str.contains('Current') | df.loan_status.str.contains('Issued')].index, inplace=True)

data_dir = '../data/'

df1 = pd.read_excel(data_dir+'2007_2011.xlsx',sheetname='Sheet1')
df2 = pd.read_excel(data_dir+'2012_2013.xlsx',sheetname='Sheet1')
df3 = pd.read_excel(data_dir+'2014.xlsx',sheetname='Sheet1')
df4 = pd.read_excel(data_dir+'2015.xlsx',sheetname='Sheet1')
df5 = pd.read_excel(data_dir+'2016_Q1.xlsx',sheetname='Sheet1')
df6 = pd.read_excel(data_dir+'2016_Q2.xlsx',sheetname='Sheet1')
df7 = pd.read_excel(data_dir+'2016_Q3.xlsx',sheetname='Sheet1')
df8 = pd.read_excel(data_dir+'2016_Q4.xlsx',sheetname='Sheet1')
df9 = pd.read_excel(data_dir+'2017_Q1.xlsx',sheetname='Sheet1')
df10 = pd.read_excel(data_dir+'2017_Q2.xlsx',sheetname='Sheet1')
df11 = pd.read_excel(data_dir+'2017_Q3.xlsx',sheetname='Sheet1')

# Only look at terminated loans (fully paid, defualt or late) and drop some datapoints that are too early
date_since_2010 = datetime.date(2010,1,1)
df1 = df1[df1.issue_d > date_since_2010]

df_list = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11]
sum([sys.getsizeof(d) for d in df_list])
sum([d.shape[0] for d in df_list])

for df in df_list:
    clean_df(df)

sum([sys.getsizeof(d) for d in df_list])
sum([d.shape[0] for d in df_list])

data = pd.concat(df_list, join = 'inner')

dict_df = pd.read_excel(data_dir + 'LCDataDictionary.xlsx',sheetname='LoanStats')
col_name = dict_df['LoanStatNew'].tolist()
col_desc = dict_df['Description'].tolist()
col_dict = dict(zip(col_name, col_desc))

# Exclude some columns that are not known beforehand
ex_col = ['last_pymnt_amnt', 'last_pymnt_d', 'mths_since_last_delinq', 'next_pymnt_d', 'pymnt_plan', 
          'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp',
          'collection_recovery_fee', 'out_prncp', 'out_prncp_inv', 'recoveries']
# Drop these columns because they contains more than 80% missing values
ex_col += ['mths_since_last_record', 'mths_since_last_major_derog']
# Drop these columns because the their values are messy
ex_col += ['title', 'emp_title']
# Drop these columns because they contains only one value or useless information 
ex_col += ['application_type', 'last_credit_pull_d', 'desc']

data = data.drop(ex_col, axis = 1)

col_dict = {k: col_dict[k] for k in col_dict if k in data.columns}

# Fill na with MISSING token for non-numeric columns
#data.desc = data.desc.fillna('MISSING')
#data.last_credit_pull_d = data.last_credit_pull_d.fillna(pd.Timestamp('20180101'))

data.loc[data['home_ownership'] == 'ANY', 'home_ownership'] = "OTHER"

# Set up labels
data['label'] = (data.loan_status.str.contains('Charged Off') | data.loan_status.str.contains('Default') | data.loan_status.str.contains('Late'))
data.label = data.label.astype(float)
col_dict['label'] = '0 for paid and in grace period, 1 for default or late or charged off'

data['issue_year'] = data['issue_d'].map(lambda x: str(x.year))
data['issue_month'] = data['issue_d'].map(lambda x: str(x.month))
data['early_year'] = data['earliest_cr_line'].map(lambda x: str(x.year))
data['early_month'] = data['earliest_cr_line'].map(lambda x: str(x.month))
data = data.drop(['issue_d', 'earliest_cr_line', 'loan_status'], axis = 1)

numeric_cols = [c for c in data.columns if data[c].dtype == 'float']
display_cols(numeric_cols, data, col_dict)
# Fill na with zero for columns that only contain numeric values
for c in numeric_cols:
    data[c] = data[c].fillna(0)

other_cols = [c for c in data.columns if c not in numeric_cols]
for c in other_cols:
    print(c, set([type(t) for t in data[c].tolist()]), len(Counter(data[c])))

# Tokenize non-numeric features
# This dictionary map feature tokens to indices 
min_count = 5
feature_dict = {}
other_cols.remove('id')
for col in other_cols:
    c = Counter(data[col])
    l = sorted(c.items(), key = lambda x: x[1], reverse = True)
    feature_dict[col] = {}
    for item in l:
        token = item[0] if item[1] >= min_count else 'UNK'
        if token not in feature_dict[col]:
            feature_dict[col][token] = len(feature_dict[col])
        else:
            continue
    # Assign id 0 to UNK tokens
    if 'UNK' in feature_dict[col]:
        data.loc[~data[col].isin(feature_dict[col]), col] = feature_dict[col]['UNK']
    data = data.replace({col: feature_dict[col]})

#data.to_csv('filter_data.csv',index=False)

data_dict = data.set_index('id').T.to_dict('dict')
random.seed(2)
indices = [k for k in data_dict]
train_data = []; test_data = []
for k in indices:
    if random.random() > 0.8:
        test_data.append(data_dict[k])
    else:
        train_data.append(data_dict[k])

np.savez('data_final', train = train_data, test = test_data, feature_dict = feature_dict)

'''
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)
'''

'''
                            id,     object,          0,                                                        A unique LC assigned ID for the loan listing.
                     loan_amnt,    float64,          0, The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
                   funded_amnt,    float64,          0,                                       The total amount committed to that loan at that point in time.
               funded_amnt_inv,    float64,          0,                         The total amount committed by investors for that loan at that point in time.
                          term,     object,          0,                 The number of payments on the loan. Values are in months and can be either 36 or 60.
                      int_rate,    float64,          0,                                                                            Interest Rate on the loan
                   installment,    float64,          0,                                     The monthly payment owed by the borrower if the loan originates.
                         grade,     object,          0,                                                                               LC assigned loan grade
                     sub_grade,     object,          0,                                                                            LC assigned loan subgrade
                     emp_title,     object,      19651,                                  The job title supplied by the Borrower when applying for the loan.*
                    emp_length,     object,          0, Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 
                home_ownership,     object,          0, The home ownership status provided by the borrower during registration or obtained from the credit report. Our values are: RENT, OWN, MORTGAGE, OTHER
                    annual_inc,    float64,          0,                        The self-reported annual income provided by the borrower during registration.
           verification_status,     object,          0,           Indicates if income was verified by LC, not verified, or if the income source was verified
                       issue_d, datetime64[ns],          0,                                                                  The month which the loan was funded
                   loan_status,     object,          0,                                                                           Current status of the loan
                    pymnt_plan,     object,          0,                                       Indicates if a payment plan has been put in place for the loan
                          desc,     object,     196952,                                                            Loan description provided by the borrower
                       purpose,     object,          0,                                           A category provided by the borrower for the loan request. 
                         title,     object,         17,                                                              The loan title provided by the borrower
                      zip_code,     object,          0,                The first 3 numbers of the zip code provided by the borrower in the loan application.
                    addr_state,     object,          0,                                           The state provided by the borrower in the loan application
                           dti,    float64,          0, A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
                   delinq_2yrs,    float64,          0, The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
              earliest_cr_line, datetime64[ns],          0,                                    The month the borrower's earliest reported credit line was opened
                fico_range_low,    float64,          0,                         The lower boundary range the borrower’s FICO at loan origination belongs to.
               fico_range_high,    float64,          0,                         The upper boundary range the borrower’s FICO at loan origination belongs to.
                inq_last_6mths,    float64,          0,                     The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
        mths_since_last_delinq,    float64,     164268,                                          The number of months since the borrower's last delinquency.
        mths_since_last_record,    float64,     258890,                                                   The number of months since the last public record.
                      open_acc,    float64,          0,                                       The number of open credit lines in the borrower's credit file.
                       pub_rec,    float64,          0,                                                                  Number of derogatory public records
                     revol_bal,    float64,          0,                                                                       Total credit revolving balance
                    revol_util,    float64,        184, Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
                     total_acc,    float64,          0,                             The total number of credit lines currently in the borrower's credit file
           initial_list_status,     object,          0,                                   The initial listing status of the loan. Possible values are – W, F
                     out_prncp,    float64,          0,                                              Remaining outstanding principal for total amount funded
                 out_prncp_inv,    float64,          0,                      Remaining outstanding principal for portion of total amount funded by investors
                   total_pymnt,    float64,          0,                                                    Payments received to date for total amount funded
               total_pymnt_inv,    float64,          0,                            Payments received to date for portion of total amount funded by investors
               total_rec_prncp,    float64,          0,                                                                           Principal received to date
                 total_rec_int,    float64,          0,                                                                            Interest received to date
            total_rec_late_fee,    float64,          0,                                                                           Late fees received to date
                    recoveries,    float64,          0,                                                                       post charge off gross recovery
       collection_recovery_fee,    float64,          0,                                                                       post charge off collection fee
                  last_pymnt_d, datetime64[ns],        262,                                                                      Last month payment was received
               last_pymnt_amnt,    float64,          0,                                                                   Last total payment amount received
                  next_pymnt_d,     object,     295338,                                                                          Next scheduled payment date
            last_credit_pull_d, datetime64[ns],         17,                                                 The most recent month LC pulled credit for this loan
          last_fico_range_high,    float64,          0,                                 The upper boundary range the borrower’s last FICO pulled belongs to.
           last_fico_range_low,    float64,          0,                                 The lower boundary range the borrower’s last FICO pulled belongs to.
    collections_12_mths_ex_med,    float64,          0,                                     Number of collections in 12 months excluding medical collections
   mths_since_last_major_derog,    float64,     238559,                                                      Months since most recent 90-day or worse rating
                   policy_code,    float64,          0,                   publicly available policy_code=1
new products not publicly available policy_code=2
              application_type,     object,          0, Indicates whether the loan is an individual application or a joint application with two co-borrowers
          pub_rec_bankruptcies,    float64,          0,                                                                 Number of public record bankruptcies
                     tax_liens,    float64,          0,                                                                                  Number of tax liens

'''