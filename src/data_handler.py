import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn import preprocessing
import time

data_dir = '../data/'

# Only look at expired data to avoid bias towards unexpired data

df1 = pd.read_excel(data_dir + '2007_2011.xlsx',sheetname='Sheet1')
df2 = pd.read_excel(data_dir + '2012_2013.xlsx',sheetname='Sheet1')
df3 = pd.read_excel(data_dir + '2014.xlsx',sheetname='Sheet1')

date_before_36 = datetime.date(2014,10,1)
date_before_60 = datetime.date(2012,10,1)
date_since_2010 = datetime.date(2010,1,1)
df1 = df1[df1.issue_d > date_since_2010]
df2 = df2[((df2.term.str.contains('60')) & (df2.issue_d < date_before_60)) | (df2.term.str.contains('36'))]
df3 = df3[(df3.term.str.contains('36')) & (df3.issue_d < date_before_36)]

data = pd.concat([df1,df2,df3],join='inner')

dict_df = pd.read_excel(data_dir + 'LCDataDictionary.xlsx',sheetname='LoanStats')
col_name = dict_df['LoanStatNew'].tolist()
col_desc = dict_df['Description'].tolist()
col_dict = dict(zip(col_name, col_desc))

# Exclude some columns that are not known beforehand
ex_col = ['last_pymnt_amnt', 'last_pymnt_d', 'mths_since_last_delinq', 'next_pymnt_d', 'pymnt_plan', 
          'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp',
          'collection_recovery_fee', 'out_prncp', 'out_prncp_inv', 'recoveries']

data = data.drop(ex_col, axis = 1)

df_col_name = data.columns.tolist()
col_dict = {k: col_dict[k] for k in col_dict if k in df_col_name}

for c in df_col_name:
  print("%30s, %10s, %10d, %100s" % (c, data[c].dtype, data[c].isnull().sum(), col_dict[c]))

data.to_csv('filter_data.csv',index=False)

data_train = data_train[['id','loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate',
         'installment','grade','sub_grade','emp_title','emp_length',
         'home_ownership','annual_inc','verification_status','issue_d',
         'loan_status','purpose','title','zip_code','addr_state','dti',
        'delinq_2yrs','earliest_cr_line','open_acc','pub_rec','last_pymnt_d',
        'last_pymnt_amnt','last_fico_range_high','last_fico_range_low','application_type',
             'revol_bal','revol_util']]

data_test = data_test[['id','loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate',
         'installment','grade','sub_grade','emp_title','emp_length',
         'home_ownership','annual_inc','verification_status','issue_d',
         'loan_status','purpose','title','zip_code','addr_state','dti',
        'delinq_2yrs','earliest_cr_line','open_acc','pub_rec','last_pymnt_d',
        'last_pymnt_amnt','last_fico_range_high','last_fico_range_low','application_type',
             'revol_bal','revol_util']]

data_train.dropna(subset=['annual_inc','loan_status','issue_d','last_pymnt_d','loan_amnt',
                          'int_rate','earliest_cr_line','open_acc','pub_rec','delinq_2yrs',
                          'grade','last_fico_range_high','last_fico_range_low','installment',
                         'funded_amnt','dti','funded_amnt_inv','revol_bal']
            ,inplace=True)

data_test.dropna(subset=['annual_inc','loan_status','issue_d','last_pymnt_d','loan_amnt',
                          'int_rate','earliest_cr_line','open_acc','pub_rec','delinq_2yrs',
                          'grade','last_fico_range_high','last_fico_range_low','installment',
                         'funded_amnt','dti','funded_amnt_inv','revol_bal']
            ,inplace=True)


# create labels for the dataset
data_train['label'] = (data_train.loan_status.str.contains('Charged Off') | 
                data_train.loan_status.str.contains('Default') | 
                data_train.loan_status.str.contains('Late'))
data_train['cr_hist'] = (data_train.issue_d - data_train.earliest_cr_line) / np.timedelta64(1, 'M')
data_train.label = data_train.label.astype(int)

data_test['label'] = (data_test.loan_status.str.contains('Charged Off') | 
                data_test.loan_status.str.contains('Default') | 
                data_test.loan_status.str.contains('Late'))
data_test['cr_hist'] = (data_test.issue_d - data_test.earliest_cr_line) / np.timedelta64(1, 'M')
data_test.label = data_test.label.astype(int)


# clean and get training/testing data 
temp = pd.get_dummies(data_train[['term','grade','emp_length','home_ownership',
                                  'verification_status','purpose']],dummy_na=True)
X_train = data_train.as_matrix(columns=['loan_amnt','funded_amnt_inv','int_rate','installment',
                                       'annual_inc','dti','delinq_2yrs','open_acc','pub_rec',
                                       'last_fico_range_high','last_fico_range_low','cr_hist'])
X_train = np.concatenate((X_train,temp.as_matrix()),axis=1)
y_train = data_train.label.as_matrix()

temp = pd.get_dummies(data_test[['term','grade','emp_length','home_ownership',
                                  'verification_status','purpose']],dummy_na=True)
X_test = data_test.as_matrix(columns=['loan_amnt','funded_amnt_inv','int_rate','installment',
                                       'annual_inc','dti','delinq_2yrs','open_acc','pub_rec',
                                       'last_fico_range_high','last_fico_range_low','cr_hist'])
X_test = np.concatenate((X_test,temp.as_matrix()),axis=1)
y_test = data_test.label.as_matrix()

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)

np.savez('data', X_train=X_train_minmax,X_test=X_test_minmax,y_train=y_train,y_test=y_test)

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