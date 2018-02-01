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