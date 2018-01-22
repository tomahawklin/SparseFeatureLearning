import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, classification_report
from sklearn import preprocessing
import time

data_dir = '../data/'

df1 = pd.read_excel(data_dir + '2007_2011.xlsx',sheetname='Sheet1')
df2 = pd.read_excel(data_dir + '2012_2013.xlsx',sheetname='Sheet1')
df3 = pd.read_excel(data_dir + '2014.xlsx',sheetname='Sheet1')
df4 = pd.read_excel(data_dir + '2015.xlsx',sheetname='Sheet1')
df5 = pd.read_excel(data_dir + '2016_Q1.xlsx',sheetname='Sheet1')
df6 = pd.read_excel(data_dir + '2016_Q2.xlsx',sheetname='Sheet1')
df7 = pd.read_excel(data_dir + '2016_Q3.xlsx',sheetname='Sheet1')

date_since_2009 = datetime.date(2010, 1, 1)
df1 = df1[df1.issue_d > date_since_2009]
data = pd.concat([df1,df2,df3,df4,df5,df6,df7],join='inner')
data = data[['id','loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate',
         'installment','grade','sub_grade','emp_title','emp_length',
         'home_ownership','annual_inc','verification_status','issue_d',
         'loan_status','purpose','title','zip_code','addr_state','dti',
        'delinq_2yrs','earliest_cr_line','open_acc','pub_rec','last_pymnt_d',
        'last_pymnt_amnt','last_fico_range_high','last_fico_range_low','application_type',
             'revol_bal','revol_util']]
data.dropna(subset=['annual_inc','loan_status','issue_d','last_pymnt_d','loan_amnt',
                          'int_rate','earliest_cr_line','open_acc','pub_rec','delinq_2yrs',
                          'grade','last_fico_range_high','last_fico_range_low','installment',
                         'funded_amnt','dti','funded_amnt_inv','revol_bal']
            ,inplace=True)

data['label'] = (data.loan_status.str.contains('Charged Off') | 
                data.loan_status.str.contains('Default') | 
                data.loan_status.str.contains('Late'))
data['cr_hist'] = (data.issue_d - data.earliest_cr_line) / np.timedelta64(1, 'M')
data.label = data.label.astype(int)


# clean and get training/testing data 
temp = pd.get_dummies(data[['term','grade','emp_length','home_ownership',
                                  'verification_status','purpose']],dummy_na=True)
X = data.as_matrix(columns=['loan_amnt','funded_amnt_inv','int_rate','installment',
                                       'annual_inc','dti','delinq_2yrs','open_acc','pub_rec',
                                       'last_fico_range_high','last_fico_range_low','cr_hist'])
X = np.concatenate((X,temp.as_matrix()),axis=1)
y = data.label.as_matrix()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)
X_test_minmax = min_max_scaler.transform(X_test)

np.savez('data', X_train=X_train_minmax,X_test=X_test_minmax,y_train=y_train,y_test=y_test)