import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('raw_data.csv')
data_A = data[data['grade'] == 'A']
data_B = data[data['grade'] == 'B']
data_C = data[data['grade'] == 'C']
data_D = data[data['grade'] == 'D']
data_E = data[data['grade'] == 'E']
data_F = data[data['grade'] == 'F']
data_G = data[data['grade'] == 'G']

def df2data(data):
    data = data[['id','loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate',
         'installment','sub_grade','emp_title','emp_length',
         'home_ownership','annual_inc','verification_status','issue_d',
         'loan_status','purpose','title','zip_code','addr_state','dti',
        'delinq_2yrs','earliest_cr_line','open_acc','pub_rec','last_pymnt_d',
        'last_pymnt_amnt','last_fico_range_high','last_fico_range_low','application_type',
             'revol_bal','revol_util']].copy()
    
    data.dropna(subset=['annual_inc','loan_status','issue_d','last_pymnt_d','loan_amnt',
                          'int_rate','earliest_cr_line','open_acc','pub_rec','delinq_2yrs',
                          'last_fico_range_high','last_fico_range_low','installment',
                         'funded_amnt','dti','funded_amnt_inv','revol_bal']
            ,inplace=True)
    data['label'] = (data.loan_status.str.contains('Charged Off') | 
                data.loan_status.str.contains('Default') | 
                data.loan_status.str.contains('Late'))
    data['cr_hist'] = (data.issue_d - data.earliest_cr_line) / np.timedelta64(1, 'M')
    data.label = data.label.astype(int)
    # clean and get training/testing data 
    temp = pd.get_dummies(data[['term','emp_length','home_ownership',
                                  'verification_status','purpose']],dummy_na=True)
    X = data.as_matrix(columns=['loan_amnt','funded_amnt_inv','int_rate','installment',
                                       'annual_inc','dti','delinq_2yrs','open_acc','pub_rec',
                                       'last_fico_range_high','last_fico_range_low','cr_hist'])
    X = np.concatenate((X,temp.as_matrix()),axis=1)
    y = data.label.as_matrix()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    return X_train, X_test, y_train, y_test

def fit_rf(X_train, X_test, y_train, y_test):
	target_names = ['Paid', 'Defaulted']
	rf_en = RandomForestClassifier(max_depth=10, criterion = 'entropy')
	rf_en.fit(X_train, y_train)
	y_pred = rf_en.predict(X_test)
	print('accuracy: ', accuracy_score(y_test, y_pred))
	print(classification_report(y_test, y_pred, target_names = target_names, digits=4))
	pos_prob = rf_en.predict_proba(X_test)[:, 1]
	print('AUC: ', roc_auc_score(y_test, pos_prob))


fit_rf(*df2data(data_A))

'''
AUC:
Grade A: 0.944
Grade B: 0.934
Grade C: 0.931
Grade D: 0.922
Grade E: 0.925
Grade F: 0.915
Grade G: 0.913
Overall: 0.938
Augment: 0.944
'''
