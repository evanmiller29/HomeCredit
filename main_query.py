import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
from math import ceil
import numpy as np
from os import chdir
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import stacknet_funcs as funcs

from os.path import join

folder = "F:/Nerdy Stuff/Kaggle/Home credit/data"
stacknet_folder = "C:/Users/Evan/PycharmProjects/StackNet/"

missing_thres = 0.85

print('Importing data...')
data = pd.read_csv(join(folder, 'application_train.csv'))
test = pd.read_csv(join(folder, 'application_test.csv'))
prev = pd.read_csv(join(folder, 'previous_application.csv'))
buro = pd.read_csv(join(folder, 'bureau.csv'))
buro_balance = pd.read_csv(join(folder, 'bureau_balance.csv'))
credit_card  = pd.read_csv(join(folder, 'credit_card_balance.csv'))
POS_CASH  = pd.read_csv(join(folder, 'POS_CASH_balance.csv'))
payments = pd.read_csv(join(folder, 'installments_payments.csv'))
lgbm_submission = pd.read_csv(join(folder, 'sample_submission.csv'))

#Separate target variable
y = data['TARGET']
del data['TARGET']

#Feature engineering
data['loan_to_income'] = data.AMT_ANNUITY/data.AMT_INCOME_TOTAL
test['loan_to_income'] = test.AMT_ANNUITY/test.AMT_INCOME_TOTAL

#One-hot encoding of categorical features in data and test sets
categorical_features = [col for col in data.columns if data[col].dtype == 'object']

one_hot_df = pd.concat([data,test])
one_hot_df = pd.get_dummies(one_hot_df, columns=categorical_features)

data = one_hot_df.iloc[:data.shape[0],:]
test = one_hot_df.iloc[data.shape[0]:,]

#Pre-processing buro_balance
print('Pre-processing buro_balance...')

buro_grouped_size = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
buro_grouped_max = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
buro_grouped_min = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()

buro_counts = buro_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)
buro_counts_unstacked = buro_counts.unstack('STATUS')
buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X',]
buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size
buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min
buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max

buro = buro.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')

# Bringing through features from: https://www.kaggle.com/shanth84/home-credit-bureau-data-feature-engineering

buro['BUREAU_LOAN_COUNT'] = buro.groupby(['SK_ID_CURR'])['SK_ID_CURR'].transform("count")
buro['BUREAU_LOAN_TYPES'] = buro[['SK_ID_CURR', 'CREDIT_TYPE']].groupby('SK_ID_CURR')['CREDIT_TYPE'].transform("nunique")
buro['AVERAGE_LOAN_TYPE'] = buro['BUREAU_LOAN_COUNT']/buro['BUREAU_LOAN_TYPES']
buro['CREDIT_ACTIVE_BINARY'] = buro['CREDIT_ACTIVE'].apply(lambda x: funcs.open_or_closed(x))
buro['ACTIVE_LOANS_PERCENTAGE_TEST'] = buro.groupby('SK_ID_CURR')['CREDIT_ACTIVE_BINARY'].transform("mean")

# Calculating the difference between credit periods

# Groupby each Customer and Sort values of DAYS_CREDIT in ascending order
grp = buro[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by = ['SK_ID_CURR'])
grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending = False)).reset_index(drop = True)#rename(index = str, columns = {'DAYS_CREDIT': 'DAYS_CREDIT_DIFF'})
print("Grouping and Sorting done")

# Calculate Difference between the number of Days
grp1['DAYS_CREDIT1'] = grp1['DAYS_CREDIT']*-1
grp1['DAYS_DIFF'] = grp1.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
grp1['DAYS_DIFF'] = grp1['DAYS_DIFF'].fillna(0).astype('uint32')
del grp1['DAYS_CREDIT1'], grp1['DAYS_CREDIT'], grp1['SK_ID_CURR']
gc.collect()
print("Difference days calculated")

buro = buro.merge(grp1, on = ['SK_ID_BUREAU'], how = 'left')
print("Difference in Dates between Previous CB applications is CALCULATED ")
print(buro.shape)

print("% of loans per customer where end date for credit is past")

buro['CREDIT_ENDDATE_BINARY'] = buro.apply(lambda x: funcs.past_end_date(x.DAYS_CREDIT_ENDDATE), axis = 1)
print("New Binary Column calculated")

grp = buro.groupby(by = ['SK_ID_CURR'])['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(index=str, columns={'CREDIT_ENDDATE_BINARY': 'CREDIT_ENDDATE_PERCENTAGE'})
buro = buro.merge(grp, on = ['SK_ID_CURR'], how = 'left')

del buro['CREDIT_ENDDATE_BINARY']

print("Average number of days in which credit expires in future")

buro['CREDIT_ENDDATE_BINARY'] = buro['DAYS_CREDIT_ENDDATE']

buro['CREDIT_ENDDATE_BINARY'] = buro.apply(lambda x: funcs.past_end_date(x.DAYS_CREDIT_ENDDATE), axis = 1)
print("New Binary Column calculated")

# We take only positive values of  ENDDATE since we are looking at Bureau Credit VALID IN FUTURE
# as of the date of the customer's loan application with Home Credit
buro_future = buro[buro['CREDIT_ENDDATE_BINARY'] == 1]
buro_future.shape

print("Calculate Difference in successive future end dates of CREDIT")

print("Create Dummy Column for CREDIT_ENDDATE")

buro_future['DAYS_CREDIT_ENDDATE1'] = buro_future['DAYS_CREDIT_ENDDATE']
# Groupby Each Customer ID
grp = buro_future[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT_ENDDATE1']].groupby(by = ['SK_ID_CURR'])
# Sort the values of CREDIT_ENDDATE for each customer ID
grp1 = grp.apply(lambda x: x.sort_values(['DAYS_CREDIT_ENDDATE1'], ascending = True)).reset_index(drop = True)
del grp
gc.collect()
print("Grouping and Sorting done")

print("Calculate the Difference in ENDDATES and fill missing values with zero")
grp1['DAYS_ENDDATE_DIFF'] = grp1.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT_ENDDATE1'].diff()
grp1['DAYS_ENDDATE_DIFF'] = grp1['DAYS_ENDDATE_DIFF'].fillna(0).astype('uint32')
del grp1['DAYS_CREDIT_ENDDATE1'], grp1['SK_ID_CURR']
gc.collect()
print("Difference days calculated")

print("Merge new feature 'DAYS_ENDDATE_DIFF' with original Data frame for BUREAU DATA")

buro = buro.merge(grp1, on = ['SK_ID_BUREAU'], how = 'left')
del grp1
gc.collect()

print("Calculate Average of DAYS_ENDDATE_DIFF")

grp = buro[['SK_ID_CURR', 'DAYS_ENDDATE_DIFF']].groupby(by = ['SK_ID_CURR'])['DAYS_ENDDATE_DIFF'].mean().reset_index().rename( index = str, columns = {'DAYS_ENDDATE_DIFF': 'AVG_ENDDATE_FUTURE'})
B = buro.merge(grp, on = ['SK_ID_CURR'], how = 'left')
del grp
#del B['DAYS_ENDDATE_DIFF']
del buro['CREDIT_ENDDATE_BINARY'], buro['DAYS_CREDIT_ENDDATE']
print(buro.shape)

print("Debt over credit ratio")

buro['AMT_CREDIT_SUM_DEBT'] = buro['AMT_CREDIT_SUM_DEBT'].fillna(0)
buro['AMT_CREDIT_SUM'] = buro['AMT_CREDIT_SUM'].fillna(0)

grp1 = buro[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
grp2 = buro[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})

buro = buro.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
buro = buro.merge(grp2, on = ['SK_ID_CURR'], how = 'left')
buro['DEBT_CREDIT_RATIO'] = buro['TOTAL_CUSTOMER_DEBT']/buro['TOTAL_CUSTOMER_CREDIT']

del buro['TOTAL_CUSTOMER_DEBT'], buro['TOTAL_CUSTOMER_CREDIT']
gc.collect()
print(buro.shape)

print("Overdue debt ratio")

buro['AMT_CREDIT_SUM_DEBT'] = buro['AMT_CREDIT_SUM_DEBT'].fillna(0)
buro['AMT_CREDIT_SUM_OVERDUE'] = buro['AMT_CREDIT_SUM_OVERDUE'].fillna(0)

grp1 = buro[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
grp2 = buro[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename( index = str, columns = { 'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})

buro = buro.merge(grp1, on = ['SK_ID_CURR'], how = 'left')
buro = buro.merge(grp2, on = ['SK_ID_CURR'], how = 'left')
del grp1, grp2
gc.collect()

buro['OVERDUE_DEBT_RATIO'] = buro['TOTAL_CUSTOMER_OVERDUE']/buro['TOTAL_CUSTOMER_DEBT']

del buro['TOTAL_CUSTOMER_OVERDUE'], buro['TOTAL_CUSTOMER_DEBT']
gc.collect()
print(buro.shape)

print("Average number of loans prolonged")

buro['CNT_CREDIT_PROLONG'] = buro['CNT_CREDIT_PROLONG'].fillna(0)
grp = buro[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']].groupby(by = ['SK_ID_CURR'])['CNT_CREDIT_PROLONG'].mean().reset_index().rename( index = str, columns = { 'CNT_CREDIT_PROLONG': 'AVG_CREDITDAYS_PROLONGED'})
buro = buro.merge(grp, on = ['SK_ID_CURR'], how = 'left')
print(buro.shape)

#Pre-processing previous_application
print('Pre-processing previous_application...')
#One-hot encoding of categorical features in previous application data set
prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']
prev = pd.get_dummies(prev, columns=prev_cat_features)
avg_prev = prev.groupby('SK_ID_CURR').mean()
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
del avg_prev['SK_ID_PREV']

#Pre-processing buro
print('Pre-processing buro...')
#One-hot encoding of categorical features in buro data set
buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']
buro = pd.get_dummies(buro, columns=buro_cat_features)
avg_buro = buro.groupby('SK_ID_CURR').mean()
avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
del avg_buro['SK_ID_BUREAU']

#Pre-processing POS_CASH
print('Pre-processing POS_CASH...')
le = LabelEncoder()
POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
POS_CASH['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#Pre-processing credit_card
print('Pre-processing credit_card...')
credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

#Pre-processing payments
print('Pre-processing payments...')
avg_payments = payments.groupby('SK_ID_CURR').mean()
avg_payments2 = payments.groupby('SK_ID_CURR').max()
avg_payments3 = payments.groupby('SK_ID_CURR').min()
del avg_payments['SK_ID_PREV']

#Join data bases
print('Joining databases...')
data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')

data = data.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')

#Remove features with many missing values
print('Removing features with more than 85% missing...')

test = test[test.columns[data.isnull().mean() < missing_thres]]
data = data[data.columns[data.isnull().mean() < missing_thres]]

#Delete customer Id
del data['SK_ID_CURR']
del test['SK_ID_CURR']

file = stacknet_folder + "train.txt"
funcs.from_sparse_to_file(file, data, deli1=" ", deli2=":", ytarget=y)

file = stacknet_folder + "test.txt"
funcs.from_sparse_to_file(file, test, deli1=" ", deli2=":", ytarget=None)