import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import numpy as np
#from gensim.models import FastText as fText

train_csv='../input/train.csv'
test_csv='../input/test.csv'
#fasttest_train='fasttest_train.csv'
#fasttext_test='fasttext_test.csv'

# load or create your dataset
print('Load data...')
df_train = pd.read_csv(train_csv)
df_train = df_train[['item_id','user_id','deal_probability']]
df_fasttext_train = pd.read_csv('fasttest_train.csv',sep=' ',names = range(300),index_col = False) 
df_train = pd.concat([df_train, df_fasttext_train], axis=1)

'''
df_test = pd.read_csv(test_csv)
df_test = df_test[['item_id','user_id']]
df_fasttext_test = pd.read_csv('fasttext_test.csv',sep=' ',names = range(301),index_col = False) 
df_test = pd.concat([df_test, df_fasttext_test], axis=1)
'''

print(df_train.head(2))
#print(df_test.head(2))
#print(df_fasttext_train.head(1))
#print(df_fasttext_test.head(1))

y = df_train.deal_probability
X = df_train.drop('deal_probability', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values 
y_train = y_train.values 

X_valid = X_valid.values 
y_valid = y_valid.values 

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

lgbm_params =  {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        # 'max_depth': 15,
        'num_leaves': 270,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.75,
        # 'bagging_freq': 5,
        'learning_rate': 0.018,
        'verbose': 0
        }


print('Start training...')
# train
gbm = lgb.train(lgbm_params,
        lgb_train,
        num_boost_round=20,
        valid_sets=lgb_eval,
        early_stopping_rounds=5)

print('Save model...')
# save model to file
gbm.save_model('model.txt')

print('Start predicting...')
# predict
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_valid, y_pred) ** 0.5)
