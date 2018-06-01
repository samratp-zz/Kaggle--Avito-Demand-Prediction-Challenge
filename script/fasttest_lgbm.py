import pandas as pd
import lightgbm as lgb


train_csv='../input/train.csv'
test_csv='../input/test.csv'
fasttest_train='fasttest_train.csv'
fasttext_test='fasttext_test.csv'

#df_train = pd.read_csv(train_csv)
#df_train = df_train[['item_id','user_id','deal_probability']]
df_fasttext_train = pd.read_csv(fasttext_train,sep=' ',names = range(300)) 

#df_test = pd.read_csv(test_csv)
#df_test = df_test[['item_id','user_id']]
df_fasttext_test = pd.read_csv(fasttext_test,sep=' ',names = range(300)) 

#print(df_train.head(2))
#print(df_test.head(2))
print(df_fasttext_train.head(1))
print(df_fasttext_test.head(1))
