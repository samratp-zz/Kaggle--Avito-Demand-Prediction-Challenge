import pandas as pd

train_csv='../input/train.csv'
test_csv='../input/test.csv'

df_train = pd.read_csv(train_csv)
#print(df_train.shape)
#print(len(df_train.item_id.unique()))
#print(len(df_train.user_id.unique()))
#df_train = df_train['','','']
#df_train.to_csv(header=None,index=False)
df_Domain = df_train[['item_id','user_id','deal_probability']]
df_Domain.to_csv('proNet-core/data/train_pro.csv',sep=' ',header=None,index=False)

df_test = pd.read_csv(test_csv)
df_target = df_test[['item_id','user_id']]
df_target.to_csv('proNet-core/data/test_pro.csv',sep=' ',header=None,index=False)

