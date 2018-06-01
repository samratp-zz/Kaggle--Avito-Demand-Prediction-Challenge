import pandas as pd

train_csv='../input/train.csv'
test_csv='../input/test.csv'

df_train = pd.read_csv(train_csv)
df_train = df_train[['item_id','user_id','deal_probability']]

df_test = pd.read_csv(test_csv)
df_test = df_test[['item_id','user_id']]

print(df_train.head(2))
print(df_test.head(2))
