import pandas as pd
import multiprocessing
from itertools import product

train_csv_ui='../input/train_filter_ui.csv'
test_csv_ui='../input/test_filter_ui.csv'
train_csv_fasttext='fasttext_train.csv'
test_csv_fasttext='fasttext_test.csv'

print("Loading data")
print("Training data")
df_train = pd.read_csv(train_csv_ui,names=['item_id','description'])
df_tf_train = pd.read_csv(train_csv_fasttext,sep=' ',names=list(range(301)))
df_tf_train = df_tf_train.drop(columns=[300])
df_train = pd.concat([df_train,df_tf_train],axis=1)
del df_tf_train

print("Testing data")
df_test = pd.read_csv(test_csv_ui,names=['item_id','description'])
df_tf_test = pd.read_csv(test_csv_fasttext,sep=' ',names=list(range(301)))
df_tf_test = df_tf_test.drop(columns=[300])
df_test = pd.concat([df_test,df_tf_test],axis=1)
del df_tf_test


def process_frame_train(col):
    #print(df_train.ix[col,'item_id'])
    item = df_train.loc[col].item_id
    df_train.loc[col] = pd.Series([])
    df_train.ix[col,'item_id']=item
    #print(df_train.ix[col,'item_id'])
    #print("========================")

def process_frame_test(col):
    item = df_test.loc[col].item_id
    df_test.loc[col] = pd.Series([])
    df_test.ix[col,'item_id']=item

if __name__ == '__main__':
    
    print("Process training data")
    df_train_remove = df_train[df_train['description'].isnull()]
    cols = df_train_remove.index.tolist()
    pool = multiprocessing.Pool()
    pool.map(process_frame_train,product(df_train,cols))
    #print(df_train)
    #exit()
    df_train.to_csv("fasttext_feature_train_muti.csv",index=False)
    del df_train

    print("=====================")

    print("Process testing data")
    df_test_remove = df_test[df_test['description'].isnull()]
    cols = df_test_remove.index.tolist()
    pool = multiprocessing.Pool()
    pool.map(process_frame_test,product(df_test,cols))
    df_test.to_csv("fasttext_feature_test_muti.csv",index=False)
    del df_test

    '''
    ##########################
    df_train_remove = df_train[df_train['description'].isnull()]
    cols = df_train_remove.index.tolist()
    print("=====")
    for col in cols:
        item = df_train.loc[col].item_id
        df_train.loc[col] = pd.Series([])
        df_train.ix[col,'item_id']=item
    df_train.to_csv("fasttext_feature_train.csv",sep=',')
    '''

    '''
    ##########################
    df_test_remove = df_test[df_test['description'].isnull()]
    cols = df_test_remove.index.tolist()
    print("=====")
    for col in cols:
        item = df_test.loc[col].item_id
        df_test.loc[col] = pd.Series([])
        df_test.ix[col,'item_id']=item
    df_test.to_csv("fasttext_feature_test.csv",sep=',')
    '''
