# coding=utf-8
import pandas as pd
import nltk
from nltk.corpus import stopwords
import fastText
import numpy as np
import json
import math
import csv

train_csv='../input/train.csv'
#test_csv='../input/test.csv'

df_train = pd.read_csv(train_csv)
#df_test = pd.read_csv(test_csv)

#preprocessing
stopwords = stopwords.words('russian')
#model = fastText.load_model('../input/ru.300.bin')

data=dict()
for i in range(len(df_train)):
    #print(df_train.description[i])
    #print(math.isnan(df_train.description[i]))
    #print(type(df_train.description[i]))
    item_id = df_train.item_id[i]
    try:
        content = df_train.description[i].split()
    except:
        #print(df_train.description[i])
        data[item_id]=" "     
        continue
        
    #print(content)
    """
    try:
        content = [word for word in content if word not in ['/','/n','/t','/r','\n','\t','\r']]
    
        //content.remove('/')
        //content.remove('/n')
        //content.remove('/t')
        //content.remove('/r')
        //content.remove('\n')
        //content.remove('\t')
        //content.remove('\r')
        //content.remove(' ')
    except:
        print("no /")    
    """
    #print(content)
    #print("================")
    content = ' '.join(content)
    data[item_id]=content    
    #outfile.write(content)

w = csv.writer(open("../input/train_filter_ui.csv", "w"))
#w = csv.writer(open("../input/test_filter.csv", "w"))
for key, val in data.items():
    w.writerow([key, val])
