import pandas as pd

dict_data = {'Treatment': ['C', 'C', 'C'], 'Biorep': ['A', 'A', 'A'], 'Techrep': [1, 1, 1], 'AAseq': ['ELVISLIVES1', 'ELVISLIVES2', 'ELVISLIVES3'], 'mz':[500.0, 500.5, 501.0]}
df_a = pd.DataFrame(dict_data)
print(df_a)
#dict_data = {'Treatment1': ['C', 'C', 'C'], 'Biorep1': ['A', 'A', 'A'], 'Techrep1': [1, 1, 1], 'AAseq1': ['ELVISLIVES2', 'ELVISLIVES3', 'ELVISLIVES1'], 'inte1':[1100.0, 1050.0, 1010.0]}
dict_data = {'AAseq': ['ELVISLIVES2', 'ELVISLIVES3', 'ELVISLIVES1'], 'inte1':[1100.0, 1050.0, 1010.0]}
df_b = pd.DataFrame(dict_data)
print(df_b)
#print(pd.concat([df_a,df_b], axis=1))
#right=df_a
#left=df_b
print(pd.merge(df_a,df_b, how='outer' ,on=['AAseq']))
df=pd.merge(df_a,df_b, how='outer' ,on=['AAseq'])
df=df.set_index('AAseq')
print(df)
