import pandas as pd

# 读取test_01.csv和test_12.csv文件
df_12 = pd.read_csv('invail_mergerd.csv')
df_01 = pd.read_csv('test_0.csv')


# 将df_01和df_12合并
merged_df = pd.concat([df_01,df_12],axis=0)



# 将结果保存到新的csv文件中
merged_df.to_csv('test_13.csv', index=False)