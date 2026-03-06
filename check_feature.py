import pandas as pd

# 加载offline1特征文件
df_feature = pd.read_pickle('/data3/fzy/TCxwtj/user_data/data/offline/feature.pkl')

# 打印特征列名称
print('表头:')
for col in df_feature.columns:
    print(col)

# 打印数据类型
print('\n特征列数据类型:')
print(df_feature.dtypes)

# 打印前几行数据
print('\n前5行数据:')
print(df_feature.head())

# 打印数值类型列的最大最小值
print('\n数值类型列的最大最小值:')
for col in df_feature.columns:
    if pd.api.types.is_numeric_dtype(df_feature[col]):
        max_val = df_feature[col].max()
        min_val = df_feature[col].min()
        print(f'{col}: 最小值={min_val}, 最大值={max_val}')

# 统计同一个user_id出现次数的最大值、最小值和平均值
if 'user_id' in df_feature.columns:
    print('\nuser_id出现次数统计:')
    user_counts = df_feature['user_id'].value_counts()
    print(f'最大值: {user_counts.max()}')
    print(f'最小值: {user_counts.min()}')
    print(f'平均值: {user_counts.mean():.2f}')

# 统计同一个user_id对应label为1的数量的最大值、最小值和平均值
if 'user_id' in df_feature.columns and 'label' in df_feature.columns:
    print('\nuser_id对应label=1的数量统计:')
    # 筛选label=1的记录
    label_1_df = df_feature[df_feature['label'] == 1]
    # 统计每个user_id对应的label=1的数量
    user_label_1_counts = label_1_df['user_id'].value_counts()
    # 对于没有label=1的user_id，数量为0
    all_users = df_feature['user_id'].unique()
    user_label_1_counts = user_label_1_counts.reindex(all_users, fill_value=0)
    print(f'最大值: {user_label_1_counts.max()}')
    print(f'最小值: {user_label_1_counts.min()}')
    print(f'平均值: {user_label_1_counts.mean():.2f}')
