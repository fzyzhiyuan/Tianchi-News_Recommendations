import argparse
import os
import pickle
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from utils import Logger, evaluate

random.seed(2026)

# 命令行参数
parser = argparse.ArgumentParser(description='物品冷启动召回')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_cold.log')
parser.add_argument('--test_size', type=int, default=1000, help='测试模式下的样本数')
parser.add_argument('--recall_num', type=int, default=50, help='每个用户召回的物品数量')
parser.add_argument('--hist_len', type=float, default=50, help='用户历史记录长度')



args = parser.parse_args()

mode = args.mode
logfile = args.logfile
test_size = args.test_size
recall_num = args.recall_num
hist_len = args.hist_len

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'物品冷启动召回，mode: {mode}')


def get_user_hist_info(df_click, item_type_dict, item_words_dict, item_created_time_dict):
    """获取用户历史行为信息"""
    user_hist_item_types = defaultdict(set)
    user_hist_item_words = defaultdict(list)
    user_last_item_created_time = defaultdict(int)
    user_interacted_items = defaultdict(set)
    user_hist_items_ordered = defaultdict(list)  # 按时间顺序存储用户历史物品
    user_hist_items_time = defaultdict(list)  # 存储用户历史物品的点击时间
    
    # 按用户和时间排序
    df_click_sorted = df_click.sort_values(['user_id', 'click_timestamp'])
    
    for _, row in tqdm(df_click_sorted.iterrows()):
        user_id = row['user_id']
        item_id = row['click_article_id']
        click_time = row['click_timestamp']/1000
        user_interacted_items[user_id].add(item_id)
        # 限制用户历史记录长度
        if len(user_hist_items_ordered[user_id]) >= hist_len:
            user_hist_items_ordered[user_id].pop(0)  # 删除最早的记录
            user_hist_items_time[user_id].pop(0)  # 同时删除对应的时间记录
        user_hist_items_ordered[user_id].append(item_id)  # 按时间顺序添加
        user_hist_items_time[user_id].append(click_time)  # 保存点击时间
        
        # 从物品信息中获取相关属性
        if item_id in item_type_dict:
            user_hist_item_types[user_id].add(item_type_dict[item_id])
        if item_id in item_words_dict:
            user_hist_item_words[user_id].append(item_words_dict[item_id])
        if item_id in item_created_time_dict:
            item_time = item_created_time_dict[item_id]
            if item_time > user_last_item_created_time[user_id]:
                user_last_item_created_time[user_id] = item_time
    
    # 计算每个用户的平均字数
    user_hist_mean_words = {}
    for user_id, words_list in user_hist_item_words.items():
        if words_list:
            user_hist_mean_words[user_id] = np.mean(words_list)
        else:
            user_hist_mean_words[user_id] = 0
    
    return user_hist_item_types, user_hist_mean_words, user_last_item_created_time, user_interacted_items, user_hist_items_ordered, user_hist_items_time


def get_item_info():
    """获取物品信息"""
    item_type_dict = {}
    item_words_dict = {}
    item_created_time_dict = {}
    
    # 从 articles.csv 文件中读取物品信息
    articles_df = pd.read_csv('../data/articles.csv')
    
    for _, row in tqdm(articles_df.iterrows()):
        item_id = row['article_id']
        item_type_dict[item_id] = row['category_id']
        item_words_dict[item_id] = row['words_count']
        # 转换为秒为单位
        item_created_time_dict[item_id] = row['created_at_ts'] / 1000
    
    return item_type_dict, item_words_dict, item_created_time_dict


def recall_cold_start(df_query, df_click, item_sim):
    """物品冷启动召回"""
    # 先获取物品信息
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info()
    
    # 获取用户历史信息
    user_hist_item_types, user_hist_mean_words, user_last_item_created_time, user_interacted_items, user_hist_items_ordered, user_hist_items_time = get_user_hist_info(df_click, item_type_dict, item_words_dict, item_created_time_dict)
    
    # 识别冷启动物品（点击次数较少的物品）
    item_click_count = df_click['click_article_id'].value_counts()
    cold_items = set(item_click_count[item_click_count <= 3].index)
    log.info(f'冷启动物品数量: {len(cold_items)}')
    
    data_list = []
    
    for user_id, item_id in tqdm(df_query.values):
        rank = {}
        
        # 获取用户历史信息
        hist_item_types = user_hist_item_types.get(user_id, set())
        hist_mean_words = user_hist_mean_words.get(user_id, 0)
        last_created_time = user_last_item_created_time.get(user_id, 0)
        interacted_items = user_interacted_items.get(user_id, set())
        hist_items_ordered = user_hist_items_ordered.get(user_id, [])
        hist_items_time = user_hist_items_time.get(user_id, [])
        last_click_time = hist_items_time[-1] if hist_items_time else 0
        
        # 对冷启动物品进行筛选和排序
        for cold_item in cold_items:
            if cold_item in interacted_items:
                continue
                
            # 获取物品信息
            curr_item_type = item_type_dict.get(cold_item, None)
            curr_item_words = item_words_dict.get(cold_item, 0)
            curr_item_created_time = item_created_time_dict.get(cold_item, 0)
            
            # 规则筛选
            if not curr_item_type or curr_item_type not in hist_item_types:
                continue
                
            if abs(curr_item_words - hist_mean_words) > 200:
                continue
                
            if last_created_time > 0:
                time_diff = abs(curr_item_created_time - last_created_time)
                # 时间差超过的过滤掉
                if time_diff >  24 * 3600:
                    continue
            
            # 计算相似度（如果有预计算的相似度）
            sim_score = 0
            if cold_item in item_sim and hist_items_ordered and len(hist_items_time) == len(hist_items_ordered):
                # 使用位置衰减和时间衰减计算加权相似度
                total_sim = 0
                total_weight = 0
                
                # 遍历用户历史物品，按时间顺序（最近的物品权重更高）
                for i, (hist_item, hist_time) in enumerate(reversed(list(zip(hist_items_ordered, hist_items_time)))):
                    if hist_item in item_sim and cold_item in item_sim[hist_item]:
                        # 位置衰减因子：越近的物品权重越高
                        # 使用指数衰减，衰减系数为0.8
                        position_weight = 0.8 ** i
                        
                        # 时间衰减因子：时间越近的物品权重越高
                        time_diff = abs(last_click_time - hist_time)
                        # 使用指数衰减
                        time_weight = 0.7 ** (time_diff / 3600)
                        
                        # 综合权重
                        weight = position_weight * time_weight
                        total_sim += item_sim[hist_item][cold_item] * weight
                        total_weight += weight
                
                if total_weight > 0:
                    sim_score = total_sim / total_weight
            
            # 如果没有相似度信息，使用默认分数
            if sim_score == 0:
                sim_score = 0.2  # 默认相似度
                
            rank[cold_item] = sim_score
        
        # 排序并限制召回数量
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:recall_num]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]
        
        # 构建结果数据框
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id
        
        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1
        
        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')
        
        data_list.append(df_temp)
    
    return pd.concat(data_list, sort=False)


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/itemcf_sim.pkl'
    elif mode == 'test':
        # 测试模式：读取部分数据
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        
        # 随机选择一部分用户
        test_users = df_query['user_id'].sample(n=test_size, random_state=2024)
        df_query = df_query[df_query['user_id'].isin(test_users)]
        df_click = df_click[df_click['user_id'].isin(test_users)]
        
        os.makedirs('../user_data/sim/test', exist_ok=True)
        sim_pkl_file = '../user_data/sim/test/itemcf_sim.pkl'
        
        log.info(f'测试模式：选取{test_size}个用户')
        log.info(f'df_click shape: {df_click.shape}')
        log.info(f'df_query shape: {df_query.shape}')
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')
        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/itemcf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    # 尝试加载已有的相似度文件
    try:
        with open(sim_pkl_file, 'rb') as f:
            item_sim = pickle.load(f)
        log.info('加载已有的相似度文件')
    except:
        # 如果没有相似度文件，创建一个空字典
        item_sim = {}
        log.info('未找到相似度文件，使用空字典')

    # 执行冷启动召回
    df_data = recall_cold_start(df_query, df_click, item_sim)

    # 对结果进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True, False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)
        log.debug(
            f'cold_start: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
        log.debug(f"标签分布: {df_data[df_data['label'].notnull()]['label'].value_counts()}")

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_cold.pkl')
    elif mode == 'test':
        os.makedirs('../user_data/data/test', exist_ok=True)
        df_data.to_pickle('../user_data/data/test/recall_cold.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_cold.pkl')
