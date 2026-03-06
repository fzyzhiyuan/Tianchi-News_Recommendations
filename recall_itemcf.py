import argparse
import math
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import Logger, evaluate

random.seed(2026)

parser = argparse.ArgumentParser(description='ItemCF增强召回')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_itemcf.log')
parser.add_argument('--test_size', type=int, default=1000, help='测试模式下的样本数')
parser.add_argument('--time_decay', type=float, default=0.8, help='时间衰减系数')
parser.add_argument('--position_weight', type=float, default=0.7, help='位置权重衰减')
parser.add_argument('--top_k_sim', type=int, default=200, help='相似物品数量')
parser.add_argument('--top_k_recall', type=int, default=100, help='召回物品数量')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
test_size = args.test_size
time_decay = args.time_decay
position_weight = args.position_weight
top_k_sim = args.top_k_sim
top_k_recall = args.top_k_recall

os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'ItemCF增强召回，mode: {mode}')


def cal_sim(df):
    user_item_ = df.groupby('user_id').agg({
        'click_article_id': list,
        'click_timestamp': list
    }).reset_index()
    
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))
    user_time_dict = dict(zip(user_item_['user_id'], user_item_['click_timestamp']))
    
    item_cnt = defaultdict(int)
    item_time_dict = defaultdict(list)
    sim_dict = {}
    
    for user_id, items in tqdm(user_item_dict.items(), desc="计算相似度"):
        times = user_time_dict[user_id]
        max_time = max(times)
        
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            item_time_dict[item].append(times[loc1])
            sim_dict.setdefault(item, {})
            
            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                
                sim_dict[item].setdefault(relate_item, 0)
                
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (position_weight ** np.abs(loc2 - loc1))
                
                time_diff = abs(times[loc2] - times[loc1])
                time_weight = time_decay ** (time_diff / (3600 * 1000))
                
                sim_dict[item][relate_item] += loc_weight * time_weight / math.log(1 + len(items))
    
    for item, relate_items in tqdm(sim_dict.items(), desc="归一化相似度"):
        for relate_item, cij in relate_items.items():
            sim_dict[item][relate_item] = cij / math.sqrt(item_cnt[item] * item_cnt[relate_item])
    
    return sim_dict, user_item_dict, user_time_dict


def recall(df_query, item_sim, user_item_dict, user_time_dict, top_k_sim=200, top_k_recall=100):
    data_list = []
    
    for user_id, item_id in tqdm(df_query.values, desc="召回"):
        rank = {}
        
        if user_id not in user_item_dict:
            continue
        
        interacted_items = user_item_dict[user_id]
        interacted_times = user_time_dict[user_id]
        
        max_time = max(interacted_times)
        
        recent_items = list(zip(interacted_items[::-1], interacted_times[::-1]))[:10]
        
        for loc, (item, click_time) in enumerate(recent_items):
            if item not in item_sim:
                continue
            
            time_diff = max_time - click_time
            time_weight = time_decay ** (time_diff / (3600 * 1000))
            
            for relate_item, wij in sorted(item_sim[item].items(), key=lambda d: d[1], reverse=True)[:top_k_sim]:
                if relate_item not in interacted_items:
                    rank.setdefault(relate_item, 0)
                    rank[relate_item] += wij * (position_weight ** loc) * time_weight
        
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:top_k_recall]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]
        
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
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        
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
    
    item_sim, user_item_dict, user_time_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()
    
    df_data = recall(df_query, item_sim, user_item_dict, user_time_dict, top_k_sim, top_k_recall)
    
    df_data = df_data.sort_values(['user_id', 'sim_score'], ascending=[True, False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')
    
    if mode in ['valid', 'test']:
        log.info(f'计算召回指标')
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total
        )
        log.debug(
            f'ItemCF增强: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
        log.debug(f"标签分布: {df_data[df_data['label'].notnull()]['label'].value_counts()}")
    
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_itemcf.pkl')
    elif mode == 'test':
        os.makedirs('../user_data/data/test', exist_ok=True)
        df_data.to_pickle('../user_data/data/test/recall_itemcf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_itemcf.pkl')
