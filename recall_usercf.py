import argparse
import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from annoy import AnnoyIndex

from utils import Logger, evaluate

# 命令行参数
parser = argparse.ArgumentParser(description='ANN UserCF召回')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_usercf.log')
parser.add_argument('--test_size', type=int, default=1000, help='测试模式下的样本数')
parser.add_argument('--topk', type=int, default=100, help='最终召回物品的数量')
parser.add_argument('--n_trees', type=int, default=100, help='Annoy索引树的数量')
parser.add_argument('--n_sim_users', type=int, default=1000, help='召回相似用户的数量')

args = parser.parse_args()

mode = args.mode
topk = args.topk
n_trees = args.n_trees
logfile = args.logfile
test_size = args.test_size
n_sim_users = args.n_sim_users

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'ANN UserCF召回，mode: {mode}, topk: {topk}, n_trees: {n_trees}, n_sim_users: {n_sim_users}')

def build_user_ann_index(user_emb_dict, embedding_dim):
    """构建用户ANN索引"""

    log.info('构建用户ANN索引...')
    annoy_index = AnnoyIndex(embedding_dim, 'angular')

    user_id_map = {}
    reverse_user_id_map = {}

    for i, (user_id, vec) in enumerate(user_emb_dict.items()):
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm # 对用户向量进行归一化

        user_id_map[user_id] = i
        reverse_user_id_map[i] = user_id
        annoy_index.add_item(i, vec)

    annoy_index.build(n_trees)
    return annoy_index, user_id_map, reverse_user_id_map

def calculate_user_embeddings(df_click, item_emb_dict, embedding_dim):
    """根据用户历史点击物品计算用户embedding"""
    log.info('计算用户embedding...')
    user_emb_dict = {}
    user_item_history = defaultdict(list)

    for _, row in df_click.iterrows():
        user_id = row['user_id']
        item_id = row['click_article_id']
        click_time = row['click_timestamp']
        user_item_history[user_id].append((item_id, click_time))

    for user_id, items_with_time in tqdm(user_item_history.items()):
        # 对用户历史点击物品按时间倒序排序
        items_with_time = sorted(items_with_time, key=lambda x: x[1], reverse=True)

        user_vec = np.zeros(embedding_dim)
        total_weight = 0
        
        # 获取最新的点击时间
        if items_with_time:
            latest_time = items_with_time[0][1]
        else:
            continue

        for idx, (item_id, click_time) in enumerate(items_with_time):
            if item_id in item_emb_dict:
                # 位置权重：最近的物品权重更高
                loc_weight = 0.8 ** idx
                
                # 时间权重：基于与最新点击时间的时间差（小时）
                time_diff = (latest_time - click_time) / (3600 * 1000)  # 转换为小时
                time_decay = 0.7 ** time_diff
                
                # 综合位置权重和时间权重
                weight = loc_weight * time_decay
                
                user_vec += item_emb_dict[item_id] * weight
                total_weight += weight

        if total_weight > 0:
            user_emb_dict[user_id] = user_vec / total_weight # 平均池化
        else:
            # 用户没有任何有效点击，一个零向量
            user_emb_dict[user_id] = np.zeros(embedding_dim)

    return user_emb_dict, user_item_history


def recall_usercf(df_query, user_annoy_index, user_id_map, reverse_user_id_map,
                   user_item_history, n_sim_users, topk):
    """生成UserCF召回结果"""
    data_list = []

    for _, row in tqdm(df_query.iterrows(),total=len(df_query)):
        query_user_id = row['user_id']
        target_item = row['click_article_id']
        interacted_items_query_user = [x[0] for x in user_item_history.get(query_user_id, [])]

        if query_user_id not in user_id_map:
            log.warning(f"Query user {query_user_id} not found in user embeddings. Skipping.")
            continue

        # 查找相似用户
        sim_user_indices, distances = user_annoy_index.get_nns_by_item(
            user_id_map[query_user_id],
            n_sim_users + 1,
            include_distances=True
        )

        rank = defaultdict(float)

        # 遍历相似用户
        for idx, dist in zip(sim_user_indices, distances):
            sim_user_id = reverse_user_id_map[idx]
            if sim_user_id == query_user_id:
                continue

            # 用户相似度
            user_sim = 1 - dist

            # 获取相似用户的历史点击物品（包含时间信息）
            sim_user_clicked_items_with_time = user_item_history.get(sim_user_id, [])
            
            # 对相似用户的历史点击物品按时间倒序排序
            sim_user_clicked_items_with_time = sorted(sim_user_clicked_items_with_time, key=lambda x: x[1], reverse=True)
            
            # 获取最新的点击时间
            if sim_user_clicked_items_with_time:
                latest_time = sim_user_clicked_items_with_time[0][1]
            else:
                continue

            for idx, (item_id, click_time) in enumerate(sim_user_clicked_items_with_time):
                # 过滤掉查询用户已经点击过的物品
                if item_id in interacted_items_query_user:
                    continue
                
                # 位置权重：相似用户最近的点击物品权重更高
                loc_weight = 0.8 ** idx
                
                # 时间权重：基于与相似用户最新点击时间的时间差（小时）
                # 假设时间戳单位是毫秒
                time_diff = (latest_time - click_time) / (3600 * 1000)  # 转换为小时
                time_decay = 0.7 ** time_diff
                
                # 综合用户相似度、位置权重和时间权重
                weight = user_sim * loc_weight * time_decay
                
                rank[item_id] += weight

        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:topk]
        item_ids = [item[0] for item in sim_items]
        item_sim_scores = [item[1] for item in sim_items]

        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = query_user_id

        if target_item == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == target_item, 'label'] = 1

        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        data_list.append(df_temp)

    return pd.concat(data_list, sort=False)


if __name__ == '__main__':

    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

    elif mode == 'test':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        test_users = df_query['user_id'].sample(n=test_size, random_state=2024)
        df_query = df_query[df_query['user_id'].isin(test_users)]
        df_click = df_click[df_click['user_id'].isin(test_users)]

        log.info(f'测试模式：选取{test_size}个用户')
        log.info(f'df_click shape: {df_click.shape}')
        log.info(f'df_query shape: {df_query.shape}')

    else: # online mode
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    # 加载文章嵌入向量
    df_emb = pd.read_pickle('../user_data/data/articles_emb.pkl')
    # 构建 item_emb_dict
    item_emb_dict = {row['article_id']: np.array(row['embedding']) for _, row in df_emb.iterrows()}
    embedding_dim = 250 # embedding 维度是 250

    # 计算用户 embedding 和用户历史交互物品
    user_emb_dict, user_item_history = calculate_user_embeddings(df_click, item_emb_dict, embedding_dim)

    # 构建用户ANN索引
    user_annoy_index, user_id_map, reverse_user_id_map = build_user_ann_index(user_emb_dict, embedding_dim)

    # 生成召回结果
    log.info('生成UserCF召回结果...')
    df_data = recall_usercf(df_query, user_annoy_index, user_id_map,
                            reverse_user_id_map, user_item_history, n_sim_users, topk)

    # 排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True, False]).reset_index(drop=True)

    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, \
        hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'ANN UserCF: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, '
            f'{hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, '
            f'{hitrate_50}, {mrr_50}'
        )
        log.debug(f"标签分布: {df_data[df_data['label'].notnull()]['label'].value_counts()}")
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_usercf.pkl')
    elif mode == 'test':
        os.makedirs('../user_data/data/test', exist_ok=True)
        df_data.to_pickle('../user_data/data/test/recall_usercf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_usercf.pkl')

    log.info(f'ANN UserCF召回完成，召回结果形状: {df_data.shape}')