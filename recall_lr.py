import argparse
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2026)

# 命令行参数
parser = argparse.ArgumentParser(description='召回合并 - 逻辑回归')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_recall_lr.log')
parser.add_argument('--test_size', type=int, default=1000, help='测试模式下的样本数')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'召回合并 - 逻辑回归: {mode}')


def mms(df):
    user_score_max = {}
    user_score_min = {}

    # 获取用户下的相似度的最大值和最小值
    for user_id, g in df[['user_id', 'sim_score']].groupby('user_id'):
        scores = g['sim_score'].values.tolist()
        user_score_max[user_id] = scores[0]
        user_score_min[user_id] = scores[-1]

    ans = []
    for user_id, sim_score in tqdm(df[['user_id', 'sim_score']].values):
        ans.append((sim_score - user_score_min[user_id]) /
                   (user_score_max[user_id] - user_score_min[user_id]) +
                   10**-3)
    return ans


def recall_result_sim(df1_, df2_):
    df1 = df1_.copy()
    df2 = df2_.copy()

    user_item_ = df1.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict1 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    user_item_ = df2.groupby('user_id')['article_id'].agg(set).reset_index()
    user_item_dict2 = dict(zip(user_item_['user_id'],
                               user_item_['article_id']))

    cnt = 0
    hit_cnt = 0

    for user in user_item_dict1.keys():
        item_set1 = user_item_dict1[user]

        cnt += len(item_set1)

        if user in user_item_dict2:
            item_set2 = user_item_dict2[user]

            inters = item_set1 & item_set2
            hit_cnt += len(inters)

    return hit_cnt / cnt


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        recall_path = '../user_data/data/offline'
    elif mode == 'test':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        recall_path = '../user_data/data/test'
        
        # 确保测试目录存在
        os.makedirs(recall_path, exist_ok=True)
        
        log.info(f'测试模式：使用{recall_path}目录下的召回结果')
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        recall_path = '../user_data/data/online'

    log.debug(f'max_threads {max_threads}')

    # 增加多种召回方法
    recall_methods = ['itemcf', 'binetwork','w2v','usercf','hot','cold']

    # 读取并处理各个召回方法的结果
    recall_features = {}
    for recall_method in tqdm(recall_methods):
        try:
            recall_result = pd.read_pickle(
                f'{recall_path}/recall_{recall_method}.pkl')
            # 归一化相似度
            recall_result['sim_score'] = mms(recall_result)
            # 保存每个召回方法的结果
            recall_features[recall_method] = recall_result[['user_id', 'article_id', 'sim_score', 'label']]
            log.info(f'成功加载 {recall_method} 召回结果')
        except Exception as e:
            log.error(f'加载 {recall_method} 召回结果失败: {e}')

    # 检查是否有有效的召回结果
    if not recall_features:
        log.error('没有有效的召回结果，程序退出')
        exit(1)

    # 合并所有召回结果
    # 从第一个召回方法开始
    first_method = list(recall_features.keys())[0]
    merged = recall_features[first_method].copy()
    merged.rename(columns={'sim_score': f'score_{first_method}'}, inplace=True)

    # 合并其他召回方法的结果
    for method in list(recall_features.keys())[1:]:
        current = recall_features[method].copy()
        current.rename(columns={'sim_score': f'score_{method}'}, inplace=True)
        merged = merged.merge(current[['user_id', 'article_id', f'score_{method}']], 
                             on=['user_id', 'article_id'], 
                             how='outer')

    # 填充缺失值
    for method in recall_features.keys():
        score_col = f'score_{method}'
        if score_col in merged.columns:
            merged[score_col] = merged[score_col].fillna(0)
            log.info(f'{method} 缺失值已填充为0')

    # 准备特征列
    feat_cols = [f'score_{method}' for method in recall_features.keys()]
    log.info(f'使用的特征列: {feat_cols}')

    # 分离训练数据和测试数据
    train_data = merged[merged['label'].notnull()].copy()
    test_data = merged[merged['label'].isnull()].copy()

    log.info(f'训练数据形状: {train_data.shape}')
    log.info(f'测试数据形状: {test_data.shape}')

    # 训练逻辑回归模型
    if not train_data.empty:
        log.info('开始训练逻辑回归模型...')
        lr = LogisticRegression(random_state=2026)
        lr.fit(train_data[feat_cols], train_data['label'])
        log.info('逻辑回归模型训练完成')
        
        # 预测训练数据和测试数据
        train_data['pred_score'] = lr.predict_proba(train_data[feat_cols])[:, 1]
        test_data['pred_score'] = lr.predict_proba(test_data[feat_cols])[:, 1]
        
        # 合并结果
        recall_final = pd.concat([train_data, test_data], sort=False)
        log.info('逻辑回归预测完成')
    else:
        # 如果没有训练数据，使用简单的平均
        log.warning('没有训练数据，使用平均相似度')
        recall_final = merged.copy()
        recall_final['pred_score'] = recall_final[feat_cols].mean(axis=1)

    # 按用户和预测分数排序
    recall_final.sort_values(['user_id', 'pred_score'], 
                         inplace=True, 
                         ascending=[True, False])

    log.debug(f'recall_final.shape: {recall_final.shape}')
    log.debug(f'recall_final: {recall_final.head()}')

    # 删除无正样本的训练集用户
    gg = recall_final.groupby(['user_id'])
    useful_recall = []

    for user_id, g in tqdm(gg):
        if g['label'].isnull().sum() > 0:
            useful_recall.append(g)
        else:
            label_sum = g['label'].sum()
            if label_sum > 1:
                print('error', user_id)
            elif label_sum == 1:
                useful_recall.append(g)

    df_useful_recall = pd.concat(useful_recall, sort=False)
    log.debug(f'df_useful_recall: {df_useful_recall.head()}')

    df_useful_recall = df_useful_recall.sort_values(
        ['user_id', 'pred_score'], ascending=[True,
                                             False]).reset_index(drop=True)
    topk = 150
    df_useful_recall = df_useful_recall.groupby('user_id').head(topk).reset_index(drop=True)
    log.debug(f"每个用户最多保留{topk}个结果，处理后总数量: {len(df_useful_recall)}")
    
    # 重命名预测分数列为sim_score
    df_useful_recall.rename(columns={'pred_score': 'sim_score'}, inplace=True)
    
    # 删除每个召回路径单独的sim列，只保留最终合并的sim值
    # cols_to_keep = ['user_id', 'article_id', 'sim_score', 'label']
    # df_useful_recall = df_useful_recall[cols_to_keep]
    # log.info('已删除每个召回路径单独的sim列，只保留最终合并的sim值')
    
    # 计算相关指标
    if mode == 'valid':
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_useful_recall[df_useful_recall['label'].notnull()], total)

        log.debug(
            f'召回合并后指标: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    df = df_useful_recall['user_id'].value_counts().reset_index()
    df.columns = ['user_id', 'cnt']
    log.debug(f"平均每个用户召回数量：{df['cnt'].mean()}")

    log.debug(
        f"标签分布: {df_useful_recall[df_useful_recall['label'].notnull()]['label'].value_counts()}"
    )

    # 保存到本地
    if mode == 'valid':
        df_useful_recall.to_pickle('../user_data/data/offline/recall.pkl')
    elif mode == 'test':
        df_useful_recall.to_pickle('../user_data/data/test/recall.pkl')
    else:
        df_useful_recall.to_pickle('../user_data/data/online/recall.pkl')
    
    log.info('召回合并完成，结果已保存')
