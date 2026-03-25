import argparse
import os
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import Logger, evaluate

random.seed(2026)
warnings.filterwarnings('ignore')

# 命令行参数
parser = argparse.ArgumentParser(description='改进的基于热度的文章召回')
parser.add_argument('--mode', default='valid', choices=['valid', 'online', 'test'])
parser.add_argument('--logfile', default='test_hot.log')
parser.add_argument('--test_size', type=int, default=1000, help='测试模式下的样本数')
parser.add_argument('--top_k', type=int, default=100, help='召回物品数量')
parser.add_argument('--time_window', type=int, default=300, help='时间窗口大小（秒）')
parser.add_argument('--n_jobs', type=int, default=1, help='并行处理的线程数')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile
test_size = args.test_size
top_k = args.top_k
time_window = args.time_window
n_jobs = args.n_jobs

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'热度召回，mode: {mode}, top_k: {top_k}, time_window: {time_window}, n_jobs: {n_jobs}')


class HotRecall:
    def __init__(self, df_click, df_article, time_window, top_k):
        self.df_click = df_click.sort_values('click_timestamp')
        self.df_article = df_article
        self.time_window = time_window
        self.top_k = top_k
        
        # 预构建映射字典
        log.info('预构建文章分类映射字典')
        self.article_category_map = dict(zip(df_article['article_id'], df_article['category_id']))
        
        # 预构建用户历史点击记录
        log.info('预构建用户历史点击记录')
        self.user_history_map = {}
        for user_id, group in df_click.groupby('user_id'):
            self.user_history_map[user_id] = set(group['click_article_id'])
        
        # 预构建用户最后点击时间
        log.info('预构建用户最后点击时间')
        self.user_last_click_map = {}
        for user_id, group in df_click.groupby('user_id'):
            self.user_last_click_map[user_id] = group['click_timestamp'].max()
        
        # 预构建用户分类偏好
        log.info('预构建用户分类偏好')
        self.user_preferences_map = {}
        for user_id, group in df_click.groupby('user_id'):
            self.user_preferences_map[user_id] = self._calculate_category_preferences(group)
    
    def _calculate_category_preferences(self, user_clicks):
        if len(user_clicks) == 0:
            return {}
        
        # 分别计算长期和短期兴趣
        latest_time = user_clicks['click_timestamp'].max()
        time_diff = latest_time - user_clicks['click_timestamp']
        
        # 短期兴趣（最近24小时）
        short_term_mask = time_diff <= 24*3600
        short_term_clicks = user_clicks[short_term_mask]
        
        # 长期兴趣（所有历史）
        category_counts = user_clicks['category_id'].value_counts()
        total_clicks = category_counts.sum()
        long_term_prefs = (category_counts / total_clicks).to_dict()
        
        if len(short_term_clicks) > 0:
            # 计算短期兴趣
            short_term_counts = short_term_clicks['category_id'].value_counts()
            short_term_prefs = (short_term_counts / short_term_counts.sum()).to_dict()
            
            # 融合长期和短期兴趣（短期占比更大）
            final_prefs = {}
            all_categories = set(long_term_prefs.keys()) | set(short_term_prefs.keys())
            for cat in all_categories:
                short_term = short_term_prefs.get(cat, 0)
                long_term = long_term_prefs.get(cat, 0)
                final_prefs[cat] = 0.7 * short_term + 0.3 * long_term
        else:
            final_prefs = long_term_prefs
        
        return final_prefs
    
    def get_user_last_click(self, user_id, exclude_timestamp=None):
        if user_id not in self.user_last_click_map:
            return None
        
        if exclude_timestamp is None:
            return self.user_last_click_map[user_id]
        
        # 如果排除的时间就是最后一次点击时间，需要重新计算
        if abs(self.user_last_click_map[user_id] - exclude_timestamp) < 1e-6:
            user_clicks = self.df_click[self.df_click['user_id'] == user_id]
            user_clicks = user_clicks[user_clicks['click_timestamp'] != exclude_timestamp]
            if len(user_clicks) == 0:
                return None
            return user_clicks['click_timestamp'].max()
        
        return self.user_last_click_map[user_id]
    
    def get_time_window_articles(self, timestamp):
        window_start = timestamp - self.time_window
        
        # 使用向量化操作筛选时间窗口内的点击
        window_clicks = self.df_click[
            (self.df_click['click_timestamp'] >= window_start) &
            (self.df_click['click_timestamp'] < timestamp)
        ]
        
        if len(window_clicks) == 0:
            return {}
        
        # 时间衰减权重计算
        time_diff = timestamp - window_clicks['click_timestamp']
        time_weights = 1 / (1 + np.log1p(time_diff / (self.time_window/4)))
        
        # 使用groupby向量化计算文章得分
        article_groups = window_clicks.groupby('click_article_id')
        
        # 计算基础得分（点击量）
        base_scores = article_groups.size()
        
        # 计算时间得分（时间衰减平均值）
        time_scores = article_groups.apply(lambda x: time_weights[x.index].mean())
        
        # 计算用户多样性得分
        diversity_scores = article_groups['user_id'].nunique() / base_scores
        
        # 综合评分
        article_scores = base_scores * time_scores * (1 + np.log1p(diversity_scores))
        
        return article_scores.to_dict()
    
    def adjust_scores_by_category(self, article_scores, user_preferences):
        adjusted_scores = {}
        for article_id, base_score in article_scores.items():
            category_id = self.article_category_map.get(article_id, -1)
            category_weight = user_preferences.get(category_id, 0.1)
            
            # 使用非线性提升
            boost = np.sqrt(1 + category_weight)
            adjusted_scores[article_id] = base_score * boost
        
        return adjusted_scores
    
    def process_user(self, row):
        user_id = row['user_id']
        target_item = row['click_article_id']
        
        last_click_time = self.get_user_last_click(user_id)
        if last_click_time is None:
            return None
            
        # 获取时间窗口内的热门文章
        article_scores = self.get_time_window_articles(last_click_time)
        if not article_scores:
            return None
            
        # 获取用户已读文章
        user_history = self.user_history_map.get(user_id, set())
        
        # 过滤掉已读文章
        article_scores = {k: v for k, v in article_scores.items() if k not in user_history}
        if not article_scores:
            return None
        
        # 获取用户分类偏好
        user_preferences = self.user_preferences_map.get(user_id, {})
        
        # 根据用户分类偏好调整得分
        adjusted_scores = self.adjust_scores_by_category(article_scores, user_preferences)
        
        # 排序并选择top_k
        sorted_articles = sorted(adjusted_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        if not sorted_articles:
            return None
        
        item_ids = [item[0] for item in sorted_articles]
        item_sim_scores = [item[1] for item in sorted_articles]
        
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id
        
        if target_item == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0
            df_temp.loc[df_temp['article_id'] == target_item, 'label'] = 1
        
        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')
        
        return df_temp


def recall_hot_articles(df_query, df_click, df_article, time_window=7200, top_k=50, n_jobs=4):
    # 初始化HotRecall对象
    log.info("信息初始化")
    hot_recall = HotRecall(df_click, df_article, time_window, top_k)
    
    data_list = []
    
    # 使用并行处理
    log.info(f'并行处理线程数: {n_jobs}')
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for _, row in df_query.iterrows():
                futures.append(executor.submit(hot_recall.process_user, row))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="热度召回"):
                result = future.result()
                if result is not None:
                    data_list.append(result)
    else:
        # 单线程处理
        for _, row in tqdm(df_query.iterrows(), total=len(df_query), desc="热度召回"):
            result = hot_recall.process_user(row)
            if result is not None:
                data_list.append(result)
    
    if not data_list:
        return pd.DataFrame(columns=['user_id', 'article_id', 'sim_score', 'label'])
    
    return pd.concat(data_list, sort=False)


def sample_test_data(df_click, df_query, test_size):
    # 1. 确保用户有足够的历史点击数据
    user_click_counts = df_click.groupby('user_id').size()
    qualified_users = user_click_counts[user_click_counts >= 5].index
    
    # 2. 获取有效的查询用户（确保有target item）
    valid_query_users = df_query[df_query['click_article_id'] != -1]['user_id'].unique()
    
    # 3. 找到同时满足条件的用户
    valid_users = set(qualified_users) & set(valid_query_users)
    
    # 4. 采样用户
    if len(valid_users) < test_size:
        log.warning(f"符合条件的用户数({len(valid_users)})小于请求的测试规模({test_size})")
        sampled_users = list(valid_users)
    else:
        sampled_users = np.random.choice(list(valid_users), size=test_size, replace=False)
    
    # 5. 对每个用户保留完整的时间序列数据
    df_click_sampled = df_click[df_click['user_id'].isin(sampled_users)]
    df_query_sampled = df_query[df_query['user_id'].isin(sampled_users)]
    
    # 6. 确保每个用户都有查询数据
    final_users = set(df_query_sampled['user_id'].unique())
    df_click_sampled = df_click_sampled[df_click_sampled['user_id'].isin(final_users)]
    
    return df_click_sampled, df_query_sampled


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
    elif mode == 'test':
        # 测试模式：读取部分数据
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
        
        log.info(f'采样测试数据...')
        df_click, df_query = sample_test_data(df_click, df_query, test_size)
        
        log.info(f'测试集统计:')
        log.info(f'用户数: {df_click["user_id"].nunique()}')
        log.info(f'点击数: {len(df_click)}')
        log.info(f'查询数: {len(df_query)}')
        log.info(f'平均每用户点击数: {len(df_click) / df_click["user_id"].nunique():.2f}')
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')
    df_click['click_timestamp'] = df_click['click_timestamp']/1000
    # 读取文章信息
    df_article = pd.read_csv('../data/articles.csv')
    
    # 数据预处理
    log.info('预处理数据...')
    df_click = df_click.merge(
        df_article[['article_id', 'category_id']], 
        left_on='click_article_id',
        right_on='article_id',
        how='left'
    )
    
    # 执行召回
    log.info('执行文章召回...')
    df_data = recall_hot_articles(
        df_query, 
        df_click, 
        df_article,
        time_window=time_window,
        top_k=top_k,
        n_jobs=n_jobs
    )
    
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
            f'改进的热度召回: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
        log.debug(
        f"标签分布: {df_data[df_data['label'].notnull()]['label'].value_counts()}"
        )

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_hot.pkl')
    elif mode == 'test':
        os.makedirs('../user_data/data/test', exist_ok=True)
        df_data.to_pickle('../user_data/data/test/recall_hot.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_hot.pkl')
    log.info(f'改进的热度召回完成，召回结果形状: {df_data.shape}')
