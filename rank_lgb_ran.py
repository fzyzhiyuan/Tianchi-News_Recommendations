import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import gc
import random
import warnings
import numpy as np

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from utils import Logger, evaluate, gen_sub, gen_detailed_result

warnings.filterwarnings('ignore')

seed = 2026
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='lightgbm 排序模型')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test_lgbran.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'lightgbm 排序模型，mode: {mode}')


def train_model(df_feature):
    # 定义特征列，使用实际的特征列名称
    # lgb_cols = ['sim_score', 'category_id', 'created_at_ts', 'words_count',
    #             'user_id_click_article_created_at_ts_diff_mean', 'user_id_click_diff_mean',
    #             'user_click_timestamp_created_at_ts_diff_mean', 'user_click_timestamp_created_at_ts_diff_std',
    #             'user_click_datetime_hour_std', 'user_clicked_article_words_count_mean',
    #             'user_click_last_article_words_count', 'user_click_last_article_created_time',
    #             'user_clicked_article_created_time_max', 'user_click_last_article_click_time',
    #             'user_clicked_article_click_time_mean', 'user_last_click_created_at_ts_diff',
    #             'user_last_click_timestamp_diff', 'user_last_click_words_count_diff',
    #             'user_id_cnt', 'article_id_cnt', 'user_id_category_id_cnt',
    #             'user_clicked_article_itemcf_sim_sum', 'user_last_click_article_itemcf_sim',
    #             'user_last_click_article_binetwork_sim', 'user_last_click_article_w2v_sim',
    #             'user_click_article_w2w_sim_sum_2']

    df_train = df_feature[df_feature['label'].notnull()]
    df_test = df_feature[df_feature['label'].isnull()]

    del df_feature
    gc.collect()

    ycol = 'label'
    # feature_names = lgb_cols
    feature_names = list(
        filter(
            lambda x: x not in [ycol, 'created_at_datetime', 'click_datetime'],
            df_train.columns))
    feature_names.sort()

    # 五折交叉验证
    k_fold = 5
    kfold = GroupKFold(n_splits=k_fold)
    trn_df = df_train

    score_list = []
    score_df = trn_df[['user_id', 'article_id','label']]
    sub_preds = np.zeros(df_test.shape[0])
    df_importance_list = []
    
    # 五折交叉验证
    for n_fold, (trn_idx, val_idx) in enumerate(
            kfold.split(trn_df[feature_names], trn_df[ycol],
                        trn_df['user_id'])):
        train_idx = trn_df.iloc[trn_idx]
        valid_idx = trn_df.iloc[val_idx]
        
        # 训练集与验证集的用户分组
        train_idx.sort_values(by=['user_id'], inplace=True)
        g_train = train_idx.groupby(['user_id'], as_index=False).count()[ycol].values
        
        valid_idx.sort_values(by=['user_id'], inplace=True)
        g_val = valid_idx.groupby(['user_id'], as_index=False).count()[ycol].values
        
        log.debug(
            f'\nFold_{n_fold + 1} Training ================================\n'
        )
        
        # 创建回调函数列表
        callbacks = [
            lgb.log_evaluation(period=100),  # 每100轮打印一次日志
            lgb.early_stopping(stopping_rounds=500)  # 早停机制
        ]
        
        # 定义模型，与cla保持一致的参数
        lgb_ranker = lgb.LGBMRanker(
            num_leaves=63,
            max_depth=-1,
            learning_rate=0.05,
            n_estimators=10000,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            random_state=seed,
            n_jobs=16,
            min_child_weight=50,
            importance_type='gain',
            device='gpu',
            gpu_use_dp=True,
            boost_from_average=True,
            label_gain=[0,1],
            objective='lambdarank'
        )
        
        # 训练模型
        lgb_ranker.fit(
            train_idx[feature_names], 
            train_idx[ycol], 
            group=g_train,
            eval_set=[(valid_idx[feature_names], valid_idx[ycol])], 
            eval_group=[g_val], 
            eval_at=[5], 
            eval_metric=['ndcg'], 
            callbacks=callbacks
        )
        
        # 预测验证集结果
        log.info(f'lightgbm 排序模型，第{n_fold+1}折验证集')
        valid_idx['pred'] = lgb_ranker.predict(
            valid_idx[feature_names], 
            num_iteration=lgb_ranker.best_iteration_
        )
        
        # 对输出结果进行排序
        valid_idx.sort_values(by=['user_id', 'pred'], ascending=[True, False], inplace=True)
        valid_idx['pred_rank'] = valid_idx.groupby(['user_id'])['pred'].rank(ascending=False, method='first')
        
        # 将验证集的预测结果放到一个列表中，后面进行拼接
        score_list.append(valid_idx[['user_id', 'article_id', 'pred', 'pred_rank']])
        
        # 计算每次交叉验证的结果相加，最后求平均
        log.info(f'lightgbm 排序模型，第{n_fold+1}折测试集')
        sub_preds += lgb_ranker.predict(
            df_test[feature_names], 
            num_iteration=lgb_ranker.best_iteration_
        )
        
        df_importance = pd.DataFrame({
            'feature_name':
            feature_names,
            'importance':
            lgb_ranker.feature_importances_,
        })
        df_importance_list.append(df_importance)
        
        joblib.dump(lgb_ranker, f'../user_data/model/lgb_ranker{n_fold}.pkl')
    
    # 特征重要性
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby([
        'feature_name'
    ])['importance'].agg('mean').sort_values(ascending=False).reset_index()
    log.debug(f'importance: {df_importance}')
    
    score_df_ = pd.concat(score_list, axis=0)
    score_df = score_df.merge(score_df_, how='left', on=['user_id', 'article_id'])
    
    # 测试集的预测结果，多次交叉验证求平均
    df_test['pred'] = sub_preds / k_fold
    df_test.sort_values(by=['user_id', 'pred'], ascending=[True, False], inplace=True)
    df_test['pred_rank'] = df_test.groupby(['user_id'])['pred'].rank(ascending=False, method='first')

    # 计算相关指标
    total = score_df['user_id'].nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        score_df, total)
    log.debug(
        f'{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )

    # 生成提交文件
    os.makedirs('../prediction_result', exist_ok=True)
    
    # 保存训练集和验证集的预测结果
    gen_detailed_result(score_df, '../prediction_result/detailed_lgbran_train_val.csv')
    log.info("训练集和验证集预测结果已保存")
    
    # 保存测试集的预测结果
    prediction = df_test[['user_id', 'article_id', 'pred']]
    gen_detailed_result(prediction, '../prediction_result/detailed_lgbran_test.csv')
    log.info("测试集预测结果已保存")
    
    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    df_sub.to_csv(f'../prediction_result/result_lgbran.csv', index=False)
    log.info("预测完成，结果已保存")


def online_predict(df_test):
    # 使用与训练时相同的特征列
    lgb_cols = ['sim_score', 'category_id', 'created_at_ts', 'words_count',
                'user_id_click_article_created_at_ts_diff_mean', 'user_id_click_diff_mean',
                'user_click_timestamp_created_at_ts_diff_mean', 'user_click_timestamp_created_at_ts_diff_std',
                'user_click_datetime_hour_std', 'user_clicked_article_words_count_mean',
                'user_click_last_article_words_count', 'user_click_last_article_created_time',
                'user_clicked_article_created_time_max', 'user_click_last_article_click_time',
                'user_clicked_article_click_time_mean', 'user_last_click_created_at_ts_diff',
                'user_last_click_timestamp_diff', 'user_last_click_words_count_diff',
                'user_id_cnt', 'article_id_cnt', 'user_id_category_id_cnt',
                'user_clicked_article_itemcf_sim_sum', 'user_last_click_article_itemcf_sim',
                'user_last_click_article_binetwork_sim', 'user_last_click_article_w2v_sim',
                'user_click_article_w2w_sim_sum_2']
    
    ycol = 'label'
    feature_names = lgb_cols

    prediction = df_test[['user_id', 'article_id']]
    prediction['pred'] = 0

    for fold_id in tqdm(range(5)):
        model = joblib.load(f'../user_data/model/lgb_ranker{fold_id}.pkl')
        pred_test = model.predict(df_test[feature_names])
        prediction['pred'] += pred_test / 5

    # 生成提交文件
    gen_detailed_result(prediction, '../prediction_result/detailed_lgbran.csv')
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('prediction_result', exist_ok=True)
    df_sub.to_csv(f'prediction_result/result_lgbran.csv', index=False)


if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/feature.pkl')

        for f in df_feature.select_dtypes('object').columns:
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))

        train_model(df_feature)
    else:
        df_feature = pd.read_pickle('../user_data/data/test/feature.pkl')
        online_predict(df_feature)
