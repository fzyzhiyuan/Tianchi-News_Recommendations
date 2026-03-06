import os
import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from utils import Logger, gen_sub, evaluate

# 设置随机种子，保证每次运行结果一致
seed = 2026
random.seed(seed)
np.random.seed(seed)

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger('../user_data/log/test_merge.log').logger
log.info('逻辑回归融合开始')


def main():
    """
    使用逻辑回归融合多个模型的结果
    """
    # 定义模型列表，格式: [(模型名称, 模型标识符), ...]
    models = [('DIN', 'din'),
              ('LGB分类', 'lgbcls'),
              ('LGB排序', 'lgbran')]
    
    # 基础路径和文件前缀
    base_path = '../prediction_result'
    file_prefix = 'detailed_'
    
    # 加载查询数据，用于评估
    import pandas as pd
    df_query = pd.read_pickle('../user_data/data/offline/query.pkl')
    
    # 读取各个模型的训练结果
    def read_train_result(model_name, model_id):
        """读取模型训练结果"""
        # 首先尝试读取训练集和验证集的合并结果
        file_path = os.path.join(base_path, f'{file_prefix}{model_id}_train_val.csv')
        if os.path.exists(file_path):
            log.info(f'读取{model_name}训练集和验证集结果: {file_path}')
            if file_path.endswith('.pkl'):
                return pd.read_pickle(file_path)
            else:
                return pd.read_csv(file_path)
        
        log.error(f'未找到{model_name}训练结果文件')
        return None
    
    # 读取各个模型的测试结果
    def read_test_result(model_name, model_id):
        """读取模型测试结果"""
        file_path = os.path.join(base_path, f'{file_prefix}{model_id}_test.csv')
        if os.path.exists(file_path):
            log.info(f'读取{model_name}测试结果: {file_path}')
            if file_path.endswith('.pkl'):
                return pd.read_pickle(file_path)
            else:
                return pd.read_csv(file_path)
        
        log.error(f'未找到{model_name}测试结果文件')
        return None
    
    # 处理模型结果，确保都有'score'列和'rank'列
    def process_model_result(df, model_name):
        """处理模型结果，确保有'score'列和'rank'列"""
        if df is None:
            return None
        # 确保有user_id和article_id列
        required_columns = ['user_id', 'article_id', 'score']
        for col in required_columns:
            if col not in df.columns:
                log.error(f'{model_name}结果缺少{col}列')
                return None
        # 如果没有rank列，计算rank
        if 'rank' not in df.columns:
            df['rank'] = df.groupby('user_id')['score'].rank(ascending=False, method='first')
        return df
    
    # 读取和处理所有模型的训练结果
    train_results = {}
    test_results = {}
    valid_models = []
    
    for model_name, model_id in models:
        # 读取训练结果
        train_result = read_train_result(model_name, model_id)
        # 读取测试结果
        test_result = read_test_result(model_name, model_id)
        
        # 处理训练结果
        processed_train = process_model_result(train_result, f'{model_name}训练集')
        # 处理测试结果
        processed_test = process_model_result(test_result, f'{model_name}测试集')
        
        if processed_train is not None and processed_test is not None:
            train_results[model_id] = processed_train
            test_results[model_id] = processed_test
            valid_models.append((model_name, model_id))
            log.info(f'{model_name}训练结果形状: {processed_train.shape}')
            log.info(f'{model_name}测试结果形状: {processed_test.shape}')
        else:
            log.warning(f'{model_name}结果处理失败，将被跳过')
    
    # 检查是否有有效模型
    if not valid_models:
        log.error('没有有效的模型结果，融合失败')
        return
    
    log.info(f'有效模型数量: {len(valid_models)}')
    
    # 合并训练结果
    try:
        # 从第一个有效模型开始
        first_model_name, first_model_id = valid_models[0]
        merged_train = train_results[first_model_id].copy()
        merged_train.rename(columns={'score': f'score_{first_model_id}', 'rank': f'rank_{first_model_id}'}, inplace=True)
        
        # 合并其他模型的训练结果
        for model_name, model_id in valid_models[1:]:
            current_result = train_results[model_id].copy()
            current_result.rename(columns={'score': f'score_{model_id}', 'rank': f'rank_{model_id}'}, inplace=True)
            merged_train = merged_train.merge(current_result, on=['user_id', 'article_id'], how='outer')
    except Exception as e:
        log.error(f'合并训练结果失败: {e}')
        return
    
    # 合并测试结果
    try:
        # 从第一个有效模型开始
        first_model_name, first_model_id = valid_models[0]
        merged_test = test_results[first_model_id].copy()
        merged_test.rename(columns={'score': f'score_{first_model_id}', 'rank': f'rank_{first_model_id}'}, inplace=True)
        
        # 合并其他模型的测试结果
        for model_name, model_id in valid_models[1:]:
            current_result = test_results[model_id].copy()
            current_result.rename(columns={'score': f'score_{model_id}', 'rank': f'rank_{model_id}'}, inplace=True)
            merged_test = merged_test.merge(current_result, on=['user_id', 'article_id'], how='outer')
    except Exception as e:
        log.error(f'合并测试结果失败: {e}')
        return
    
    log.info(f'合并后训练数据形状: {merged_train.shape}')
    log.info(f'合并后测试数据形状: {merged_test.shape}')
    
    # 用0填充训练数据的缺失值
    for model_name, model_id in valid_models:
        score_col = f'score_{model_id}'
        rank_col = f'rank_{model_id}'
        if score_col in merged_train.columns:
            merged_train[score_col] = merged_train[score_col].fillna(0)
            log.info(f'{model_name}训练数据的分数缺失值已用0填充')
        if rank_col in merged_train.columns:
            merged_train[rank_col] = merged_train[rank_col].fillna(0)
            log.info(f'{model_name}训练数据的排名缺失值已用0填充')
    
    # 用0填充测试数据的缺失值
    for model_name, model_id in valid_models:
        score_col = f'score_{model_id}'
        rank_col = f'rank_{model_id}'
        if score_col in merged_test.columns:
            merged_test[score_col] = merged_test[score_col].fillna(0)
            log.info(f'{model_name}测试数据的分数缺失值已用0填充')
        if rank_col in merged_test.columns:
            merged_test[rank_col] = merged_test[rank_col].fillna(0)
            log.info(f'{model_name}测试数据的排名缺失值已用0填充')
    
    # 准备特征列
    feat_cols = []
    for model_name, model_id in valid_models:
        score_col = f'score_{model_id}'
        if score_col in merged_train.columns:
            feat_cols.append(score_col)
            # 检查是否存在rank列，如果不存在则计算
            rank_col = f'rank_{model_id}'
            if rank_col not in merged_train.columns:
                merged_train[rank_col] = merged_train.groupby('user_id')[score_col].rank(ascending=False, method='first')
                merged_test[rank_col] = merged_test.groupby('user_id')[score_col].rank(ascending=False, method='first')
            feat_cols.append(rank_col)
    
    log.info(f'使用的特征列: {feat_cols}')
    
    # 准备训练数据
    trn_x = merged_train[feat_cols]
    trn_y = merged_train['label']
    
    log.info(f'训练数据形状: {trn_x.shape}')
    log.info(f'训练标签分布: {trn_y.value_counts()}')
    
    # 定义并训练逻辑回归模型
    log.info('开始训练逻辑回归模型...')
    lr = LogisticRegression(random_state=seed)
    lr.fit(trn_x, trn_y)
    log.info('逻辑回归模型训练完成')
    
    # 预测测试数据
    tst_x = merged_test[feat_cols]
    merged_test['pred_score'] = lr.predict_proba(tst_x)[:, 1]
    log.info(f'测试数据预测完成，形状: {merged_test.shape}')
    
    # 对训练数据也进行预测，用于评估
    merged_train['pred_score'] = lr.predict_proba(trn_x)[:, 1]
    
    # 评估训练数据的预测结果
    log.info('开始评估训练数据的预测结果...')
    # 对预测结果按 user_id 和 pred_score 排序
    merged_train.sort_values(['user_id', 'pred_score'], inplace=True, ascending=[True, False])
    # 计算用户总数，使用 merged_train 中的用户数
    total = merged_train['user_id'].nunique()
    # 调用 evaluate 函数计算评估指标
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        merged_train, total
    )
    # 记录评估结果
    log.info(
        f'训练数据评估结果: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )
    
    # 合并训练和测试结果
    merged = pd.concat([merged_train, merged_test], sort=False)
    
    # 生成提交文件
    try:
        # 只使用测试数据生成提交结果
        temp_prediction = merged_test[['user_id', 'article_id']].copy()
        temp_prediction['pred'] = merged_test['pred_score']
        
        df_sub = gen_sub(temp_prediction)
        df_sub.sort_values(['user_id'], inplace=True)
        
        # 保存融合结果
        os.makedirs(base_path, exist_ok=True)
        df_sub.to_csv(os.path.join(base_path, 'result_lr.csv'), index=False)
        log.info(f'逻辑回归融合结果已保存到 {os.path.join(base_path, "result_lr.csv")}')
        
        # 保存融合的详细结果
        # 测试数据详细结果
        merged_test['rank'] = merged_test.groupby('user_id')['pred_score'].rank(ascending=False, method='first')
        merged_test[['user_id', 'article_id', 'pred_score', 'rank']].to_csv(os.path.join(base_path, 'merge_test_detailed.csv'), index=False)
        log.info(f'测试数据融合详细结果已保存到 {os.path.join(base_path, "merge_test_detailed.csv")}')
        
        # 训练数据详细结果
        merged_train['rank'] = merged_train.groupby('user_id')['pred_score'].rank(ascending=False, method='first')
        merged_train[['user_id', 'article_id', 'pred_score', 'rank', 'label']].to_csv(os.path.join(base_path, 'merge_train_detailed.csv'), index=False)
        log.info(f'训练数据融合详细结果已保存到 {os.path.join(base_path, "merge_train_detailed.csv")}')
    except Exception as e:
        log.error(f'生成提交文件失败: {e}')
    
    log.info('逻辑回归融合完成')


if __name__ == '__main__':
    main()
