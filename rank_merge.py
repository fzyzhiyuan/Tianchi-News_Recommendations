import os
import pandas as pd
import numpy as np

from utils import Logger, gen_sub, evaluate

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger('../user_data/log/test_ensemble.log').logger
log.info('模型集成开始')


def main():
    """
    集成多个模型的结果
    """
    # 定义模型列表，格式: [(模型名称, 模型标识符), ...]
    # 模型标识符用于生成文件路径和权重键
    models = [('DIN', 'din'),
              ('LGB分类', 'lgbcls'),
              ('LGB排序', 'lgbran')]
    
    # 定义模型权重，格式: {模型标识符: 权重值, ...}
    weights = {
        'din': 1,
        'lgbcls': 10,
        'lgbran':10
    }
    
    # 基础路径和文件前缀
    base_path = '../prediction_result'
    file_prefix = 'detailed_'
    
    # 读取各个模型的预测结果
    def read_model_result(model_name, model_id):
        """读取模型结果"""
        # 通过拼接得到文件路径
        file_path = os.path.join(base_path, f'{file_prefix}{model_id}_test.csv')
        if os.path.exists(file_path):
            log.info(f'读取{model_name}结果: {file_path}')
            if file_path.endswith('.pkl'):
                return pd.read_pickle(file_path)
            else:
                return pd.read_csv(file_path)
        log.error(f'未找到{model_name}结果文件: {file_path}')
        return None
    
    # 处理模型结果，确保都有'score'列
    def process_model_result(df, model_name):
        """处理模型结果，确保有'score'列"""
        if df is None:
            return None
        # 确保有user_id和article_id列
        required_columns = ['user_id', 'article_id', 'score']
        for col in required_columns:
            if col not in df.columns:
                log.error(f'{model_name}结果缺少{col}列')
                return None
        return df
    
    # 读取和处理所有模型结果
    model_results = {}
    valid_models = []
    
    for model_name, model_id in models:
        # 读取模型结果
        result = read_model_result(model_name, model_id)
        # 处理模型结果
        processed_result = process_model_result(result, model_name)
        
        if processed_result is not None:
            model_results[model_id] = processed_result
            valid_models.append((model_name, model_id))
            log.info(f'{model_name}结果形状: {processed_result.shape}')
            log.info(f'{model_name}列名: {processed_result.columns.tolist()}')
        else:
            log.warning(f'{model_name}结果处理失败，将被跳过')
    
    # 检查是否有有效模型
    if not valid_models:
        log.error('没有有效的模型结果，集成失败')
        return
    
    log.info(f'有效模型数量: {len(valid_models)}')
    
    # 合并结果
    try:
        # 从第一个有效模型开始
        first_model_name, first_model_id = valid_models[0]
        merged = model_results[first_model_id].copy()
        merged.rename(columns={'score': f'score_{first_model_id}'}, inplace=True)
        
        # 合并其他模型
        for model_name, model_id in valid_models[1:]:
            current_result = model_results[model_id].copy()
            current_result.rename(columns={'score': f'score_{model_id}'}, inplace=True)
            merged = merged.merge(current_result, on=['user_id', 'article_id'], how='outer')
    except Exception as e:
        log.error(f'合并结果失败: {e}')
        return
    
    log.info(f'合并后结果形状: {merged.shape}')
    log.info(f'合并后列名: {merged.columns.tolist()}')
    
    # 计算加权平均
    weighted_sum = 0
    weight_total = 0
    
    # 用0填充缺失值
    for model_name, model_id in valid_models:
        score_col = f'score_{model_id}'
        if score_col in merged.columns:
            merged[score_col] = merged[score_col].fillna(0)
            log.info(f'{model_name}的缺失值已用0填充')
    
    for model_name, model_id in valid_models:
        weight = weights.get(model_id, 0)
        if weight > 0:
            score_col = f'score_{model_id}'
            if score_col in merged.columns:
                weighted_sum += merged[score_col] * weight
                weight_total += weight
                log.info(f'使用{model_name}的权重: {weight}')
            else:
                log.warning(f'{model_name}的分数列不存在，将被跳过')
    
    if weight_total > 0:
        merged['score'] = weighted_sum / weight_total
    else:
        log.error('没有有效的权重配置，集成失败')
        return
    
    # 生成提交文件
    try:
        # 为gen_sub函数创建临时DataFrame，使用'pred'列名
        temp_prediction = merged[['user_id', 'article_id']].copy()
        temp_prediction['pred'] = merged['score']
        
        df_sub = gen_sub(temp_prediction)
        df_sub.sort_values(['user_id'], inplace=True)
        
        # 保存集成结果
        os.makedirs(base_path, exist_ok=True)
        df_sub.to_csv(os.path.join(base_path, 'result.csv'), index=False)
        log.info(f'集成结果已保存到 {os.path.join(base_path, "result.csv")}')
        
        # 保存集成的详细结果
        merged['rank'] = merged.groupby('user_id')['score'].rank(ascending=False, method='first')
        merged[['user_id', 'article_id', 'score', 'rank']].to_csv(os.path.join(base_path, 'ensemble_detailed.csv'), index=False)
        log.info(f'集成详细结果已保存到 {os.path.join(base_path, "ensemble_detailed.csv")}')
    except Exception as e:
        log.error(f'生成提交文件失败: {e}')
    
    log.info('模型集成完成')


if __name__ == '__main__':
    main()
