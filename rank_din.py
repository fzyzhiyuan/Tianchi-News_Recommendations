import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import gc

import random
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GroupKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import tensorflow_addons as tfa

# 导入deepctr
from deepctr.models import DIN
from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import Logger, evaluate, gen_sub,gen_detailed_result

warnings.filterwarnings('ignore')
seed = 2026
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='DIN 排序')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test_din.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'DIN 排序，mode: {mode}')

# 配置
MAX_SEQ_LENGTH = 100  # 用户历史行为序列长度
EMBEDDING_DIM = 64   # 嵌入维度
BATCH_SIZE = 1024     # 批次大小
EPOCHS = 2           # 训练轮数

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val, df_val):
        super(MetricsCallback, self).__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.df_val = df_val
    
    def on_epoch_end(self, epoch, logs=None):
        # 预测验证集
        val_pred = self.model.predict(self.x_val, batch_size=BATCH_SIZE)
        df_oof = self.df_val.copy()
        df_oof['pred'] = val_pred.flatten()
        df_oof.sort_values(['user_id', 'pred'], inplace=True, ascending=[True, False])
        
        # 计算指标
        total = df_oof['user_id'].nunique()
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_oof, total
        )
        log.info(f'Epoch {epoch+1} metrics: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10},{hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40},{hitrate_50}, {mrr_50}')


# 数据准备函数
def get_din_feats_columns(df, dense_fea, sparse_fea, behavior_fea, hist_behavior_fea, vocab_sizes, emb_dim=64, max_len=20):
    """
    数据准备函数:
    df: 数据集
    dense_fea: 数值型特征列
    sparse_fea: 离散型特征列
    behavior_fea: 用户的候选行为特征列
    hist_behavior_fea: 用户的历史行为特征列
    embedding_dim: embedding的维度
    max_len: 用户序列的最大长度
    """
    
    # 计算每个离散特征的词汇表大小
    sparse_feature_columns = []
    for feat in sparse_fea:
        sparse_feature_columns.append(SparseFeat(feat, vocabulary_size=vocab_sizes[feat], embedding_dim=emb_dim))
    
    dense_feature_columns = [DenseFeat(feat, 1) for feat in dense_fea]
    
    # 历史行为特征列
    var_feature_columns = [VarLenSparseFeat(SparseFeat(feat, vocabulary_size=vocab_sizes['article_id'],embedding_dim=emb_dim, embedding_name='article_id'), maxlen=max_len) for feat in hist_behavior_fea]
    
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns + var_feature_columns
    
    # 建立x, x是一个字典的形式
    x = {}
    for name in get_feature_names(dnn_feature_columns):
        if name in hist_behavior_fea:
            # 这是历史行为序列
            his_list = [l for l in df[name]]
            x[name] = pad_sequences(his_list, maxlen=max_len, padding='post')      # 二维数组
        else:
            x[name] = df[name].values
    
    return x, dnn_feature_columns

def train_model(df_feature, df_click):
    """
    训练DIN模型
    """
    # global_user_vocab = max(df_feature['user_id'].max(),df_click['user_id'].max()) + 1
    # global_article_vocab = max(df_feature['article_id'].max(),df_click['click_article_id'].max()) + 1
    # global_click_environment_vocab = df_feature['user_last_click_environment'].max() + 1
    # global_click_deviceGroup_vocab = df_feature['user_last_click_deviceGroup'].max() + 1
    # global_click_os_vocab = df_feature['user_last_click_os'].max() + 1
    # global_click_country_vocab = df_feature['user_last_click_country'].max() + 1
    # global_click_region_vocab = df_feature['user_last_click_region'].max() + 1
    # global_click_referrer_type_vocab = df_feature['user_last_click_referrer_type'].max() + 1
    # global_category_vocab = df_feature['category_id'].max() + 1
    
    # 构建用户-物品映射
    df_click_sorted = df_click.sort_values(['user_id', 'click_timestamp'])
    user_item_ = df_click_sorted.groupby('user_id')['click_article_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))
    
    # 构建历史行为序列
    def get_user_hist(user_id, target_article_id):
        hist_articles = user_item_dict.get(user_id, [])
        # 过滤掉目标物品，避免数据泄露
        hist_articles = [item for item in hist_articles if item != target_article_id]
        # 取最近的MAX_SEQ_LENGTH个物品
        hist_articles = hist_articles[-MAX_SEQ_LENGTH:]
        return hist_articles
    
    # 应用函数构建历史行为序列
    df_feature['hist_article_id'] = df_feature.apply(
        lambda x: get_user_hist(x['user_id'], x['article_id']),
        axis=1
    )
    
    # 定义特征列
    # 离散特征
    sparse_fea = ['user_id', 'article_id', 'category_id','user_last_click_environment', 'user_last_click_deviceGroup', 'user_last_click_os', 'user_last_click_country', 'user_last_click_region', 'user_last_click_referrer_type','user_most_click_environment','user_most_click_deviceGroup','user_most_click_os','user_most_click_country','user_most_click_region','user_most_click_referrer_type']
    vocab_sizes = {}
    for col in sparse_fea:
        if col in df_click.columns:
            vocab_sizes[col] = max(df_click[col].max(), df_feature[col].max())+ 1
        elif 'click_'+col in df_click.columns:
            vocab_sizes[col] = max(df_click['click_'+col].max(), df_feature[col].max())+ 1
        else:
            vocab_sizes[col] = df_feature[col].max() + 1
    
    # vocab_sizes = {'user_id':global_user_vocab, 
    #               'article_id':global_article_vocab, 
    #               'category_id':global_category_vocab,
    #               'user_last_click_environment':global_click_environment_vocab,
    #               'user_last_click_deviceGroup':global_click_deviceGroup_vocab,
    #               'user_last_click_os':global_click_os_vocab,
    #               'user_last_click_country':global_click_country_vocab,
    #               'user_last_click_region':global_click_region_vocab,
    #               'user_last_click_referrer_type':global_click_referrer_type_vocab,}
    
    # 候选行为特征
    behavior_fea = ['article_id']
    
    # 历史行为特征
    hist_behavior_fea = ['hist_article_id']
    
    # 数值特征
    # numerical_features = [
    #     'sim_score', 'user_id_cnt', 'article_id_cnt', 'user_id_category_id_cnt',
    #     'user_id_click_article_created_at_ts_diff_mean', 'user_id_click_diff_mean',
    #     'user_click_timestamp_created_at_ts_diff_mean', 'user_click_timestamp_created_at_ts_diff_std',
    #     'user_click_datetime_hour_std', 'user_clicked_article_words_count_mean'
    # ]
    numerical_features = list(
        filter(
            lambda x: x not in sparse_fea+behavior_fea+hist_behavior_fea+['label','created_at_datetime', 'click_datetime'],
            df_feature.columns))

    # 处理缺失值
    for feat in numerical_features:
        if feat in df_feature.columns:
            df_feature[feat] = df_feature[feat].fillna(0)
        else:
            df_feature[feat] = 0
    
    # 数值特征归一化
    dense_fea = numerical_features
    mm = MinMaxScaler()
    for feat in dense_fea:
        df_feature[feat] = mm.fit_transform(df_feature[[feat]])
    
    # 准备数据
    df_train = df_feature[df_feature['label'].notnull()].copy()
    df_test = df_feature[df_feature['label'].isnull()].copy()
    
    del df_feature
    gc.collect()
    
    log.info(f"训练数据大小: {len(df_train)}")
    log.info(f"测试数据大小: {len(df_test)}")
    
    # 准备测试数据
    log.info("准备测试数据...")
    x_tst, dnn_feature_columns = get_din_feats_columns(
        df_test, dense_fea, sparse_fea, behavior_fea, hist_behavior_fea, vocab_sizes,
        emb_dim=EMBEDDING_DIM, max_len=MAX_SEQ_LENGTH
    )
    
    # 五折交叉验证
    kfold = GroupKFold(n_splits=5)
    oof = []
    prediction = df_test[['user_id', 'article_id']].copy()
    prediction['pred'] = 0
    
    for fold_id, (trn_idx, val_idx) in enumerate(
            kfold.split(df_train, df_train['label'], df_train['user_id'])):
        log.info(f'\nFold_{fold_id + 1} Training ================================\n')
        
        # 划分训练集和验证集
        df_trn = df_train.iloc[trn_idx]
        df_val = df_train.iloc[val_idx]
        
        # 准备训练数据
        log.info("准备训练数据...")
        x_trn, _ = get_din_feats_columns(
            df_trn, dense_fea, sparse_fea, behavior_fea, hist_behavior_fea, vocab_sizes,
            emb_dim=EMBEDDING_DIM, max_len=MAX_SEQ_LENGTH
        )
        y_trn = df_trn['label'].values
        
        # 准备验证数据
        log.info("准备验证数据...")
        x_val, _ = get_din_feats_columns(
            df_val, dense_fea, sparse_fea, behavior_fea, hist_behavior_fea, vocab_sizes,
            emb_dim=EMBEDDING_DIM, max_len=MAX_SEQ_LENGTH
        )
        y_val = df_val['label'].values
        
        # 构建模型
        log.info("构建DIN模型...")
        model = DIN(dnn_feature_columns, behavior_fea, 
                    # dnn_use_bn=True,
                    # dnn_dropout=0.5,
                    # l2_reg_dnn=0.0001,
                    # l2_reg_embedding=0.0001,
                    seed=seed)
        model.summary(print_fn=log.info)
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss=tfa.losses.SigmoidFocalCrossEntropy(gamma=2, alpha=0.75),
            # loss='binary_crossentropy',
            metrics=['binary_crossentropy', tf.keras.metrics.AUC()]
        )
        
        # 回调函数
        callbacks = [
            ModelCheckpoint(
                f'../user_data/model/din_model_fold{fold_id}.h5',
                monitor='val_auc',
                save_best_only=True,
                mode='max'
            ),
            MetricsCallback(x_val, y_val, df_val[['user_id', 'article_id', 'label']])
        ]
        
        # 训练模型
        log.info(f"Fold_{fold_id + 1}开始训练模型...")
        history = model.fit(
            x_trn, y_trn,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # 保存模型
        os.makedirs('../user_data/model', exist_ok=True)
        model.save(f'../user_data/model/din_model_fold{fold_id}.h5')
        log.info(f"模型 {fold_id} 保存成功")
        
        # 预测验证集
        val_pred = model.predict(x_val, batch_size=BATCH_SIZE)
        df_oof = df_val[['user_id', 'article_id', 'label']].copy()
        df_oof['pred'] = val_pred.flatten()
        oof.append(df_oof)
        
        # 预测测试集
        test_pred = model.predict(x_tst, batch_size=BATCH_SIZE)
        prediction['pred'] += test_pred.flatten() / 5
    
    # 合并验证集结果
    df_train_val = pd.concat(oof)
    
    # 计算评估指标
    df_train_val.sort_values(['user_id', 'pred'], inplace=True, ascending=[True, False])
    total = df_train_val['user_id'].nunique()
    hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
        df_train_val, total
    )
    log.info(
        f'evaluate:{hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
    )
    
    # 保存所有数据集的预测结果
    os.makedirs('../prediction_result', exist_ok=True)
    
    # 保存训练集和验证集的合并结果
    gen_detailed_result(df_train_val, '../prediction_result/detailed_din_train_val.csv')
    log.info("训练集和验证集预测结果已合并保存")
    
    # 保存测试集预测结果
    gen_detailed_result(prediction, '../prediction_result/detailed_din_test.csv')
    log.info("测试集预测结果已保存")
    
    # 生成提交文件
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    df_sub.to_csv(f'../prediction_result/result_din.csv', index=False)
    log.info("预测完成，结果已保存")

def online_predict(df_feature, df_click):
    """
    线上预测
    """
    global_user_vocab = max(df_feature['user_id'].max(),df_click['user_id'].max()) + 1
    global_article_vocab = max(df_feature['article_id'].max(),df_click['click_article_id'].max()) + 1
    global_category_vocab = df_feature['category_id'].max() + 1
    
    # 构建用户-物品映射
    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))
    
    # 构建历史行为序列
    def get_user_hist(user_id, target_article_id):
        hist_articles = user_item_dict.get(user_id, [])
        # 过滤掉目标物品，避免数据泄露
        hist_articles = [item for item in hist_articles if item != target_article_id]
        # 取最近的MAX_SEQ_LENGTH个物品
        hist_articles = hist_articles[-MAX_SEQ_LENGTH:]
        return hist_articles
    
    # 应用函数构建历史行为序列
    df_feature['hist_article_id'] = df_feature.apply(
        lambda x: get_user_hist(x['user_id'], x['article_id']),
        axis=1
    )
    
    # 定义特征列
    # 离散特征
    sparse_fea = ['user_id', 'article_id', 'category_id']
    vocab_sizes = {'user_id':global_user_vocab, 
                  'article_id':global_article_vocab, 
                  'category_id':global_category_vocab}
    
    # 候选行为特征
    behavior_fea = ['article_id']
    
    # 历史行为特征
    hist_behavior_fea = ['hist_article_id']
    
    # 数值特征
    numerical_features = list(
        filter(
            lambda x: x not in sparse_fea+behavior_fea+hist_behavior_fea+['label','created_at_datetime', 'click_datetime'],
            df_feature.columns))
    
    # 处理缺失值
    for feat in numerical_features:
        if feat in df_feature.columns:
            df_feature[feat] = df_feature[feat].fillna(0)
        else:
            df_feature[feat] = 0
    
    # 数值特征归一化
    dense_fea = numerical_features
    mm = MinMaxScaler()
    for feat in dense_fea:
        df_feature[feat] = mm.fit_transform(df_feature[[feat]])
    
    # 准备测试数据
    x_tst, _ = get_din_feats_columns(
        df_feature, dense_fea, sparse_fea, behavior_fea, hist_behavior_fea, vocab_sizes,
        emb_dim=EMBEDDING_DIM, max_len=MAX_SEQ_LENGTH
    )
    
    # 加载五折模型并预测
    prediction = df_feature[['user_id', 'article_id']].copy()
    prediction['pred'] = 0
    
    from tqdm import tqdm
    for fold_id in tqdm(range(5)):
        # 加载模型，包含自定义层
        from deepctr.layers import AttentionSequencePoolingLayer, NoMask
        model = tf.keras.models.load_model(f'../user_data/model/din_model_fold{fold_id}.h5', 
                                          custom_objects={'AttentionSequencePoolingLayer': AttentionSequencePoolingLayer, 'NoMask': NoMask})
        # 预测
        test_pred = model.predict(x_tst, batch_size=BATCH_SIZE)
        prediction['pred'] += test_pred.flatten() / 5
    
    # 生成提交文件
    gen_detailed_result(prediction,'../prediction_result/detailed_din.csv')
    df_sub = gen_sub(prediction)
    df_sub.sort_values(['user_id'], inplace=True)
    os.makedirs('prediction_result', exist_ok=True)
    df_sub.to_csv(f'prediction_result/din_result.csv', index=False)
    log.info("线上预测完成，结果已保存")

if __name__ == '__main__':
    if mode == 'valid':
        df_feature = pd.read_pickle('../user_data/data/offline/feature.pkl')
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        # 处理类别特征
        for f in df_feature.select_dtypes('object').columns:
            if f in ['user_id', 'article_id', 'category_id']:
                continue
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))
        
        train_model(df_feature, df_click)
    else:
        df_feature = pd.read_pickle('../user_data/data/online/feature.pkl')
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        
        # 处理类别特征
        for f in df_feature.select_dtypes('object').columns:
            if f in ['user_id', 'article_id', 'category_id']:
                continue
            lbl = LabelEncoder()
            df_feature[f] = lbl.fit_transform(df_feature[f].astype(str))
        
        online_predict(df_feature, df_click)