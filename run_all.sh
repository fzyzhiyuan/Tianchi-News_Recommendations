#!/bin/bash
cd /data3/fzy/TCxwtj/code
# # 1. 数据处理
python data.py --mode valid

# # # 2. 召回
python recall_w2v.py --mode valid

python recall_binetwork.py --mode valid

python recall_itemcf.py --mode valid

python recall_usercf.py --mode valid

python recall_hot.py --mode valid

python recall_cold.py --mode valid

# # 合并召回结果
# python recall.py --mode valid
python recall_lr.py --mode valid

# 3. 特征工程
python rank_feature.py --mode valid

# 4. 模型训练
# 运行LightGBM模型
python rank_lgb_cls.py --mode valid

python rank_lgb_ran.py --mode valid

# 运行DIN模型
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH python rank_din.py --mode valid

# 5. 模型融合
# python rank_merge.py
python rank_merge_lr.py
