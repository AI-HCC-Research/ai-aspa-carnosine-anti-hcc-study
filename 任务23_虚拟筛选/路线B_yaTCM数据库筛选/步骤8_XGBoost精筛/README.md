# 步骤8 — XGBoost 模型精筛

## 目的

对 IFD 得到的 45 个柔性对接复合物, 用 XGBoost 模型做亲和力精排, 并与对接分数交叉验证。

## 代码

| 文件                               | 说明             |
| -------------------------------- | -------------- |
| `code/9_tcm_screening_kaggle.py` | Kaggle 执行脚本, 与训练 notebook 逐 cell 一致: PDBbind 训练→XGBoost/集成→45 复合物精排 |

模型训练 Notebook (`公共模块_XGBoost模型与评估/2_xgboost_model.ipynb`) 的 Phase 5
即对本步骤 45 个复合物的实际执行记录 (含肌肽定位分析)。

## 输入数据

| 文件              | 说明                                  |
| --------------- | ----------------------------------- |
| `data/ASPA.csv` | 45 个 IFD 复合物的对接分数表                  |
| 45 个复合物 PDB     | 见 [步骤9](../步骤9_Top10最终结果/) 与 IFD 输出 |

## 方法

特征提取 → StandardScaler → 27维选择 → XGBoost 预测, 与 `predict.py` 同口径;
模型文件: `公共模块_XGBoost模型与评估/xgboost_aspa_final.json`。

## 输出

45 个化合物的 XGBoost 精排结果, 与 IFD/Glide 分数共同确定 Top10 → 步骤9。
