# MODEL CARD — ASPA 亲和力精排模型 (XGBoost)

## 模型标识
| 项目 | 内容 |
|---|---|
| 模型文件 | `xgboost_aspa_final.json` (XGBoost Booster, JSON 格式) |
| 任务类型 | 回归 — 预测蛋白-配体结合亲和力代理分数 (pKd / -logKd 尺度, 越大越强) |
| 框架版本 | XGBoost 3.4.1 (兼容 >= 1.6 的 JSON 加载) |
| 输入维度 | 27 个特征 (由 54 维原始特征经重要性中位数阈值选择) |
| 原始特征 | 38 维蛋白-配体三维相互作用特征 + 17 维 RDKit 配体描述符(共55维, 训练集实得54列), 定义见 `src/3_unified_screening.py` 与 `predict.py` |
| 预处理 | StandardScaler 标准化 (fit 于训练划分, seed=42) → SelectFromModel(median) |
| 随机种子 | 42 (数据划分/基准模型/Optuna/集成成员统一) |

## 训练数据
- **来源**: PDBbind v2019 refined set (DeepChem S3 镜像, 公开学术数据集)
- **样本量**: 4,849 个蛋白-配体复合物 (训练 3,879 / 测试 970, 8:2 划分, random_state=42)
- **标签**: -logKd/Ki (解析自 PDBbind index 文件第4列)
- **许可**: 公开学术数据集, 不可再分发原始结构, 获取方式见 `data/数据说明.md`

## 训练流程
1. 复现入口: `notebooks/2_xgboost_model.ipynb` (完整训练+调参+集成, Kaggle T4 GPU)
2. 快速固化: `src/5_finish_ensemble.py` (跳过 Optuna, 直接固化已确定的最优超参)
3. 关键超参 (Optuna 100 次搜索最优):
   - max_depth=12, learning_rate=0.00797, n_estimators=1600
   - subsample=0.531, colsample_bytree=0.799, min_child_weight=9
   - gamma=0.544, reg_alpha=0.298, reg_lambda=2.610, max_leaves=123
   - 基准对比: max_depth=6 / lr=0.1 / n_estimators=1000 (调优前)

## 性能 (PDBbind v2019 refined 测试集, n=970)
| 模型 | Pearson r | RMSE |
|---|---|---|
| XGBoost 基准 (调优前) | 0.7575 | 1.3553 |
| **XGBoost (Optuna调优, 本模型)** | **0.7611** | **1.3486** |
| LightGBM | 0.7507 | 1.3754 |
| CatBoost | 0.7629 | 1.3622 |
| 加权集成 (0.33/0.33/0.34) | 0.7645 | 1.3490 |

> 注: 不同 XGBoost 版本重训允许 ±0.001 级漂移; 集成模型 `ensemble_models.pkl`
> 体积较大未随包分发, 可用 `src/5_finish_ensemble.py` 在数分钟内重建。

## 预期用途与边界
- ✅ 用途: 对**已完成分子对接的 ASPA (PDB 4MXU) 复合物**进行亲和力精排序
- ✅ 输入: 复合物 PDB (蛋白 ATOM 记录 + 配体 HETATM 记录), 见 `predict.py`
- ⚠️ 边界: 仅适用于与训练集(4MXU口袋类复合物)相似的化学空间; 不输出绝对结合能
- ⚠️ 分数为代理排序指标, 已与 Glide/IFD 对接分数做双榜交叉验证 (见 results/final_top20_consensus.csv)

## 公平性与局限
- 训练集为 PDBbind 通用复合物, 未针对特定人群/物种有偏向性设计
- 预测精度受对接 pose 质量影响; 建议先用 Glide SP/XP 或 Vina 获得合理 pose

## 溯源
- 训练代码: `notebooks/2_xgboost_model.ipynb` (Kaggle 执行记录含全部 cell 输出)
- 评估代码: `src/6_eval_bootstrap.py`, `src/7_lowdata_eval.py`
- 评估报告数据: `results/ablation_perseed.csv`, `results/lowdata_results.csv`
