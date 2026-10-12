# 公共模块 — XGBoost 模型训练、评估与预测入口

两条筛选路线 (A: MolGPT, B: yaTCM) 共用同一个 XGBoost 亲和力模型。
本目录收录该模型的**训练、调参、集成、评估与预测**全部材料。

## 文件
| 文件 | 说明 |
|---|---|
| `xgboost_aspa_final.json` | **最终模型** (XGBoost Booster, JSON, 17 MB) |
| `MODEL_CARD.md` | 模型卡片: 训练数据/超参/性能/边界 |
| `predict.py` | 预测入口: 复合物PDB目录 → 精排CSV (与根目录 predict.py 相同, 此副本路径自洽) |
| `2_xgboost_model.ipynb` | 完整训练 Notebook (Kaggle T4 实际执行记录, 含全部输出) |
| `5_finish_ensemble.py` | 集成固化脚本 (XGB+LGBM+CatBoost, 权重 0.33/0.33/0.34) |
| `6_eval_bootstrap.py` | Bootstrap 统计评估 |
| `7_lowdata_eval.py` | 低数据量学习效率评估 |
| `8_plot_lowdata_fig.py` | 评估图绘制 |
| `results/ablation_perseed.csv` | 消融实验逐种子记录 |
| `results/lowdata_results.csv` | 低数据量评估记录 |
| `results/model_evaluation.png` | 模型评估图 |

## 训练数据
PDBbind v2019 refined set (4,849 复合物, 8:2 划分, seed=42),
获取方式见 [../data/数据说明.md](../data/数据说明.md)。

## 性能 (测试集 n=970)
Pearson r = **0.7611**, RMSE = **1.3486** (Optuna 调优后); 集成 r = 0.7645。

## 使用
```bash
# 预测 (首次需先重建预处理参数, 见 predict.py 头部说明)
python predict.py --input <复合物PDB目录> --output results.csv
```

## 随机种子
全流程 seed=42; 消融实验多种子结果见 `results/ablation_perseed.csv`。
