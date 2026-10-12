# logs 执行记录索引

## 2. XGBoost 训练执行记录

- `公共模块_XGBoost模型与评估/2_xgboost_model.ipynb` — Kaggle 平台真实执行记录,
  cell outputs 含: 数据维度、基准/调优/集成各阶段 Pearson 与 RMSE、特征选择结果、模型保存路径。
- `公共模块_XGBoost模型与评估/5_finish_ensemble.py` 文件头注释逐字记录 notebook 参考指标。

## 3. 评估运行记录

- `公共模块_XGBoost模型与评估/results/ablation_perseed.csv` — 消融×种子逐次结果
- `公共模块_XGBoost模型与评估/results/lowdata_results.csv` — 低数据量逐次记录
- `路线A.../步骤2_筛选与分子对接/results/property_calibration.csv` — 性质校准记录

## 4. MD 运行记录

- `MD验证_ASPA-肌肽复合物/data/*.xvg` 头部含 GROMACS 2025.4 完整命令行
  (如 `gmx rms -s md_0_1.tpr -f md.xtc -o pic.rmsd.xvg`, 创建时间 2026-03-25)
- `FINAL_RESULTS_MMPBSA.dat` 头部含 gmx\_MMPBSA 1.6.4 版本与全部输入拓扑/掩码记录

## 5. 环境指纹

| 环节           | 环境                                                  |
| ------------ | --------------------------------------------------- |
| MolGPT 训练/生成 | Linux 服务器, PyTorch + CUDA, Python 3.10              |
| XGBoost 训练   | Kaggle Notebook, Tesla T4, xgboost 3.4.1            |
| 统一筛选与对接      | Linux 服务器 (GNINA/Vina/RDKit 三级引擎自动回退)               |
| Schrödinger  | Windows 工作站 CADD (2026-1) + Linux 集群 gpu01 (2026-1) |
| MD           | Ubuntu 24.04, GROMACS 2025.4 + gmx\_MMPBSA 1.6.4    |

