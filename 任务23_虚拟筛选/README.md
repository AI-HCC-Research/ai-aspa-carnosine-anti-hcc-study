# ASPA 靶向小分子虚拟筛选与 XGBoost 精排 — 代码提交包

> 按大赛《附件5-代码提交要求》整理, **按实际实验流程逐步骤组织**。
> 项目以人源天冬氨酰酶 **ASPA (PDB 4MXU)** 为靶点, 双路线并行筛选,
> 最终鉴定天然二肽 **肌肽 (Carnosine)** 为综合排名第一的候选分子, 并经 100 ns MD + MM/PBSA 验证。

---

## 1. 双路线流程总览

```
路线A — MolGPT 生成式虚拟筛选 (全代码可复现)
  步骤1 MolGPT训练与定向生成 (ZINC-250K → 5,000分子)
  步骤2 筛选与分子对接 (5000→3754→500→Vina/GNINA对接499)
  步骤3 XGBoost粗筛 (55维特征 → Top20)
  步骤4 Top10最终结果 (双榜共识)

路线B — yaTCM 数据库虚拟筛选 (Schrödinger 商业软件, 传统方案溯源)
  步骤0 原始数据库 (yaTCM/TCMBank, 47,696分子)
  步骤1 蛋白准备、口袋预测与网格 (4MXU + SiteMap + Glide Grid)
  步骤2 LigPrep配体预处理 (47,696→58,048构型)
  步骤3 ADMET QikProp筛选 (58,048→22,227)
  步骤4 HTVS高通量对接 (22,227→5,597)
  步骤5 SP标准精度对接 (5,597→1,120)
  步骤6 XP高精度对接 (1,120→224)
  步骤7 IFD柔性对接 (224→45)
  步骤8 XGBoost精筛 (45个复合物精排)
  步骤9 Top10最终结果 (肌肽排名第一, IFD -8.592)

公共模块 — XGBoost 模型训练/评估/预测 (两路线共用)
MD验证 — ASPA-肌肽 100 ns GROMACS + MM/PBSA
```

## 2. 目录结构

```
提交包_ASPA代码/
├── README.md / requirements.txt / predict.py(主入口)
│
├── 路线A_MolGPT生成式筛选/
│   ├── README.md                        路线A流程说明
│   ├── 步骤1_MolGPT训练与定向生成/        code/(训练,生成) + results/(5000分子)
│   ├── 步骤2_筛选与分子对接/             code/(筛选,对接) + data/(受体,分子库) + results/(Top500,图件)
│   ├── 步骤3_XGBoost粗筛/               results/(Top20)
│   └── 步骤4_Top10最终结果/              final_top10 + 双榜共识Top20
│
├── 路线B_yaTCM数据库筛选/
│   ├── README_流程说明.md               Schrödinger全流程参数完整披露
│   ├── 步骤0_原始数据库/                 yaTCM来源与获取方式
│   ├── 步骤1_蛋白准备_口袋预测与网格/     4MXU + 15构象 + SiteMap + Grid(.in/.sh/.log)
│   ├── 步骤2_LigPrep配体预处理/          .inp/.sh/剔除索引
│   ├── 步骤3_ADMET_QikProp筛选/          参数说明 + 批次示例
│   ├── 步骤4_HTVS高通量对接/             参数说明 + 批次示例
│   ├── 步骤5_SP标准精度对接/             参数说明 + 批次示例
│   ├── 步骤6_XP高精度对接/               参数说明 + 批次示例
│   ├── 步骤7_IFD柔性对接/                .inp/.sh/.log/report.csv (完整日志)
│   ├── 步骤8_XGBoost精筛/               code/(Kaggle版代码) + data/(45复合物打分表)
│   └── 步骤9_Top10最终结果/              FleDock.csv + Top1-5 PDB + 2D图
│
├── 公共模块_XGBoost模型与评估/
│   ├── xgboost_aspa_final.json          ★最终模型 (Pearson 0.7611 / RMSE 1.3486)
│   ├── MODEL_CARD.md                    模型卡片
│   ├── predict.py                       预测入口 (路径自洽副本)
│   ├── 2_xgboost_model.ipynb            完整训练Notebook(含执行输出)
│   ├── 5_finish_ensemble.py 等评估脚本
│   └── results/                         消融/低数据量/评估图
│
├── MD验证_ASPA-肌肽复合物/
│   ├── data/                            GROMACS .xvg + gmx_MMPBSA .dat 原始输出
│   ├── figures/                         5张MD成图
│   └── README.md
│
├── figures_src/                         申报图件绘图源码 + 逐图对照表
├── data/数据说明.md                     全部数据集来源与许可
└── logs/README.md                       运行日志索引与环境指纹
```

---

## 3. 快速开始

### 3.1 环境
```bash
pip install -r requirements.txt
```

### 3.2 一键预测 (主入口)
```bash
# 对路线B Top5 复合物复算 XGBoost 精排:
python predict.py --input 路线B_yaTCM数据库筛选/步骤9_Top10最终结果/ --output results.csv
```
> 首次运行需预处理参数 `公共模块_XGBoost模型与评估/preprocess_params.json`;
> 若缺失, 在存有 PDBbind 特征缓存的环境重建一次 (确定性, seed=42):
> `python predict.py --build-preprocess pdbbind_features.csv pdbbind_labels.npy`

### 3.3 重绘申报图件
```bash
python figures_src/fig_funnel_molgpt.py    # MolGPT筛选漏斗
python figures_src/fig_dual_ranking.py     # Top5双排名
python figures_src/fig_chem_compare.py     # 分子结构对比
python figures_src/fig_md_analysis.py      # MD验证组图(5张)
```

---

## 4. 关键结果

| 结果 | 位置 |
|---|---|
| **肌肽综合排名第一** (双路线双榜共识) | `路线A.../步骤4/final_top20_consensus.csv` + `路线B.../步骤9/FleDock.csv` |
| XGBoost 模型性能 Pearson 0.7611 / RMSE 1.3486 | `公共模块.../MODEL_CARD.md` + notebook 执行记录 |
| MD 验证 (RMSD 收敛, MM/PBSA 负值结合能) | `MD验证_ASPA-肌肽复合物/` |

## 5. 数据与软件披露
- 公开数据集: ZINC-250K / PDBbind v2019 / yaTCM / TCMBank / PDB 4MXU — 获取方式见 `data/数据说明.md`
- 商业软件: Schrödinger 2024-4 (工作站) / 2024-1 (集群), 高校科研授权 — 参数日志见路线B各步骤
- 随机种子: 全流程 seed=42
- 未随包项: MolGPT .pt 权重、ensemble_models.pkl、超大 .maegz/.xtc — 均已在对应步骤 README 注明重建路线

## 6. 常见问题
**Q: 口袋图/2D相互作用图/3D结合图为何无 Python 代码?**
A: 系 Maestro GUI 可视化产物, 按《附件5》传统方案以参数+日志+结构文件溯源 (路线B 步骤1/7/9)。

**Q: 预测分数含义?**
A: pKd/-logKd 代理分数, 越大亲和力预测越强, 用于排序而非绝对结合能。
