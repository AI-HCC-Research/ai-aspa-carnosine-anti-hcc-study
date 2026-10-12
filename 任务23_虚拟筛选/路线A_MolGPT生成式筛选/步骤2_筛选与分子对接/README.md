# 步骤2 — 筛选与分子对接

## 目的
对 5,000 个生成分子执行逐级筛选: 2D 规则过滤 → 去重 → QED 抽样 → 三维构象 → 分子对接。

## 筛选漏斗
| 阶段 | 方法 | 数量 |
|---|---|---|
| 生成分子库 | 步骤1 输出 | 5,000 |
| 二维过滤与去重 | Lipinski/PAINS 简化SMARTS + 去重 | 3,754 |
| QED 抽样 | QED 排序取 Top | 500 |
| 三维构象与对接 | RDKit ETKDG 构象 + Vina/GNINA 对接 | 499 |

## 代码
| 文件 | 说明 |
|---|---|
| `code/3_unified_screening.py` | 统一筛选主管线: 2D过滤→QED→对接→特征提取→XGBoost精排 (含 PDBbind 特征构建) |
| `code/4_molgpt_screening.py` | MolGPT 支路筛选执行版: GNINA→Vina→RDKit经验 三级对接引擎自动回退, 并生成全部分布图 |

## 输入数据 (`data/`)
| 文件 | 说明 |
|---|---|
| `receptor_4MXU.pdb` | ASPA 受体 (PDB 4MXU) |
| `Carnosine_sentinel.sdf` | 肌肽哨兵分子 (流程自检用) |
| `A3_MGPT_molgpt_only.csv` | MolGPT 生成分子库 |
| `A3_MGPT_machine_filter.csv` | 机器过滤后子集 |

## 输出 (`results/`)
| 文件 | 说明 |
|---|---|
| `prefilter_top500.csv` | QED 抽样 Top500 |
| `property_calibration.csv` | 生成分子性质校准记录 |
| `plots/01~07_*.png` | 库性质/漏斗/打分分布/Top20排名/结构图 |
| `plots/top1~5_MGPT_*.pdb` | Top5 候选对接复合物结构 |

## 运行
```bash
python code/3_unified_screening.py   # 主管线 (自动检测安装依赖, 需 Vina 环境)
python code/4_molgpt_screening.py    # 支路执行 + 绘图
```

## 对接引擎说明
三级回退: GNINA (深度学习打分) → AutoDock Vina → RDKit 经验打分;
实际执行路径记录在结果 CSV 的"对接引擎"列, 打分均为有效负值 (kcal/mol)。
