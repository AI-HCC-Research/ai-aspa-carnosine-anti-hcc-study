# 从序列到最优活性构象的 AI 筛选流程

<br />

***

## 一、流程总览

| 步骤 | 内容          | 方法                                    | 执行方式          | 关键产物                                                  |
| -- | ----------- | ------------------------------------- | ------------- | ----------------------------------------------------- |
| 1  | AI 蛋白三维结构预测 | HelixFold3 v3.5.8（云端预训练模型）；本地脚本解析平台结果 | 云端预测 + 本地解析脚本 | 5 个 ranked 构象（pTM≈0.96，pLDDT≈92.7），summary.csv        |
| 2  | 前置构象过滤      | GNN 口袋开放性评估（复现实现，纯 numpy）             | 本地脚本          | 最优起始构象 rank4 → `ASPA_best_start.pdb`，gnn\_scores.csv  |
| 3  | 蛋白结构预处理     | PDBFixer/OpenMM：去平台头/规范链/加氢           | 本地脚本          | `ASPA_ready_pdbfixer.pdb`（2514→5009 原子）；归档 4730 原子版保留 |
| 4  | 分子动力学轨迹分析   | GROMACS（biobb 平台）+ UMAP 降维            | 平台模拟 + 本地脚本   | `UMAP_ASPA_trajectory.png`，开放态构象                      |
| 5  | 口袋特征聚类      | HDBSCAN + UMAP                        | 本地脚本          | 4 簇，`hdbscan_result.csv`                              |
| 6  | 构象活性打分      | 两级随机森林（23 维特征）                        | 本地脚本          | `conformation_ranking.csv`，**Top1 = conf\_13**        |

链路：`序列 → 5 预测构象 → 1 起始构象 → MD 构象池 → 口袋聚类 → RF 排名 → Top1 构象`

***

## 二、目录结构

```
任务1提交_project/
├── predict.py                 # 【主入口】一键复现步骤一→六，产出最终排名
├── README.md                  # 本文件
├── requirements.txt           # 依赖清单
├── src/                       # 核心源代码（含注释、相对路径、固定种子）
│   ├── step1_structure_prediction.py    # 步骤一: 平台结果解析与置信度分析
│   ├── step2_gnn_conformation_filter.py# 步骤二: GNN 开放性过滤 (复现实现)
│   ├── step3_structure_preprocessing.py# 步骤三: PDBFixer 加氢 (产物另存)
│   ├── step4_umap_analysis.py          # 步骤四: MD 轨迹 UMAP
│   ├── step5_hdbscan_clustering.py     # 步骤五: HDBSCAN 聚类
│   ├── step6_data_downloader.py        # 步骤六: 外部数据下载
│   ├── step6_feature_extractor.py      # 步骤六: 特征提取
│   ├── step6_rf_trainer.py             # 步骤六: 两级 RF 训练与打分
│   └── step6_run_all.py                # 步骤六: 一键入口
├── models/                    # 模型权重
│   ├── rf_level1_baseline.pkl         # Level-1 RF（45 样本）
│   ├── rf_level2_enhanced.pkl         # Level-2 RF（102 样本，最终评测用）
│   └── level2_feature_weights.npy     # Level-2 特征权重
├── data/                      # 输入数据（按步骤分子目录）
│   ├── step1/  sequence.fasta, helixfold3_user_input.json
│   ├── step2/  candidate_summary.csv, candidates/rank1~rank5/（5 PDB）
│   ├── step3/  ASPA_best_start.pdb
│   ├── step4/  biobb.MDsetup.xtc/.gro/.tpr/.cpt/.top.zip/.top.pdb, *.xvg
│   ├── step5/  pocket_features.csv（45 行 × 6 列）
│   └── step6/  conformations/（15 PDB）, rcsb_aspa/（19 PDB）, raw_csv/, ...
├── notebooks/                 # 实际运行 Notebook
│   ├── step4_UMAP_ASPA.ipynb          # Colab 轨迹 UMAP 原 notebook
│   └── step6_demo_result_review.ipynb # 结果复核演示
├── results/                   # 运行结果（按步骤分子目录）
│   ├── step1/  ranked_structures/（5 组 PDB/JSON）, summary.csv, *.png
│   ├── step2/  ASPA_best_start.pdb, gnn_scores.csv, step2_gnn_candidate_ranking.png
│   ├── step3/  ASPA_ready.pdb（归档 4730 原子）, ASPA_ready_pdbfixer.pdb（5009）, *.png
│   ├── step4/  UMAP_ASPA_trajectory.png, open_conformations.zip
│   ├── step5/  hdbscan_result.csv, hdbscan_umap.png, representative_pockets.csv ...
│   └── step6/  conformation_ranking.csv【最终结果】, rf_conformation_result.json ...
└── logs/                      # 运行日志与溯源材料
    ├── step4_run_log.md
    ├── step4_biobb_platform_record.md      # biobb 第三方平台使用记录（名称/版本/时间/参数/许可）
    ├── step4_biobb_platform_screenshot.docx# biobb 平台项目页截图
    └── step6/  training_report_level1/2, download_summary.json, 运行环境与随机种子记录.txt
```

***

## 三、环境配置

- **Python** 3.9+（Google Colab Python 3.12 实测；3.9/3.10 兼容）
- 硬件：CPU 即可，步骤六推理模式秒出结果，无需 GPU
- 步骤三依赖 OpenMM/PDBFixer；**pip 版 OpenMM 在 Windows 上加氢可能崩溃**
  （C++ 原生访问违规），建议用 conda 安装该部分依赖（见下方替代方案）；
  未安装时该步报错不中断流程（步骤四使用已提供的轨迹）

**方案 A：pip（适合 Colab / Linux）**

```bash
pip install -r requirements.txt
```

**方案 B：conda + pip 混合（Windows 本地推荐，解决 step3 崩溃）**

```bash
# 1. 用 conda 创建环境 (含 OpenMM/PDBFixer 官方二进制)
conda create -n aspa python=3.12 openmm pdbfixer -c conda-forge -y
conda activate aspa
# 2. 其余依赖用 pip
pip install -r requirements.txt --no-deps openmm pdbfixer
```

***

## 四、快速运行（一键复现）

```bash
python predict.py                 # 步骤一→六，默认用预训练模型推理 (conf_13)
python predict.py --retrain       # 从头重训 RF (结果可能因环境微有差异)
python predict.py --retrain --with-download  # 重训 + 联网下载增强数据
```

> **默认推理模式**直接加载 `models/` 下 Colab 训练的 pkl 模型，
> 秒出结果且与环境无关，稳定复现 conf\_13 Top1 (0.974)。

运行结束后核对 **`results/step6/conformation_ranking.csv`**（最终结果文件），
字段含：排名、构象编号、活性综合分、最高可药性、簇多样性数、簇分布、
RF 三口袋活性概率、CavityPlus 三口袋可药性。

**最终结果：Top1 = conf\_13，活性综合分 0.974，三口袋簇多样性 3/3。**

***

## 五、分步骤复现

### 步骤一：HelixFold3 结构预测（云端）+ 本地结果解析

云端预测复现：

1. 登录 HelixFold3 / PaddleHelix 云平台；
2. 上传 `data/step1/sequence.fasta`（ASPA，313 aa）；
3. 参数与原始记录一致：`model_type=HelixFold3, ensemble=1, recycle=10`
   （原始调用记录见 `data/step1/helixfold3_user_input.json`）；
4. 下载 5 个 ranked 构象及 all\_results.json，放入 `results/step1/ranked_structures/`。
   平台条款：`results/step1/HelixFold3_Terms_of_Use.md`。

本地解析（预测结果文件已随包提供）：

```bash
python src/step1_structure_prediction.py
# 读取 results/step1/ranked_structures/rank1~5/ 下 all_results.json + PDB
# 产出 summary.csv、逐残基 pLDDT 曲线、rank1 pLDDT 着色图
```

### 步骤二：GNN 前置构象过滤（复现实现）

```bash
python src/step2_gnn_conformation_filter.py
# 读取 data/step2/candidates/rank1~5/ 与 candidate_summary.csv
```

- 以 313 个 CA 为节点（12 Å 连边），2 层 SGC 消息传递编码，
  读出两项结构域张开度特征，与 pLDDT 标准化后加权（0.6/0.4）；
- 选出 rank4（**非 pTM 最高者**——预测置信度高不等于口袋开放）；
- 产出 `gnn_scores.csv`、`results/step2/ASPA_best_start.pdb`、候选对比图；
  输出 PDB 经 MD5 与步骤一 rank4 核验一致。

### 步骤三：结构预处理（PDBFixer 标准流程）

```bash
python src/step3_structure_preprocessing.py
# 输入 data/step3/ASPA_best_start.pdb
# 输出 results/step3/ASPA_ready_pdbfixer.pdb（5009 原子）
```

- 移除平台 HEADER/TITLE/REMARK/MODEL、链标识规范化、pH 7.4 加氢；
- 重原子坐标不变，313 残基齐全，C 端 OXT 保留；
- 归档 `ASPA_ready.pdb`（4730 原子）保持原样不覆盖，差异见第七节。

### 步骤四：MD 轨迹 UMAP 分析

```bash
python src/step4_umap_analysis.py
# 默认读取 data/step4/biobb.MDsetup.xtc/.gro
```

- MD 模拟由 biobb（BioExcel Building Blocks，GROMACS 引擎）完成，
  平台使用记录见 `logs/step4_biobb_platform_record.md`（含调用时间
  2026-08-21、输入、关键参数与许可），同目录附平台截图；
- data/step4/ 另归档平台原始产物：.tpr/.cpt/.top.zip/.top.pdb
  及能量、回旋半径、RMSD 等 .xvg 文件；
- 分析：mdtraj 提取每帧 CA 坐标展平 → UMAP（n\_neighbors=15，min\_dist=0.1，seed=42）；
- 输出构象景观图（颜色=帧索引/时间）+ `umap_embedding.csv`。

### 步骤五：HDBSCAN 口袋聚类

```bash
python src/step5_hdbscan_clustering.py
```

- 输入 15 构象 × 3 口袋 = 45 条特征（可药性/体积/深度/疏水表面积）；
- Z-score 标准化后 HDBSCAN 参数网格搜索（轮廓系数选优），得 4 个簇；
- 输出聚类结果、UMAP 散点图、代表口袋、Top10、构象汇总等文件。

### 步骤六：RF 构象打分

```bash
python src/step6_run_all.py                  
--retrain --no-download  # 重训但断网
```

***

## 六、模型声明

### RandomForest 模型

- **输入**：23 维口袋特征（几何 5 + 理化/相互作用 10 + ASPA 特异 5 + 辅助 3）
- **输出**：口袋高可药性概率 → 构象活性综合分
  （70% 最大口袋概率 + 20% 可药性峰值/12 + 10% 簇多样性）
- **特征重要性 Top3（Level-2）**：cavity\_druggability 0.27、
  hydrophobic\_sasa 0.17、pocket\_volume 0.12

### 标签为三源投票伪标签（无实验活性数据，重点说明）

≥2 票标为正样本：① CavityPlus 可药性 > 中位数；
② 启发式规则（体积/疏水比/氢键/Zn 等）≥7/10；
③ Zn 配位 + 催化残基完整度 ≥5 分。

**已知局限**：伪标签非实验真值；RCSB 增强样本口袋为几何规则定义；
绝对概率需实验校准。增强样本与待打分 MD 构象无重叠，采用 OOB + 5 折 CV 评估。

### 第三方资源声明

HelixFold3（百度飞桨 PaddleHelix，云端平台调用）、biobb/GROMACS（Apache-2.0/LGPL）、
OpenMM（LGPL）、PDBFixer（MIT）、
scikit-learn（BSD）、mdtraj（BSD）、umap-learn（BSD）、HDBSCAN（BSD）、
RCSB PDB 19 个 ASPA 结构（wwPDB 学术使用）、CavityPlus 可药性评分。
下载记录见 `results/step6/download_summary.json`（本次 BindingDB/ChEMBL 批量命中为 0，如实记录）。

***

## 七、复现实现与差异声明

1. **步骤一**：预测本身在 HelixFold3 云端完成（无本地推理代码），
   `src/step1_structure_prediction.py` 只对平台下载结果（all\_results.json + PDB）
   做解析、汇总与可视化，不重新预测。
2. **步骤二**：原始 GNN 筛选在团队在线环境完成、代码未归档。
   `src/step2_gnn_conformation_filter.py` 为按相同筛选准则
   （构象开放度为主、pLDDT 为辅，0.6/0.4 加权）编写的确定性复现实现：
   纯 numpy 的 2 层 SGC，全部特征从 PDB 实时计算、无硬编码结果；
   已验证选中 rank4，输出与归档文件 MD5 一致。
3. **步骤三（4730/5009 原子差异）**：归档 `results/step3/ASPA_ready.pdb`
   （4730 原子）经逐原子核查缺 279 个 α 氢（HA），且个别芳香环带
   非标准氢（TRP 多 HD2 / HIS 多 HE2），应为当时定制脚本按简化几何
   规则加氢所致。`src/step3_structure_preprocessing.py` 为 PDBFixer 标准
   流程复现，产物 5009 原子（化学上更完整），**另存为
   `ASPA_ready_pdbfixer.pdb`，归档文件保持原样**，差异在此与
   `results/step3/结果说明.txt` 中如实记录。
4. **步骤六**：标签为三源投票伪标签。

***

## 八、下游衔接

Top1 构象 **conf\_13** → 与实验晶体结构的优化/验证 → MolGPT 约束分子生成
→ DiffDock 对接 + XGBoost 活性预测，服务于"肌肽靶向 ASPA 调控 HCC 谷氨酰胺代谢"课题。
