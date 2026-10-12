# ASPA 靶向小分子虚拟筛选说明。

***

## 1. 流程概述

本研究针对 ASPA 蛋白开展基于结构的虚拟筛选，从 TCM（传统中药）小分子数据库中逐级筛选靶向 ASPA 的候选化合物。全部结构筛选工作由 **Schrödinger 套件**（Glide / LigPrep / Epik / QikProp / Prime / SiteMap / IFD）完成，最终得到 **45 个候选化合物**及其结合模式，作为后续 XGBoost 机器学习精筛的输入。

**筛选漏斗总览：**

| 阶段            | 方法                       | 输入分子数                   | 输出分子数      |
| ------------- | ------------------------ | ----------------------- | ---------- |
| S1 配体库来源      | TCM 数据库（YaTCM/TCMBank）   | —                       | 47,696     |
| S2 配体预处理      | LigPrep + Epik（含立体异构体展开） | 47,696                  | 58,048 个构型 |
| S3 ADMET 性质筛选 | QikProp                  | 58,048                  | 22,227     |
| S4 高通量对接      | Glide HTVS               | 22,227                  | 5,597      |
| S5 标准精度对接     | Glide SP                 | 5,597                   | 1,120      |
| S6 超高精度对接     | Glide XP                 | 1,120                   | 224        |
| S7 柔性对接       | Induced Fit Docking（IFD） | 224（45 个化合物批次，48 个结合模式） | 45         |
| S8 结合模式分析     | Maestro 可视化 + 打分汇总       | 45                      | Top 5      |

***

## 2. 软件与运行环境披露

| 项目            | 内容                                                                                                                |
| ------------- | ----------------------------------------------------------------------------------------------------------------- |
| 软件名称          | Schrödinger Suite（Maestro 图形界面 + 命令行作业系统）                                                                         |
| 版本（受体/Grid 侧） | Schrödinger **2026-1**（build 118，安装目录 `D:\Schordinger26-1`，Windows 工作站 CADD）                                      |
| 版本（柔性对接集群侧）   | Schrödinger **2026-1**（build 135，安装目录 `/work/zsugroup/haiyunma/YQW/Run/APP/Schordinger24-1`，Linux 集群 gpu01.local） |
| 力场            | OPLS\_2005（全程统一）                                                                                                  |
| 调用时间          | Grid：2026-04-25 23:58；IFD：2026-04-26 13:53 – 18:32                                                                |
| 复现性说明         | 商业软件流程不可随代码包分发；本文件按官方默认算法与下述参数完整披露                                                                                |

***

## 3. 数据来源

| 数据               | 来源                                                | 许可/说明                                                             |
| ---------------- | ------------------------------------------------- | ----------------------------------------------------------------- |
| TCM 小分子库         | YaTCM  传统中药成分数据库（下载时间：2026 年）                     | 学术免费使用；原始库经 LigPrep 处理后留存（IFD 日志中的配体源 `ligprep_HY_yatcm.sdf` 即该库） |
| 训练数据（XGBoost 部分） | PDBbind v2019 refined set（DeepChem S3 镜像，免注册公开下载） | 公开学术数据集                                                           |

***

## 4. 分步参数与结果明细

### S1. Glide 受体格点（Grid）生成

- **输入文件**：`Grid-ASPA-4MXU.in`（完整原文）

```text
FORCEFIELD    OPLS_2005
GRID_CENTER   -43.1510764444444, 9.164855777777783, -41.70969411111115
GRIDFILE      Grid-ASPA-4MXU.zip
INNERBOX      10, 10, 10
OUTERBOX      20.299206299209448, 20.299206299209448, 20.299206299209448
RECEP_FILE    Grid-ASPA-4MXU.maegz
```

- **调用记录**：JobId `CADD-0-67eabba6`，Glide 2026-1 build 118，开始 2026-04-24 23:58:32，受体网格 21³×1.0 Å，对接网格 56³，总耗时 18 秒。

### S4. 配体库预处理（LigPrep + Epik）

- **输入文件**：`ligprep_9.inp`（完整原文）

```text
INPUT_FILE_NAME         ligprep_9.sdf
OUT_MAE                 ligprep_9-out.maegz
MAX_ATOMS               100
FORCE_FIELD             16          # OPLS_2005
EPIK                    yes         # Epik pH 7.0 ± 2.0 电离态预测
DETERMINE_CHIRALITIES   no
IGNORE_CHIRALITIES      no
NUM_STEREOISOMERS       32          # 每分子最多 32 个立体异构体
```

- **结果**：47,696 个 TCM 分子展开为 58,048 个电离/立体构型；被剔除分子索引记录于 `ligprep_9-dropped-indices.txt`。

### S5. ADMET 性质筛选（QikProp）

- **操作**：Virtual Screening Workflow（VSW）中分批调用 QikProp（57 个批次输入文件 `VSW-ASPA-4MXU-QIKPROP-001~060-in.maegz`），按 Lipinski 五规则 + 反应性/毒性警示过滤。
- **结果**：58,048 → **22,227** 个分子进入对接。

### S6–S8. Glide 三级对接（HTVS → SP → XP）

统一对接盒 = S3 生成的 `Grid-ASPA-4MXU.zip`，力场 OPLS\_2005；在 VSW 流程中分批执行：

| 级别       | 精度             | 结果文件（批次）                                                                     | 批次数  | 通过                             |
| -------- | -------------- | ---------------------------------------------------------------------------- | ---- | ------------------------------ |
| S6 高通量筛选 | Glide **HTVS** | `VSW-ASPA-4MXU-DOCK_HTVS_1-001~060_lib.maegz`                                | 60   | 22,227 → **5,597**             |
| S7 标准精度  | Glide **SP**   | `VSW-ASPA-4MXU-DOCK_SP_1-001~059_lib.maegz`                                  | 59   | 5,597 → **1,120**              |
| S8 超高精度  | Glide **XP**   | `VSW-ASPA-4MXU-DOCK_XP_1-001~055_pv.maegz`、`VSW-ASPA-4MXU-XP_OUT_1_pv.maegz` | 55+1 | 1,120 → **224**（45 个化合物进入 IFD） |

### S9. 柔性对接 — Induced Fit Docking（IFD）

- **平台**：Linux 集群 gpu01.local，Schrödinger 2026-1 build 135，Glide/Prime 各 64 CPU 并行。
- **调用时间**：开始 2026-04-0 13:53:54（JobId `gpu01-0-67eb7f41`），完成 2026-04-27 18:32:35，总耗时 16,720 秒（约 4 小时 39 分），48/48 子作业全部成功。
- **输入**：受体 `InducedFit_rec.mae`（4MXU 预处理结构）；配体 `InducedFit.maegz`（XP 命中的 45 个化合物）。
- **参数文件** `InducedFit.inp` 关键设置（完整工作流共 6 个 STAGE）：

| STAGE                    | 关键参数                                                                                                                           | 含义               |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| ① GLIDE\_DOCKING2（初对接）   | INNERBOX 10.0；OUTERBOX auto；受体范德华缩放 **0.50**；配体范德华缩放 0.50；DOCKING\_PRECISION **SP**；DOCKING\_POSES\_PER\_LIG **20**；OPLS\_2005 | 硬受体下宽松生成初始姿势     |
| ② COMPILE\_RESIDUE\_LIST | DISTANCE\_CUTOFF **5.0 Å**                                                                                                     | 圈定配体 5 Å 内残基参与优化 |
| ③ PRIME\_REFINEMENT      | NUMBER\_OF\_PASSES **1**；OPLS\_2005；隐式溶剂（无膜）                                                                                   | 侧链重排与能量最小化       |
| ④ SORT\_AND\_FILTER      | 按 `r_psp_Prime_Energy` 保留 **30.0** kcal/mol 窗口                                                                                 | 能量过滤             |
| ⑤ SORT\_AND\_FILTER      | 按 `r_psp_Prime_Energy` 每配体保留 **1** 个                                                                                           | 去冗余              |
| ⑥ GLIDE\_DOCKING2（重对接）   | 受体范德华缩放 **1.00**；配体缩放 0.80；SP；每配体 1 姿势                                                                                         | 优化后受体上精确重对接      |
| ⑦ SCORING                | **IFDScore = 1.0×r\_i\_glide\_gscore + 0.05×r\_psp\_Prime\_Energy**；输出 `report.csv`                                            | 综合评分             |

- **结果**（`report.csv`，48 个结合模式，字段：Entry / Ligand / IFDScore / glide\_gscore / Prime\_Energy / 文件路径）：
  - IFDScore 范围：**-655.39 \~ -643.96**；最佳 Entry 30（IFDScore -655.39，GScore -9.46）
  - 最佳 GScore：Entry 46（**-10.53** kcal/mol）
  - 45 个化合物全部获得有效结合模式，进入最终排名。

### S10. 结合模式分析与最终排名

- **操作**：Maestro 中对 45 个复合物进行对齐、相互作用图（π-π、氢键、盐桥）可视化分析，汇总 IFD/Glide 打分。
- **最终结果**（`FleDock.csv`，45 个化合物按对接打分升序）：

| 排名 | 化合物（English name）                         | 对接打分 (kcal/mol) |
| -- | ----------------------------------------- | --------------- |
| 1  | Boehmeriasin A                            | -9.076          |
| 2  | N-Acetyl-L-cysteine                       | -8.710          |
| 3  | **Carnosine（肌肽）**                         | -8.592          |
| 4  | Isocitric acid                            | -8.435          |
| 5  | Creatine                                  | -8.423          |
| 6  | gamma-Aminobutyric acid                   | -8.306          |
| …  | …（共 45 个，完整见文件）                           | …               |
| 45 | (3R)-2-hydroxymethylbutane-1,2,3,4-tetrol | -3.701          |

- **配套结构文件**：45 个化合物-ASPA 复合物 PDB（`FleDock/` 目录）与 Top1–Top5 对齐结构（`Top1.pdb` \~ `Top5.pdb`），命名即化合物英文名。

***

## 5. 复现性说明与等效开源方案

1. **商业软件部分**：Schrödinger 为商业许可软件，无法随本代码包分发与重跑。本文件已披露全部版本、参数、调用时间与输出，符合"第三方平台使用记录"的溯源要求。
2. **机器学习精筛部分**：完全使用开源工具（Python + XGBoost/RDKit/scikit-learn），可独立复现，详见《02\_XGBoost模型与Notebook代码说明.md》。
3. **等效开源对照方案**（如需无 Schrödinger 环境复核结构筛选趋势）：
   - 受体/配体准备：RDKit + PDBFixer；
   - 口袋检测：fpocket（与 SiteMap 结论可交叉验证 `top10_pockets.csv`）；
   - 对接：AutoDock Vina（对接盒沿用本文件 S3 的中心坐标与 20.3 Å 外盒）；
   - 本项目 Notebook 中已内置该等效路径的说明与可选调用（AutoDock Vina 重对接命令模板）。
4. **下游验证**：Top3 候选肌肽（Carnosine）另经 AutoDock Vina 复对接（最佳结合能 -4.7 kcal/mol，9 个姿势）与 GROMACS 分子动力学 + MM-PBSA 验证，原始数据存于 `对接/ASPA_Carnosine/` 与 `对接/ASPA_Carnosine---动力学模拟/`。

***

*整理日期：2026-08-26；依据原始结果文件逐项核对整理*
