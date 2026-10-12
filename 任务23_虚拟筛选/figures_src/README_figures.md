# figures\_src — 申报图件溯源代码对照表

## 一、代码生成图件对照

| 申报图件                                                             | 绘图脚本                                      | 输入数据                                | 运行命令                                             |
| ---------------------------------------------------------------- | ----------------------------------------- | ----------------------------------- | ------------------------------------------------ |
| MolGPT 虚拟筛选漏斗图 (5000→3754→500→499→20)                            | `fig_funnel_molgpt.py`                    | 脚本内置计数(与各阶段CSV行数一致)                 | `python figures_src/fig_funnel_molgpt.py`        |
| Top5 双模型排名交叉验证柱状图                                                | `fig_dual_ranking.py`                     | `results/final_top20_consensus.csv` | `python figures_src/fig_dual_ranking.py`         |
| NAA / 4MXU配体 / 肌肽 分子结构对比图                                        | `fig_chem_compare.py`                     | RDKit 内置绘制                          | `python figures_src/fig_chem_compare.py`         |
| MD 验证组图 (RMSD/RMSF/SASA/Rg/氢键/MMPBSA/残基分解)                       | `fig_md_analysis.py`                      | `MD验证_ASPA-肌肽复合物/data/*.xvg, *.dat` | `python figures_src/fig_md_analysis.py`          |
| 筛选流程各阶段分布图 (01\_library\_properties, 04\_score\_distributions 等) | `路线A.../步骤2/code/4_molgpt_screening.py`   | 筛选中间CSV                             | 见步骤2 README                                      |
| XGBoost vs Docking 散点图 (03\_xgb\_vs\_dock\_scatter)              | `路线A.../步骤2/code/3_unified_screening.py`  | 精排中间结果                              | 见步骤2 README                                      |
| 低数据量效率图 / 消融柱状图                                                  | `公共模块_XGBoost模型与评估/8_plot_lowdata_fig.py` | `公共模块.../results/*.csv`             | `python 公共模块_XGBoost模型与评估/8_plot_lowdata_fig.py` |

对应输出图件实物已随包存于 `路线A.../步骤2/results/plots/` 与 `MD验证_ASPA-肌肽复合物/figures/`。

## 二、GUI 导出图件 (无代码, 按传统方案溯源)

以下图件由 **Schrödinger Maestro 2026-1 **GUI 手动导出, 属于商业软件可视化产物, 无 Python 生成代码:

| 图件         | 内容                  | 溯源文件位置                                        |
| ---------- | ------------------- | --------------------------------------------- |
| 蛋白三维结构+口袋图 | 4MXU 与 SiteMap 预测口袋 | `路线B.../步骤1/SiteMap.pdb`, `top10_pockets.csv` |
| 2D 相互作用图   | Top1 配体-残基相互作用平面图   | `路线B.../步骤9/Top1-2D.png`                      |
| 3D 结合模式图   | Top1-5 结合构象         | `路线B.../步骤9/Top1-5.pdb`                       |

**复核方式**: 用 Maestro 打开对应 `.pdb`/会话文件即可重现视图; 对接参数与日志见
`路线B.../步骤1/Grid参数/`, `步骤7_IFD柔性对接/` (含完整命令行 `.sh` 与运行日志 `.log`)。

