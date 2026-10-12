# MD 验证 — ASPA-肌肽复合物 100 ns 分子动力学

## 目的
对综合排名第一的 **肌肽 (Carnosine)** 与 ASPA (4MXU) 复合物进行结合稳定性验证。

## 流程
```
GROMACS 2025.4  100 ns MD 模拟 (AMBER 力场, md.tpr/topol.top)
  → 轨迹分析: RMSD / RMSF / SASA / Rg / 氢键数 (.xvg)
  → gmx_MMPBSA 1.6.4  结合自由能 + 残基能量分解 (.dat, 100帧)
```

## 文件
| 目录/文件 | 说明 |
|---|---|
| `data/pic.rmsd.xvg` `rmsd2` `rmsd3` | 蛋白骨架/配体/复合物 RMSD |
| `data/pic.rmsf.xvg` | 逐残基 RMSF |
| `data/pic.area.xvg` | SASA |
| `data/pic.gyrate.xvg` | 回旋半径 Rg |
| `data/pic.hbnum.xvg` | 氢键数 |
| `data/FINAL_RESULTS_MMPBSA.dat` | MM/PBSA 平均结合自由能 (kcal/mol) |
| `data/FINAL_DECOMP_MMPBSA.dat` | 残基能量分解 (gmx_MMPBSA idecomp=2) |
| `figures/md_*.png` | 5 张成图 (RMSD/RMSF/SASA-Rg-Hbonds/结合能/残基分解) |

## 重绘图件
```bash
cd ..
python figures_src/fig_md_analysis.py
```
脚本读取 `data/` 的 .xvg/.dat, 重绘 `figures/` 全部 5 张图。

## 运行环境
Ubuntu 24.04, GROMACS 2025.4, gmx_MMPBSA 1.6.4 (MMPBSA.py v.16.0 内核);
受体 mask :1-313, 配体 mask :314 (MOL), 298.15 K, 100 帧 PBSA。
原始轨迹 (.xtc, 数十 GB) 未随包; .xvg/.dat 已为重绘全部图件的充分数据。
