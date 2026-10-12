# 步骤9 — Top10 最终结果

## 目的

汇总 IFD 对接分数与 XGBoost 精排分数, 输出 yaTCM 支路最终 Top10/Top5 候选及结合模式。

## 文件

| 文件                                 | 说明                                                   |
| ---------------------------------- | ---------------------------------------------------- |
| `FleDock.csv`                      | 45 个候选最终排名表 (**Top3 = Carnosine 肌肽, IFD 分数 -8.592**) |
| `Top1.pdb` – `Top5.pdb`            | Top5 候选复合物结构 (蛋白+配体, 可直接用于 predict.py 复算)            |
| `Top1-2D.png`                      | Top1 配体-残基二维相互作用图 (Maestro GUI 导出)                   |
| `screening_funnel_schrodinger.png` | yaTCM 支路筛选漏斗图                                        |
| `xgb_top5_structures.png`          | Top5 结构总览图                                           |

## 结果要点

- **肌肽 (Carnosine) 在本支路排名第一** (IFD -8.592 kcal/mol);
- Top1-5 结合模式 PDB 为 MD 验证与申报书 3D/2D 结合图的源文件。

## 后续验证

肌肽-ASPA 复合物进入 100 ns MD + MM/PBSA 验证:
[MD验证\_ASPA-肌肽复合物](../../MD验证_ASPA-肌肽复合物/)。
