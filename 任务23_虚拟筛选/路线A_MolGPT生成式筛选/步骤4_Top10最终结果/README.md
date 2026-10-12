# 步骤4 — Top10 最终结果

## 目的
汇总 XGBoost 粗筛与对接打分双榜, 计算综合排名分, 得到最终 Top10 候选分子。

## 方法
对双榜 (XGBoost 排名 + Docking 排名) 分别计算百分位, 等权综合为"综合排名分",
降序得到共识排名; 与 yaTCM 支路 (路线B) Top 候选合并评定时,
**肌肽 (Carnosine) 位列综合排名第一**。

## 输出
| 文件 | 说明 |
|---|---|
| `final_top10_candidates.csv` | 最终 Top10 候选清单 |
| `final_top20_consensus.csv` | 双榜共识 Top20 (含分子名称/SMILES/来源/XGBoost分/对接分/各自百分位/综合排名分) |
| `final_top20_consensus.xlsx` | 同上, Excel 版 |

## 后续验证
排名第一的肌肽进入 MD 验证, 见根目录 [MD验证_ASPA-肌肽复合物](../../MD验证_ASPA-肌肽复合物/)。
