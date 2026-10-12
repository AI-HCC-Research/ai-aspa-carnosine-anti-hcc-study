# 步骤7 — IFD 柔性对接 (Induced Fit Docking)

## 目的

对 XP 输出的 224 个分子做诱导契合对接, 允许口袋侧链柔性重排, 得到最终 45 个候选化合物的结合模式。

## 筛选量级

**224 个分子 (45 个化合物批次) → 45 个候选 (48 个结合模式)**

## 文件

| 文件                  | 说明                  |
| ------------------- | ------------------- |
| `InducedFit.inp`    | IFD 输入参数 (完整原文)     |
| `InducedFit.sh`     | 集群提交脚本              |
| `report.csv`        | IFD 打分结果汇总          |
| `output_214241.out` | 集群作业输出 (作业号 214241) |

## 运行记录

- 软件: Schrödinger \*\*2026-1 \*\*, Linux 集群 gpu01.local
- 时间: 2025-04-01 13:53 – 18:32
- 配体源: `ligprep_HY_yatcm.sdf` (步骤2 产物)
- 输出: `InducedFit-out.maegz` (10.8 MB, 未随包, 可用 report.csv 核对打分)

## 流向

45 个 IFD 复合物 → [步骤8 XGBoost 精筛](../步骤8_XGBoost精筛/)
