# 步骤2 — LigPrep 配体预处理

## 目的

对 yaTCM 原始库 47,696 个分子做质子化/电离态/立体异构展开, 生成对接就绪的配体库。

## 筛选量级

**47,696 个分子 → 58,048 个电离/立体构型**

## 文件

| 文件                              | 说明                                                                                |
| ------------------------------- | --------------------------------------------------------------------------------- |
| `ligprep_9.inp`                 | LigPrep 输入参数 (完整原文): OPLS\_2005, Epik pH 7.0±2.0, 每分子最多 32 个立体异构体, MAX\_ATOMS=100 |
| `ligprep_9.sh`                  | 提交脚本                                                                              |
| `ligprep_9-dropped-indices.txt` | 被剔除分子的索引记录                                                                        |

## 关键参数 (ligprep\_9.inp)

```text
FORCE_FIELD 16 (OPLS_2005)
EPIK yes          # Epik 电离态预测 pH 7.0 ± 2.0
NUM_STEREOISOMERS 32
DETERMINE_CHIRALITIES no
```

## 输出

`ligprep_9-out.maegz` (58,048 构型, \~70 MB 未随包) → 进入 [步骤3 QikProp](../步骤3_ADMET_QikProp筛选/)

## 软件

Schrödinger 2026-1 LigPrep + Epik。
