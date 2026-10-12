# 运行日志（整理自 Google Colab 实际运行记录，2026-08）

## 一、运行环境

| 项目       | 记录                                            |
| -------- | --------------------------------------------- |
| 平台       | Google Colab（T4 GPU 运行时；本分析实际仅用 CPU）          |
| Notebook | UMAP\_ASPA.ipynb                              |
| Python   | 3.12（/usr/local/lib/python3.12/dist-packages） |
| 运行日期     | 2026-04-08                                    |

## 二、依赖安装日志（Cell 1：`!pip install mdtraj umap-learn scikit-learn matplotlib`）

```
Collecting mdtraj
  Downloading mdtraj-1.11.1.post2-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl (8.4 MB)
Requirement already satisfied: umap-learn in /usr/local/lib/python3.12/dist-packages (0.5.12)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.6.1)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.12/dist-packages (3.10.0)
Requirement already satisfied: numpy<2.5,~=2.0 in ... (2.0.x)
Requirement already satisfied: scipy in /usr/local/lib/python3.12/dist-packages (1.16.3)
...
Installing collected packages: mdtraj
Successfully installed mdtraj-1.11.1.post2
```

关键依赖实测版本：mdtraj 1.11.1.post2 / umap-learn 0.5.12 /
scikit-learn 1.6.1 / matplotlib 3.10.0 / numpy 2.0.x / scipy 1.16.3
（与 requirements.txt 一致）。

## 三、分析运行记录（Cell 2：加载轨迹 + UMAP + 绘图）

```
/usr/local/lib/python3.12/dist-packages/umap/umap_.py:1952: UserWarning:
  n_jobs value 1 overridden to 1 by setting random_state.
  warn(...)
```

- **参数**：`n_neighbors=15, min_dist=0.1, n_components=2, random_state=42`
- **上述 UserWarning 属预期现象**：固定随机种子后 umap-learn 强制单线程
  （保证可复现），不影响结果；
- **输出**：`UMAP Projection of ASPA MD Trajectory`（10×8 散点图，
  viridis 颜色映射，颜色条为帧索引 0–50）；
- **导出结果**：`UMap图.png`（40,033 字节），即
  `results/UMAP_ASPA_trajectory.png`。

## 四、随机种子记录

| 环节      | 种子/固定方式                       |
| ------- | ----------------------------- |
| UMAP 降维 | `random_state=42`（含嵌入初始化与负采样） |
| 绘图      | 无随机性（纯散点绘制）                   |

同数据、同版本依赖下重跑 `python main.py` 可复现一致结果。
