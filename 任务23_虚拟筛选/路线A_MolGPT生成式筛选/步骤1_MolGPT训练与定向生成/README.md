# 步骤1 — MolGPT 模型训练与定向生成

## 目的
训练 MolGPT (基于 GPT 的分子生成模型), 并定向采样生成 5,000 个高成药性 (QED) 候选小分子。

## 代码
| 文件 | 说明 |
|---|---|
| `code/1_molgpt_train.py` | MolGPT 训练主脚本, 含三模型消融实验 (完整模型 / 无scaffold / 无属性条件) |
| `code/2_directed_generation.py` | 定向生成脚本, 按 QED 条件采样 5,000 个分子并去重 |

## 输入数据
- **ZINC-250K**: 250,000 个类药 SMILES (MolGPT 官方清洗版)
  获取: https://github.com/devalab/molgpt → `250k_rndm_zinc_drugs_clean_3.csv`

## 输出
| 文件 | 说明 |
|---|---|
| `results/directed_qed_5000.csv` | 定向生成的 5,000 个分子 (SMILES + QED 等性质) |

消融实验逐种子记录见 `公共模块_XGBoost模型与评估/results/ablation_perseed.csv`。

## 运行
```bash
python code/1_molgpt_train.py        # GPU, 数小时; 固定多种子
python code/2_directed_generation.py # 需训练好的 .pt 权重
```

## 说明
- 模型权重 (.pt) 存于训练服务器, 未随包; 重训即可复现 (随机种子已固定)。
- 环境: Python 3.10 + PyTorch (CUDA), 详见根目录 requirements.txt。
