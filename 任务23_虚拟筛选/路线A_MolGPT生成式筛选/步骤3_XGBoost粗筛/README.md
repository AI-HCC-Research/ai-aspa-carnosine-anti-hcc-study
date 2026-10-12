# 步骤3 — XGBoost 粗筛

## 目的
对步骤2 对接成功的 499 个分子 pose, 用训练好的 XGBoost 模型进行亲和力打分粗筛。

## 方法
1. 从对接复合物 (蛋白 ATOM + 配体 HETATM) 提取 55 维特征
   (38 维三维相互作用 + 17 维配体描述符);
2. StandardScaler 标准化 + 重要性中位数选择 (54→27 维);
3. `xgboost_aspa_final.json` 预测 pKd 代理分数, 降序排名。

## 执行方式
本步骤由 `步骤2` 主管线脚本自动调用完成 (见 `../步骤2_筛选与分子对接/code/3_unified_screening.py` Phase 6);
也可独立复算:
```bash
cd ../..
python predict.py --input <对接复合物PDB目录> --output results/final_top20_xgboost.csv
```
模型与训练细节: [公共模块_XGBoost模型与评估](../../公共模块_XGBoost模型与评估/)。

## 输出
| 文件 | 说明 |
|---|---|
| `results/final_top20_xgboost.csv` | XGBoost 粗筛 Top20 (分数降序) |
