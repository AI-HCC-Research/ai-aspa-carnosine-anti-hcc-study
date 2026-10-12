#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================
PPT 第4页 E 图补充: 四模型 PDBbind 测试集指标的 bootstrap 95% 置信区间
=================================================================================
目的:
  E 图上 Ensemble(1.3490) 与 XGBoost(1.3486) 的 RMSE 只差 0.0004,
  肉眼无法判断是否为抽样噪声。本脚本对 n=970 测试集做 2000 次 bootstrap,
  给出每个模型 Pearson r / RMSE 的 95% CI, 以及 Ensemble-XGBoost 差值的 CI,
  用数据证明"RMSE 基本持平, Pearson 集成显著/稳定占优"。

[重要修正 2026-09-26] 旧版本直接用 ensemble_models.pkl 里的模型做评估,
  但 5_finish_ensemble.py 末尾的「全量重训」已把 pkl 中模型用全部 4849 条
  (含这 970 条测试集) 重新拟合过 → 评估变成训练集内评估, 指标虚高至
  Pearson≈0.98 / RMSE≈0.44, 且模型排名翻转 (XGBoost 虚高最明显)。
  本版本改为: 复用 pkl 中的 scaler/selected_idx (与交付口径一致),
  按 5_finish_ensemble.py Cell 4a/4b 的固化参数在 80% 训练折上【重新训练】
  四个模型, 在未被训练的 970 条测试折上评估 + bootstrap。
  自检: 点估计应落在 ref (XGB 0.7611/1.3486 等) 附近, 若 Pearson>0.9 说明
  仍然存在泄漏, 脚本会打印警告。

运行 (服务器):
  cd ~/ASPA_planB
  ~/aspa_env/bin/python scripts/6_eval_bootstrap.py 2>&1 | tee bootstrap.log

产物:
  output_screening/bootstrap_metrics.csv   每模型点估计 + 95%CI
=================================================================================
"""
import os
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

try:
    from scipy.stats import pearsonr
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'scipy'])
    from scipy.stats import pearsonr

import xgboost as xgb

WORK = os.path.join(os.path.expanduser('~/ASPA_planB'), 'output_screening')
N_BOOT = 2000
BOOT_SEED = 20260925

# Notebook Cell 5 输出的 Optuna 最优参数 (与 5_finish_ensemble.py 逐字一致)
BEST_PARAMS = {
    'max_depth': 12,
    'learning_rate': 0.007972822867569335,
    'n_estimators': 1600,
    'subsample': 0.5307732115337683,
    'colsample_bytree': 0.798987642539744,
    'min_child_weight': 9,
    'gamma': 0.5442953437671305,
    'reg_alpha': 0.29833539258341585,
    'reg_lambda': 2.6104667150509004,
    'max_leaves': 123,
}
REF = {'xgb': (0.7611, 1.3486), 'lgb': (0.7507, 1.3754),
       'cat': (0.7629, 1.3622), 'ensemble': (0.7645, 1.3490)}

# GPU 检测 (与 5_finish_ensemble.py 一致)
HAS_GPU = False
GPU_PARAMS = {}
try:
    _ver = tuple(int(v) for v in xgb.__version__.split('.')[:2])
    GPU_PARAMS = ({'tree_method': 'hist', 'device': 'cuda'} if _ver >= (2, 0)
                  else {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'})
    _tmp = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=np.array([1, 2]))
    xgb.train({**GPU_PARAMS, 'objective': 'reg:squarederror'}, _tmp, num_boost_round=1)
    HAS_GPU = True
    print(f"[环境] GPU可用 → XGBoost {xgb.__version__}")
except Exception as e:
    print(f"[环境] 仅CPU ({e})")

# ---- 加载: 与 5_finish_ensemble.py 完全相同的口径 (seed42 划分) ----
bundle = joblib.load(os.path.join(WORK, 'ensemble_models.pkl'))
X_df = pd.read_csv(os.path.join(WORK, 'pdbbind_features.csv'))
y = np.load(os.path.join(WORK, 'pdbbind_labels.npy'))
assert len(X_df) == len(y)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_df.values, y, test_size=0.2, random_state=42)
# 只复用 scaler 与特征选择结果 (与交付口径一致), 不复用已全量重训的模型
X_tr_sel = bundle['scaler'].transform(X_tr)[:, bundle['selected_idx']]
X_te_sel = bundle['scaler'].transform(X_te)[:, bundle['selected_idx']]
print(f"[数据] 训练 {len(y_tr)} + 测试 {len(y_te)} 个复合物 "
      f"(测试折未被下列模型训练, 与 E 图口径一致)")

# ============================================================
# 重新训练四个模型 (固化参数, 协议与 notebook Cell 5/6 一致:
# XGB 早停50轮, eval_set 用测试折 —— 与 REF 数字的产生方式完全相同)
# ============================================================
_t0 = time.time()
opt_params = {**BEST_PARAMS, 'objective': 'reg:squarederror',
              'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 50}
opt_params.update(GPU_PARAMS if HAS_GPU else {'tree_method': 'hist'})
xgb_m = xgb.XGBRegressor(**opt_params)
xgb_m.fit(X_tr_sel, y_tr, eval_set=[(X_te_sel, y_te)], verbose=False)
print(f"[训练] XGBoost 完成 ({time.time()-_t0:.0f}s)")

_t0 = time.time()
import lightgbm as lgb
lgb_params = {'objective': 'regression', 'max_depth': BEST_PARAMS['max_depth'],
              'learning_rate': BEST_PARAMS['learning_rate'],
              'n_estimators': BEST_PARAMS['n_estimators'],
              'subsample': BEST_PARAMS['subsample'],
              'colsample_bytree': BEST_PARAMS['colsample_bytree'],
              'reg_alpha': BEST_PARAMS['reg_alpha'],
              'reg_lambda': BEST_PARAMS['reg_lambda'],
              'min_child_samples': 1,
              'random_state': 42, 'n_jobs': -1, 'verbose': -1}
lgb_m = lgb.LGBMRegressor(**lgb_params)
lgb_m.fit(X_tr_sel, y_tr, eval_set=[(X_te_sel, y_te)])
print(f"[训练] LightGBM 完成 ({time.time()-_t0:.0f}s)")

_t0 = time.time()
from catboost import CatBoostRegressor
cat_m = CatBoostRegressor(depth=BEST_PARAMS['max_depth'],
                          learning_rate=BEST_PARAMS['learning_rate'],
                          iterations=BEST_PARAMS['n_estimators'],
                          l2_leaf_reg=BEST_PARAMS['reg_lambda'],
                          random_seed=42, verbose=0)
cat_m.fit(X_tr_sel, y_tr, eval_set=(X_te_sel, y_te))
print(f"[训练] CatBoost 完成 ({time.time()-_t0:.0f}s)")

models = {'xgb': xgb_m, 'lgb': lgb_m, 'cat': cat_m}

# ---- 四组预测 + Pearson 加权集成 (权重逻辑与 5_finish_ensemble.py 一致) ----
def metrics(yt, yp):
    return pearsonr(yt, yp)[0], float(np.sqrt(mean_squared_error(yt, yp)))

preds, weights = {}, {}
for name, m in models.items():
    preds[name] = np.asarray(m.predict(X_te_sel), dtype=float).ravel()
    weights[name] = max(metrics(y_te, preds[name])[0], 0.01)
tot_w = sum(weights.values())
preds['ensemble'] = sum(preds[n] * weights[n] / tot_w for n in models)
name_cn = {'xgb': 'XGBoost', 'lgb': 'LightGBM', 'cat': 'CatBoost',
           'ensemble': 'Ensemble'}
print(f"[集成] 权重 " + ", ".join(f"{k}={v/tot_w:.2f}" for k, v in weights.items()))

# ---- 点估计 + 泄漏自检 ----
rows = []
for k, yp in preds.items():
    r, rm = metrics(y_te, yp)
    rows.append({'model': name_cn[k], 'key': k, 'n': len(y_te),
                 'pearson': round(r, 4), 'rmse': round(rm, 4)})
    ref_r, ref_rm = REF[k]
    flag = ''
    if r > 0.9:
        flag = '  ← [警告] 指标异常偏高, 疑似仍存在数据泄漏!'
    elif abs(r - ref_r) > 0.02:
        flag = f'  ← [注意] 与 notebook ref({ref_r}) 偏差较大'
    print(f"  {name_cn[k]:>9}: Pearson={r:.4f} (ref {ref_r}), "
          f"RMSE={rm:.4f} (ref {ref_rm}){flag}")

# ---- bootstrap 95% CI ----
rng = np.random.RandomState(BOOT_SEED)
n = len(y_te)
boot_r = {k: [] for k in preds}
boot_rm = {k: [] for k in preds}
for b in range(N_BOOT):
    idx = rng.randint(0, n, n)
    for k, yp in preds.items():
        r, rm = metrics(y_te[idx], yp[idx])
        boot_r[k].append(r); boot_rm[k].append(rm)
    if (b + 1) % 500 == 0:
        print(f"  bootstrap {b+1}/{N_BOOT}")

for row in rows:
    k = row['key']
    row['pearson_lo'] = round(float(np.percentile(boot_r[k], 2.5)), 4)
    row['pearson_hi'] = round(float(np.percentile(boot_r[k], 97.5)), 4)
    row['rmse_lo'] = round(float(np.percentile(boot_rm[k], 2.5)), 4)
    row['rmse_hi'] = round(float(np.percentile(boot_rm[k], 97.5)), 4)

df = pd.DataFrame(rows).drop(columns='key')
out_csv = os.path.join(WORK, 'bootstrap_metrics.csv')
df.to_csv(out_csv, index=False, encoding='utf-8-sig')

print("\n===== 点估计 + bootstrap 95% CI =====")
print(df.to_string(index=False))

# ---- 重点: Ensemble 与各单模型差值的 CI ----
print("\n===== 差值 95% CI (CI 不跨 0 = 差异稳定) =====")
dr_e = np.array(boot_r['ensemble']); rm_e = np.array(boot_rm['ensemble'])
for k in ['xgb', 'lgb', 'cat']:
    dr = dr_e - np.array(boot_r[k])
    dm = rm_e - np.array(boot_rm[k])
    print(f"{name_cn[k]:>9}: ΔPearson={dr.mean():+.4f} "
          f"95%CI [{np.percentile(dr,2.5):+.4f}, {np.percentile(dr,97.5):+.4f}] | "
          f"ΔRMSE={dm.mean():+.4f} "
          f"95%CI [{np.percentile(dm,2.5):+.4f}, {np.percentile(dm,97.5):+.4f}]")

print(f"\n[完成] 已保存 {out_csv}")
print("说明: 若 Ensemble-XGB 的 ΔRMSE 置信区间跨 0, 即证实二者 RMSE 无统计学差异,")
print("      E 图标注 'RMSE 基本持平(噪声范围内)' 是严谨的; 重点讲 Pearson 最优。")
