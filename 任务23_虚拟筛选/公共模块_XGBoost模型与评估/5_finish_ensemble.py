#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方案B续训 — 直接固化Notebook的Optuna最优参数, 跳过100次调参, 补全 ensemble_models.pkl
=================================================================================
依据: notebookxgboost-84abfa0e8c.ipynb Cell 5 实际输出 (Kaggle T4 原始口径)
  Optuna优化后: RMSE=1.3486, Pearson=0.7611
  最优参数 = BEST_PARAMS (见下, 逐字取自notebook输出)
  Cell 6 集成: LightGBM Pearson=0.7507, CatBoost Pearson=0.7629, XGBoost 0.7611
              权重 xgb=0.33 / lgb=0.33 / cat=0.34, 集成 RMSE=1.3490, Pearson=0.7645

数据一致性已核实:
  Notebook: 训练3879 + 测试970 = 4849样本, 54特征 → median选择27特征 (seed=42)
  服务器:   pdbbind_features.csv = 4849样本 × 54列, JSON模型 num_feature=27 ✓

本脚本流程 (无Optuna, GPU约2~4分钟):
  加载缓存CSV/NPY → 8:2划分(seed42) → StandardScaler → 基准模型 → median选择器
  → 用BEST_PARAMS训练opt XGBoost → LightGBM/CatBoost(同notebook参数) → Pearson定权
  → 全量重训 → ensemble_models.pkl + checkpoint_phase4a.pkl
  各步Pearson/RMSE与notebook参考值对照打印 (XGBoost 3.4.1允许微小版本漂移)

运行:
  cd ~/ASPA_planB
  python scripts/5_finish_ensemble.py
"""

import os, sys, subprocess, time
import numpy as np
import pandas as pd

print("=" * 60)
print("方案B续训: 固化Notebook最优参数 → ensemble_models.pkl (跳过Optuna)")
print("=" * 60)

# ---- 依赖 ----
for _pkg, _pip in [('xgboost', 'xgboost'), ('sklearn', 'scikit-learn'),
                    ('scipy', 'scipy'),
                    ('lightgbm', 'lightgbm'), ('catboost', 'catboost')]:
    try:
        __import__(_pkg)
    except ImportError:
        print(f"[依赖] 安装 {_pip} ...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', _pip], timeout=900)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from scipy.stats import pearsonr
import joblib

# GPU检测 (与notebook/方案B一致)
HAS_GPU = False
GPU_PARAMS = {}
try:
    xgb_ver = tuple(int(v) for v in xgb.__version__.split('.')[:2])
    GPU_PARAMS = ({'tree_method': 'hist', 'device': 'cuda'} if xgb_ver >= (2, 0)
                  else {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'})
    _tmp = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=np.array([1, 2]))
    xgb.train({**GPU_PARAMS, 'objective': 'reg:squarederror'}, _tmp, num_boost_round=1)
    HAS_GPU = True
    print(f"[环境] GPU可用 → XGBoost {xgb.__version__} {GPU_PARAMS}")
except Exception as e:
    print(f"[环境] 仅CPU ({e})")

WORK_DIR = os.path.join(os.path.expanduser('~/ASPA_planB'), 'output_screening')
FEAT_CSV = os.path.join(WORK_DIR, 'pdbbind_features.csv')
LABEL_NPY = os.path.join(WORK_DIR, 'pdbbind_labels.npy')
for _label, _p in [('PDBbind特征矩阵', FEAT_CSV), ('PDBbind标签', LABEL_NPY)]:
    if not os.path.exists(_p):
        raise FileNotFoundError(f"[缺输入] {_label}: {_p}")

# ============================================================
# Notebook Cell 5 输出的 Optuna 最优参数 (逐字固化, 不重跑调优)
# ============================================================
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
# Notebook参考指标 (用于对照, XGBoost版本不同允许微小漂移)
REF = {'rmse_opt': 1.3486, 'pr_opt': 0.7611,
       'rmse_lgb': 1.3754, 'pr_lgb': 0.7507,
       'rmse_cat': 1.3622, 'pr_cat': 0.7629,
       'rmse_ens': 1.3490, 'pr_ens': 0.7645}

# ============================================================
# 加载缓存特征/标签 + 数据口径核对
# ============================================================
X_train_df = pd.read_csv(FEAT_CSV)
y_train = np.load(LABEL_NPY)
assert len(X_train_df) == len(y_train), '特征与标签行数不一致'
print(f"[数据] 缓存特征 {X_train_df.shape[0]}样本 × {X_train_df.shape[1]}列 "
      f"(notebook口径: 4849×54)")
if X_train_df.shape != (4849, 54):
    print(f"  [警告] 数据形状与notebook(4849,54)不同! 固化参数是在该口径上调出的, 请留意")

# ============================================================
# 与 notebook Cell 4/5 一致: 划分 → 标准化 → 基准模型 → median选择器
# ============================================================
X_tr, X_te, y_tr, y_te = train_test_split(X_train_df.values, y_train, test_size=0.2, random_state=42)
print(f"[划分] 训练{X_tr.shape[0]} + 测试{X_te.shape[0]} (notebook: 3879+970)")

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

base_params = {'objective': 'reg:squarederror', 'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 1000,
               'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 1, 'gamma': 0,
               'reg_alpha': 0.1, 'reg_lambda': 1.0, 'random_state': 42, 'n_jobs': -1,
               'early_stopping_rounds': 50}
if HAS_GPU:
    base_params.update(GPU_PARAMS)
else:
    base_params['tree_method'] = 'hist'

print("[训练] 基准模型(早停, 复现median选择器)...")
_t0 = time.time()
base_model = xgb.XGBRegressor(**base_params)
base_model.fit(X_tr_scaled, y_tr, eval_set=[(X_te_scaled, y_te)], verbose=False)
y_base = base_model.predict(X_te_scaled)
rmse_b = np.sqrt(mean_squared_error(y_te, y_base))
pr_b = pearsonr(y_te, y_base)[0]
print(f"  基准: RMSE={rmse_b:.4f}(ref 1.3553), Pearson={pr_b:.4f}(ref 0.7575) ({time.time()-_t0:.0f}s)")

selector = SelectFromModel(base_model, threshold='median', prefit=True)
selected_idx = selector.get_support()
n_selected = int(sum(selected_idx))
print(f"[特征选择] {X_train_df.shape[1]}→{n_selected}个特征 (notebook: 54→27)")
X_tr_sel = X_tr_scaled[:, selected_idx]
X_te_sel = X_te_scaled[:, selected_idx]
feature_names_sel = X_train_df.columns[selected_idx].tolist()
if n_selected != 27:
    print(f"  [警告] 选中特征数{n_selected} ≠ notebook的27 (XGBoost版本差异导致边界特征漂移)")

# ============================================================
# 4a: 直接用固化的最优参数训练 XGBoost (跳过Optuna)
# ============================================================
print("\n[4a] 使用Notebook固化最优参数 (跳过100次Optuna) ...")
print(f"  参数: {BEST_PARAMS}")
opt_params = {**BEST_PARAMS,
              'objective': 'reg:squarederror',
              'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 50}
if HAS_GPU:
    opt_params.update(GPU_PARAMS)
else:
    opt_params['tree_method'] = 'hist'

_t0 = time.time()
opt_model = xgb.XGBRegressor(**opt_params)
opt_model.fit(X_tr_sel, y_tr, eval_set=[(X_te_sel, y_te)], verbose=False)
y_opt = opt_model.predict(X_te_sel)
rmse_opt = np.sqrt(mean_squared_error(y_te, y_opt))
pr_opt = pearsonr(y_te, y_opt)[0]
print(f"  XGBoost: RMSE={rmse_opt:.4f}(ref {REF['rmse_opt']}), "
      f"Pearson={pr_opt:.4f}(ref {REF['pr_opt']}) ({time.time()-_t0:.0f}s)")

# ============================================================
# 4b: 加权集成 — 参数与notebook Cell 6 完全一致 (LGB含reg_alpha/lambda/min_child_samples; Cat含l2_leaf_reg)
# ============================================================
print("\n[4b] 加权模型集成(XGBoost+LightGBM+CatBoost)...")
ensemble_models = {'xgb': opt_model}
ensemble_weights = {}

try:
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
    y_lgb = lgb_m.predict(X_te_sel)
    rmse_lgb = np.sqrt(mean_squared_error(y_te, y_lgb))
    pr_lgb = pearsonr(y_te, y_lgb)[0]
    print(f"  LightGBM: RMSE={rmse_lgb:.4f}(ref {REF['rmse_lgb']}), "
          f"Pearson={pr_lgb:.4f}(ref {REF['pr_lgb']})")
    ensemble_models['lgb'] = lgb_m
    ensemble_weights['lgb'] = max(pr_lgb, 0.01)
except Exception as e:
    print(f"  LightGBM跳过: {e.__class__.__name__}: {e}")
    pr_lgb = None

try:
    from catboost import CatBoostRegressor
    cat_m = CatBoostRegressor(depth=BEST_PARAMS['max_depth'],
                               learning_rate=BEST_PARAMS['learning_rate'],
                               iterations=BEST_PARAMS['n_estimators'],
                               l2_leaf_reg=BEST_PARAMS['reg_lambda'],
                               random_seed=42, verbose=0)
    cat_m.fit(X_tr_sel, y_tr, eval_set=(X_te_sel, y_te))
    y_cat = cat_m.predict(X_te_sel)
    rmse_cat = np.sqrt(mean_squared_error(y_te, y_cat))
    pr_cat = pearsonr(y_te, y_cat)[0]
    print(f"  CatBoost: RMSE={rmse_cat:.4f}(ref {REF['rmse_cat']}), "
          f"Pearson={pr_cat:.4f}(ref {REF['pr_cat']})")
    ensemble_models['cat'] = cat_m
    ensemble_weights['cat'] = max(pr_cat, 0.01)
except Exception as e:
    print(f"  CatBoost跳过: {e.__class__.__name__}: {e}")
    pr_cat = None

ensemble_weights['xgb'] = max(pr_opt, 0.01)

total_w = sum(ensemble_weights.values())
y_ens = np.zeros_like(y_te, dtype=float)
for name in ensemble_models:
    pred = ensemble_models[name].predict(X_te_sel)
    y_ens += pred * (ensemble_weights[name] / total_w)
rmse_ens = np.sqrt(mean_squared_error(y_te, y_ens))
pr_ens = pearsonr(y_te, y_ens)[0]
weight_str = ", ".join(f"{k}={v/total_w:.2f}" for k, v in ensemble_weights.items())
print(f"  加权集成: RMSE={rmse_ens:.4f}(ref {REF['rmse_ens']}), "
      f"Pearson={pr_ens:.4f}(ref {REF['pr_ens']})")
print(f"  权重: {weight_str}  (notebook: xgb=0.33, lgb=0.33, cat=0.34)")

# ============================================================
# 全量重训 (与notebook Cell 6 一致)
# ============================================================
import shutil
_json_path = os.path.join(WORK_DIR, 'xgboost_aspa_final.json')
_bak_path = os.path.join(WORK_DIR, 'xgboost_aspa_final.interrupted_backup.json')
if os.path.exists(_json_path) and not os.path.exists(_bak_path):
    shutil.copy2(_json_path, _bak_path)
    print(f"\n[备份] 服务器中断期产物已保留: {os.path.basename(_bak_path)}")

print("\n[全量重训] 用全部4849数据重训最终模型(去早停)...")
X_all_scaled = scaler.transform(X_train_df.values)
X_all_sel = X_all_scaled[:, selected_idx]
final_params = {k: v for k, v in opt_params.items() if k != 'early_stopping_rounds'}
final_model = xgb.XGBRegressor(**final_params)
final_model.fit(X_all_sel, y_train, verbose=False)
final_model.save_model(_json_path)
for name in ensemble_models:
    if name != 'xgb':
        ensemble_models[name].fit(X_all_sel, y_train)
ensemble_models['xgb'] = final_model
joblib.dump({'models': ensemble_models, 'weights': ensemble_weights,
             'scaler': scaler, 'selector': selector, 'selected_idx': selected_idx,
             'feature_names_sel': feature_names_sel,
             'X_train_df_columns': X_train_df.columns.tolist()},
            os.path.join(WORK_DIR, 'ensemble_models.pkl'))
print("[保存] xgboost_aspa_final.json (固化参数版, 1600棵树) + ensemble_models.pkl")

# checkpoint (与notebook/方案B同构)
checkpoint = {
    'opt_model': opt_model, 'opt_params': opt_params,
    'study_best_params': BEST_PARAMS,          # notebook Cell 5 最优参数
    'scaler': scaler, 'selector': selector, 'selected_idx': selected_idx,
    'feature_names_sel': feature_names_sel,
    'X_all_sel': X_all_sel, 'y_train': y_train,
    'X_train_df_columns': X_train_df.columns.tolist(),
    'rmse_opt': rmse_opt, 'pr_opt': pr_opt,
    'rmse_ens': rmse_ens, 'pr_ens': pr_ens,
}
joblib.dump(checkpoint, os.path.join(WORK_DIR, 'checkpoint_phase4a.pkl'))
print(f"[完成] 集成: RMSE={rmse_ens:.4f}, Pearson={pr_ens:.4f}")

# ============================================================
# 回读校验
# ============================================================
_chk = joblib.load(os.path.join(WORK_DIR, 'ensemble_models.pkl'))
assert set(_chk.keys()) == {'models', 'weights', 'scaler', 'selector',
                            'selected_idx', 'feature_names_sel', 'X_train_df_columns'}
assert _chk['X_train_df_columns'] == X_train_df.columns.tolist()
_dummy = scaler.transform(X_train_df.values[:2])[:, _chk['selected_idx']]
for _n, _m in _chk['models'].items():
    _ = _m.predict(_dummy)
try:
    _n_tree_final = int(final_model.n_estimators)   # sklearn属性, 各版本通用
except AttributeError:
    _n_tree_final = len(final_model.get_booster().get_dump())  # 回退: 数dump出的树
print("=" * 60)
print("ensemble_models.pkl 已通过回读校验")
print(f"  集成成员: {list(_chk['models'].keys())}")
print(f"  JSON最终模型树数: {_n_tree_final} (固化参数n_estimators=1600)")
print("  现在运行 4_molgpt_screening.py 即获得完整三路集成精排")
print("=" * 60)
