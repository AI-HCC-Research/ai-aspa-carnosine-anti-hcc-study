#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py — ASPA 靶向小分子 XGBoost 精排: 一键预测入口
=================================================================
功能: 对一批「蛋白-配体对接复合物 PDB 文件」提取 55 维特征
      (38 维三维相互作用 + 17 维配体描述符, 与训练口径完全一致),
      经 StandardScaler 标准化 + 重要性中位数特征选择(54→27)后,
      用训练好的 XGBoost 模型 (models/xgboost_aspa_final.json)
      预测 pKd/-logKd 亲和力的代理分数, 并输出排名 CSV。

与训练的一致性:
  - 特征提取函数与 src/3_unified_screening.py 逐字一致;
  - 预处理参数(特征顺序/scaler均值方差/27个选中特征索引)存于
    models/preprocess_params.json, 由 --build-preprocess 从
    PDBbind 特征缓存确定性重建(seed=42, 基准参数同训练脚本)。

用法:
  1) 预测(需已有 models/preprocess_params.json 或 ensemble_models.pkl):
       python predict.py --input 复合物PDB目录 --output results.csv
  2) 重建预处理参数(在存有 pdbbind_features.csv/labels.npy 的环境执行一次):
       python predict.py --build-preprocess pdbbind_features.csv pdbbind_labels.npy

依赖: numpy, pandas, rdkit, xgboost, scikit-learn
"""

import os
import sys
import json
import argparse
import tempfile
import numpy as np
import pandas as pd

# ---- RDKit ----
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
except ImportError:
    sys.exit("[依赖缺失] 请先安装 RDKit:  pip install rdkit  (或 conda install -c conda-forge rdkit)")

try:
    import xgboost as xgb
except ImportError:
    sys.exit("[依赖缺失] 请先安装 XGBoost:  pip install xgboost")

HERE = os.path.dirname(os.path.abspath(__file__))
# 模型与预处理参数位于 公共模块_XGBoost模型与评估/ (根目录predict.py为其便捷入口)
_COMMON = os.path.join(HERE, '公共模块_XGBoost模型与评估')
MODEL_JSON = os.path.join(_COMMON, 'xgboost_aspa_final.json')
PREPROCESS_JSON = os.path.join(_COMMON, 'preprocess_params.json')
ENSEMBLE_PKL = os.path.join(_COMMON, 'ensemble_models.pkl')

# ============================================================
# 特征定义 (与 src/3_unified_screening.py 完全一致, 勿改顺序)
# ============================================================
HYDROPHOBIC_RES = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO'}
POSITIVE_RES    = {'ARG', 'LYS', 'HIP', 'HIS'}
NEGATIVE_RES    = {'ASP', 'GLU'}
POLAR_RES       = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
AROMATIC_RES    = {'PHE', 'TYR', 'TRP', 'HIS'}


def parse_pdb_atoms(pdb_lines):
    """解析PDB的ATOM/HETATM记录, 返回原子信息列表"""
    atoms = []
    for line in pdb_lines:
        if line.startswith(('ATOM', 'HETATM')):
            try:
                atoms.append({
                    'coord': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                    'element': (line[76:78].strip() if len(line) > 76 else ''.join(c for c in line[12:16] if c.isalpha())[:2]).upper() or 'C',
                    'resname': line[17:20].strip(),
                    'resnum': int(line[22:26]),
                    'chain': line[21].strip(),
                    'atom_name': line[12:16].strip(),
                    'bfactor': float(line[60:66]) if len(line) > 66 and line[60:66].strip() else 0.0,
                })
            except (ValueError, IndexError):
                continue
    return atoms


def compute_interaction_features(protein_atoms, ligand_atoms):
    """计算蛋白-配体三维相互作用特征(38个: 21原始 + 17增强)"""
    if not protein_atoms or not ligand_atoms:
        return None

    prot_coords = np.array([a['coord'] for a in protein_atoms])
    lig_coords  = np.array([a['coord'] for a in ligand_atoms])
    prot_elem   = [a['element'] for a in protein_atoms]
    lig_elem    = [a['element'] for a in ligand_atoms]
    prot_res    = [a['resname'] for a in protein_atoms]
    prot_resnum = [a['resnum'] for a in protein_atoms]
    prot_bf     = [a['bfactor'] for a in protein_atoms]

    diff = lig_coords[:, np.newaxis, :] - prot_coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    dist_safe = np.maximum(dist, 0.5)

    min_dists = np.min(dist, axis=1)
    contact_mask = dist < 5.0
    contact_dists = dist[contact_mask]

    prot_hb = np.array([1 if e in ('N', 'O', 'S') else 0 for e in prot_elem])
    lig_hb  = np.array([1 if e in ('N', 'O', 'S', 'F') else 0 for e in lig_elem])
    hb_mask = (dist < 3.5) & (prot_hb[np.newaxis, :] > 0) & (lig_hb[:, np.newaxis] > 0)

    prot_c = np.array([1 if e == 'C' else 0 for e in prot_elem])
    lig_c  = np.array([1 if e == 'C' else 0 for e in lig_elem])
    hp_mask = (dist < 4.5) & (prot_c[np.newaxis, :] > 0) & (lig_c[:, np.newaxis] > 0)

    prot_pos = np.array([1 if (e == 'N' and r in POSITIVE_RES) else 0 for e, r in zip(prot_elem, prot_res)])
    prot_neg = np.array([1 if (e == 'O' and r in NEGATIVE_RES) else 0 for e, r in zip(prot_elem, prot_res)])
    lig_pos  = np.array([1 if e == 'N' else 0 for e in lig_elem])
    lig_neg  = np.array([1 if e == 'O' else 0 for e in lig_elem])
    salt1 = (dist < 5.0) & (prot_pos[np.newaxis, :] > 0) & (lig_neg[:, np.newaxis] > 0)
    salt2 = (dist < 5.0) & (prot_neg[np.newaxis, :] > 0) & (lig_pos[:, np.newaxis] > 0)

    prot_arom = np.array([1 if (e == 'C' and r in AROMATIC_RES) else 0 for e, r in zip(prot_elem, prot_res)])
    arom_mask = (dist < 5.0) & (prot_arom[np.newaxis, :] > 0) & (lig_c[:, np.newaxis] > 0)

    contact_res = set()
    res_types = {'hydro': 0, 'pos': 0, 'neg': 0, 'polar': 0, 'other': 0}
    for j in range(len(protein_atoms)):
        if np.any(contact_mask[:, j]):
            r = prot_res[j]
            contact_res.add((prot_resnum[j], r))
            if r in HYDROPHOBIC_RES: res_types['hydro'] += 1
            elif r in POSITIVE_RES: res_types['pos'] += 1
            elif r in NEGATIVE_RES: res_types['neg'] += 1
            elif r in POLAR_RES: res_types['polar'] += 1
            else: res_types['other'] += 1
    total_ca = max(sum(res_types.values()), 1)

    prot_bf_arr = np.array(prot_bf)
    contact_prot = np.any(contact_mask, axis=0)
    mean_bf = float(np.mean(prot_bf_arr[contact_prot])) if np.any(contact_prot) else 0.0

    neighbors = np.sum(contact_mask, axis=1)

    decay = np.exp(-dist_safe / 3.0)
    hb_strength = float(np.sum(decay * hb_mask))
    hp_strength = float(np.sum(decay * hp_mask))
    salt_strength = float(np.sum(decay * (salt1 + salt2)))
    arom_strength = float(np.sum(decay * arom_mask))
    total_interaction_strength = float(np.sum(decay * contact_mask))

    sigma = 3.0
    eps = 0.2
    inv_r6 = (sigma / dist_safe) ** 6
    inv_r12 = inv_r6 ** 2
    lj_energy = 4.0 * eps * (inv_r12 - inv_r6)
    lj_contact = lj_energy * contact_mask
    vdw_energy = float(np.sum(lj_contact))

    prot_charge = np.array([1.0 if (e == 'N' and r in POSITIVE_RES) else (-1.0 if (e == 'O' and r in NEGATIVE_RES) else 0.0)
                            for e, r in zip(prot_elem, prot_res)])
    lig_charge = np.array([1.0 if e == 'N' else (-1.0 if e == 'O' else 0.0) for e in lig_elem])
    coulomb = (prot_charge[np.newaxis, :] * lig_charge[:, np.newaxis]) / dist_safe
    elec_energy = float(np.sum(coulomb * contact_mask))

    n_lig_heavy = max(len(ligand_atoms), 1)
    lig_efficiency_hb = float(np.sum(hb_mask)) / n_lig_heavy
    lig_efficiency_hp = float(np.sum(hp_mask)) / n_lig_heavy
    lig_efficiency_contact = float(np.sum(contact_mask)) / n_lig_heavy

    lig_contact_frac = float(np.sum(np.any(contact_mask, axis=1)) / n_lig_heavy)
    prot_contact_frac = float(np.sum(np.any(contact_mask, axis=0)) / max(len(protein_atoms), 1))
    shape_complementarity = lig_contact_frac * prot_contact_frac

    p25_dist = float(np.percentile(min_dists, 25))
    p75_dist = float(np.percentile(min_dists, 75))
    closest_contact = float(np.min(dist[contact_mask])) if np.any(contact_mask) else 99.0

    features = {
        'min_distance': float(np.min(min_dists)),
        'mean_min_distance': float(np.mean(min_dists)),
        'median_min_distance': float(np.median(min_dists)),
        'mean_contact_dist': float(np.mean(contact_dists)) if len(contact_dists) > 0 else 0.0,
        'std_contact_dist': float(np.std(contact_dists)) if len(contact_dists) > 0 else 0.0,
        'num_close_contacts': int(np.sum(contact_mask)),
        'num_lig_atoms_in_contact': int(np.sum(np.any(contact_mask, axis=1))),
        'num_prot_atoms_in_contact': int(np.sum(np.any(contact_mask, axis=0))),
        'contact_ratio': float(np.sum(np.any(contact_mask, axis=1)) / max(len(ligand_atoms), 1)),
        'num_hbonds': int(np.sum(hb_mask)),
        'num_hydrophobic': int(np.sum(hp_mask)),
        'num_salt_bridges': int(np.sum(salt1) + np.sum(salt2)),
        'num_aromatic_contacts': int(np.sum(arom_mask)),
        'num_contact_residues': len(contact_res),
        'pocket_hydro_ratio': res_types['hydro'] / total_ca,
        'pocket_charged_ratio': (res_types['pos'] + res_types['neg']) / total_ca,
        'pocket_polar_ratio': res_types['polar'] / total_ca,
        'mean_contact_bfactor': mean_bf,
        'avg_neighbors': float(np.mean(neighbors)),
        'max_neighbors': float(np.max(neighbors)),
        'hb_decay_strength': hb_strength,
        'hp_decay_strength': hp_strength,
        'salt_decay_strength': salt_strength,
        'arom_decay_strength': arom_strength,
        'total_interaction_strength': total_interaction_strength,
        'vdw_energy': vdw_energy,
        'elec_energy': elec_energy,
        'total_binding_energy': vdw_energy + elec_energy,
        'lig_eff_hb': lig_efficiency_hb,
        'lig_eff_hp': lig_efficiency_hp,
        'lig_eff_contact': lig_efficiency_contact,
        'shape_complementarity': shape_complementarity,
        'lig_contact_fraction': lig_contact_frac,
        'prot_contact_fraction': prot_contact_frac,
        'p25_min_dist': p25_dist,
        'p75_min_dist': p75_dist,
        'closest_contact_dist': closest_contact,
    }
    return features


def compute_ligand_descriptors(mol):
    """RDKit计算配体分子描述符(17个)"""
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol) if mol.GetNumHeavyAtoms() > 0 else mol
    except Exception:
        pass
    return {
        'lig_MW': Descriptors.MolWt(mol),
        'lig_LogP': Descriptors.MolLogP(mol),
        'lig_TPSA': Descriptors.TPSA(mol),
        'lig_num_HBD': Descriptors.NumHDonors(mol),
        'lig_num_HBA': Descriptors.NumHAcceptors(mol),
        'lig_num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'lig_num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'lig_num_heavy_atoms': mol.GetNumHeavyAtoms(),
        'lig_num_rings': rdMolDescriptors.CalcNumRings(mol),
        'lig_num_saturated_rings': rdMolDescriptors.CalcNumSaturatedRings(mol),
        'lig_num_aliphatic_rings': rdMolDescriptors.CalcNumAliphaticRings(mol),
        'lig_num_heteroatoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
        'lig_fraction_sp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'lig_num_amide_bonds': rdMolDescriptors.CalcNumAmideBonds(mol),
        'lig_num_valence_electrons': Descriptors.NumValenceElectrons(mol),
        'lig_formal_charge': Chem.GetFormalCharge(mol),
        'lig_mol_mr': Descriptors.MolMR(mol),
    }


def extract_features_from_complex(complex_pdb_path):
    """从复合物PDB(蛋白ATOM+配体HETATM)提取55维特征"""
    try:
        with open(complex_pdb_path, 'r') as f:
            lines = f.readlines()
        protein_atoms = parse_pdb_atoms([l for l in lines if l.startswith('ATOM')])
        ligand_lines = [l for l in lines if l.startswith('HETATM')]
        ligand_atoms = parse_pdb_atoms(ligand_lines)
        if not protein_atoms or not ligand_atoms:
            return None
        inter = compute_interaction_features(protein_atoms, ligand_atoms)
        if not inter:
            return None
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as tf:
            for l in ligand_lines:
                tf.write(l)
            tf.write("END\n")
            lig_path = tf.name
        try:
            lig_mol = Chem.MolFromPDBFile(lig_path, removeHs=False) or Chem.MolFromPDBFile(lig_path, removeHs=True)
            desc = compute_ligand_descriptors(lig_mol) if lig_mol else None
        except Exception:
            desc = None
        finally:
            os.unlink(lig_path)
        if desc is None:
            heavy = sum(a['element'] != 'H' for a in ligand_atoms)
            desc = {'lig_MW': heavy * 12.0, 'lig_LogP': 0.0, 'lig_TPSA': 0.0, 'lig_num_HBD': 0, 'lig_num_HBA': 0,
                    'lig_num_rotatable_bonds': 0, 'lig_num_aromatic_rings': 0, 'lig_num_heavy_atoms': heavy,
                    'lig_num_rings': 0, 'lig_num_saturated_rings': 0, 'lig_num_aliphatic_rings': 0,
                    'lig_num_heteroatoms': sum(a['element'] in ('N', 'O', 'S', 'F', 'P') for a in ligand_atoms),
                    'lig_fraction_sp3': 0.0, 'lig_num_amide_bonds': 0, 'lig_num_valence_electrons': 0,
                    'lig_formal_charge': 0, 'lig_mol_mr': 0.0}
        return {**inter, **desc}
    except Exception as e:
        print(f"  [错误] {complex_pdb_path}: {e}")
        return None


# ============================================================
# 预处理参数: 构建 / 加载
# ============================================================
def build_preprocess(feat_csv, label_npy):
    """从 PDBbind 特征缓存确定性重建 scaler + median特征选择器。
    与 notebooks/2_xgboost_model.ipynb Cell 3c / src/5_finish_ensemble.py 口径一致:
    seed=42 8:2划分 → StandardScaler → 基准XGBoost(早停) → SelectFromModel(median)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectFromModel

    print("[重建预处理] 加载特征缓存 ...")
    X_df = pd.read_csv(feat_csv)
    y = np.load(label_npy)
    assert len(X_df) == len(y), '特征与标签行数不一致'
    print(f"  特征矩阵 {X_df.shape[0]}样本 × {X_df.shape[1]}列")

    X_tr, X_te, y_tr, y_te = train_test_split(X_df.values, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    base_params = {'objective': 'reg:squarederror', 'max_depth': 6, 'learning_rate': 0.1,
                   'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8,
                   'min_child_weight': 1, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                   'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 50,
                   'tree_method': 'hist'}
    print("[重建预处理] 训练基准模型(早停, CPU hist) ...")
    base_model = xgb.XGBRegressor(**base_params)
    base_model.fit(X_tr_scaled, y_tr, eval_set=[(X_te_scaled, y_te)], verbose=False)

    selector = SelectFromModel(base_model, threshold='median', prefit=True)
    selected_idx = selector.get_support()
    feat_all = X_df.columns.tolist()
    feat_sel = [f for f, s in zip(feat_all, selected_idx) if s]
    print(f"  特征选择: {len(feat_all)}→{len(feat_sel)}个 (重要性中位数阈值)")

    params = {
        'feature_names_all': feat_all,
        'selected_names': feat_sel,
        'selected_idx': [int(i) for i, s in enumerate(selected_idx) if s],
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'note': 'seed=42 8:2 split; StandardScaler; XGBRegressor base(median selector); '
                '与 notebooks/2_xgboost_model.ipynb 及 src/5_finish_ensemble.py 同口径',
    }
    with open(PREPROCESS_JSON, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    print(f"[保存] {PREPROCESS_JSON}")


def load_preprocess():
    """加载预处理参数, 优先 preprocess_params.json, 其次 ensemble_models.pkl"""
    if os.path.exists(PREPROCESS_JSON):
        with open(PREPROCESS_JSON, 'r', encoding='utf-8') as f:
            p = json.load(f)
        return p['feature_names_all'], p['selected_idx'], np.array(p['scaler_mean']), np.array(p['scaler_scale'])
    if os.path.exists(ENSEMBLE_PKL):
        try:
            import joblib
        except ImportError:
            sys.exit("[依赖缺失] pip install joblib")
        cache = joblib.load(ENSEMBLE_PKL)
        scaler = cache['scaler']
        feat_all = cache.get('X_train_df_columns') or cache.get('feature_names_all')
        sel_idx = [int(i) for i, s in enumerate(cache['selected_idx']) if s]
        return feat_all, sel_idx, scaler.mean_, scaler.scale_
    sys.exit("[缺预处理参数] 未找到 models/preprocess_params.json 或 models/ensemble_models.pkl\n"
             "           请先运行: python predict.py --build-preprocess pdbbind_features.csv pdbbind_labels.npy\n"
             "           (pdbbind_features.csv/labels.npy 由 src/3_unified_screening.py 第一阶段生成)")


# ============================================================
# 主流程: 批量预测
# ============================================================
def predict(input_dir, output_csv):
    if not os.path.exists(MODEL_JSON):
        sys.exit(f"[缺模型] {MODEL_JSON}")

    feat_all, sel_idx, mean_, scale_ = load_preprocess()
    print(f"[预处理] 特征{len(feat_all)}维 → 选中{len(sel_idx)}维; scaler已加载")

    booster = xgb.Booster()
    booster.load_model(MODEL_JSON)
    print(f"[模型] 已加载 {MODEL_JSON} (num_feature={booster.num_features()})")

    pdb_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.pdb')])
    if not pdb_files:
        sys.exit(f"[错误] 目录中无 PDB 文件: {input_dir}")
    print(f"[输入] {len(pdb_files)} 个复合物 PDB")

    rows = []
    for pf in pdb_files:
        path = os.path.join(input_dir, pf)
        feats = extract_features_from_complex(path)
        if feats is None:
            print(f"  [跳过] {pf} 特征提取失败")
            continue
        x = pd.DataFrame([feats])
        for col in feat_all:
            if col not in x.columns:
                x[col] = 0.0
        x = x[feat_all].values.astype(float)
        x_scaled = (x - mean_) / scale_
        x_sel = x_scaled[:, sel_idx]
        pred = float(booster.predict(xgb.DMatrix(x_sel))[0])
        rows.append({'complex_pdb': pf, 'xgb_predicted_score': pred})
        print(f"  {pf}: {pred:.4f}")

    if not rows:
        sys.exit("[错误] 无成功预测的复合物")

    df = pd.DataFrame(rows).sort_values('xgb_predicted_score', ascending=False).reset_index(drop=True)
    df.insert(0, 'rank', range(1, len(df) + 1))
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n[输出] {output_csv}  (Top1: {df.iloc[0]['complex_pdb']} = {df.iloc[0]['xgb_predicted_score']:.4f})")
    print("注: 分数为 pKd/-logKd 代理值, 越大预测亲和力越强")


def main():
    ap = argparse.ArgumentParser(description='ASPA XGBoost 精排预测入口')
    ap.add_argument('--input', help='复合物PDB目录 (蛋白ATOM + 配体HETATM)')
    ap.add_argument('--output', default='results.csv', help='输出CSV路径 (默认 results.csv)')
    ap.add_argument('--build-preprocess', nargs=2, metavar=('FEATURES_CSV', 'LABELS_NPY'),
                    help='从PDBbind特征缓存重建预处理参数后退出')
    args = ap.parse_args()

    if args.build_preprocess:
        build_preprocess(args.build_preprocess[0], args.build_preprocess[1])
        return
    if not args.input:
        ap.error('预测模式需要 --input 复合物PDB目录')
    predict(args.input, args.output)


if __name__ == '__main__':
    main()
