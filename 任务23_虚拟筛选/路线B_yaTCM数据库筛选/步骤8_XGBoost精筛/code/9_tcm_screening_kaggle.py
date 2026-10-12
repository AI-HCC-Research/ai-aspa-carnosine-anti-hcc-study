#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 路线B 步骤8: XGBoost 精筛 (Kaggle 执行脚本)
# 本文件由 notebookxgboost-84abfa0e8c.ipynb 原样转出 (15 个 code cell, # %% 分隔),
# 与 notebook 逻辑完全一致; 运行环境: Kaggle Python 3.12 + T4 GPU, 数据集 aspa-docking-data。

# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# %%
# %% ===================== 环境准备 =====================
import os, sys, gzip, tarfile, warnings, math, time, subprocess, tempfile
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')

# ---- 检测GPU ----
# 注意: XGBoost 2.0+移除了gpu_hist, 改用 tree_method='hist' + device='cuda'
HAS_GPU = False
GPU_PARAMS = {}  # 根据XGBoost版本动态填充
try:
    import xgboost as xgb
    xgb_ver = tuple(int(v) for v in xgb.__version__.split('.')[:2])
    if xgb_ver >= (2, 0):
        GPU_PARAMS = {'tree_method': 'hist', 'device': 'cuda'}       # XGBoost 2.0+
    else:
        GPU_PARAMS = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}  # 旧版
    _tmp = xgb.DMatrix(np.array([[1,2],[3,4]]), label=np.array([1,2]))
    try:
        xgb.train({**GPU_PARAMS, 'objective':'reg:squarederror'}, _tmp, num_boost_round=1)
        HAS_GPU = True
        print(f"[环境] GPU可用 → XGBoost {xgb.__version__} 使用 {GPU_PARAMS}")
    except Exception as e:
        HAS_GPU = False
        print(f"[环境] 仅CPU → XGBoost {xgb.__version__} 使用hist (GPU检测失败: {e})")
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','xgboost'])
    import xgboost as xgb
    print("[环境] 已安装XGBoost, 仅CPU模式")

# ---- 确保RDKit ----
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
    print("[环境] RDKit已就绪")
except ImportError:
    subprocess.run([sys.executable,'-m','pip','install','-q','rdkit'])
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

# 屏蔽RDKit的化学结构警告/错误输出(不影响功能,只是不刷屏)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# ---- 路径配置 ----
KAGGLE_INPUT = '/kaggle/input'
ASPA_DATA_DIR = os.path.join(KAGGLE_INPUT, 'datasets', 'dongxiaojia0504', 'aspa-docking-data')
FLEDOCK_DIR   = ASPA_DATA_DIR   # 45个PDB文件直接在数据集根目录
SCREENING_CSV = os.path.join(ASPA_DATA_DIR, 'ASPA.csv')
WORK_DIR      = '/kaggle/working'
PDBBIND_DIR   = os.path.join(WORK_DIR, 'pdbbind_data')
os.makedirs(PDBBIND_DIR, exist_ok=True)

# 性能达标标准(Pearson R 或 RMSE达到任一即视为达标)
TARGET_PEARSON = 0.65
TARGET_RMSE    = 1.5

print("=" * 60)
print("环境准备完成 | GPU={} | ASPA数据={}".format(HAS_GPU, ASPA_DATA_DIR))
print("=" * 60)

# %%
# %% ===================== 公共函数: PDB解析与特征提取 =====================

# 氨基酸残基分类
HYDROPHOBIC_RES = {'ALA','VAL','LEU','ILE','PHE','TRP','MET','PRO'}
POSITIVE_RES    = {'ARG','LYS','HIP','HIS'}
NEGATIVE_RES    = {'ASP','GLU'}
POLAR_RES       = {'SER','THR','ASN','GLN','TYR','CYS'}
AROMATIC_RES    = {'PHE','TYR','TRP','HIS'}

def parse_pdb_atoms(pdb_lines):
    """解析PDB的ATOM/HETATM记录, 返回原子信息列表"""
    atoms = []
    for line in pdb_lines:
        if line.startswith(('ATOM','HETATM')):
            try:
                atoms.append({
                    'coord': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                    'element': (line[76:78].strip() if len(line)>76 else ''.join(c for c in line[12:16] if c.isalpha())[:2]).upper() or 'C',
                    'resname': line[17:20].strip(),
                    'resnum': int(line[22:26]),
                    'chain': line[21].strip(),
                    'atom_name': line[12:16].strip(),
                    'bfactor': float(line[60:66]) if len(line)>66 and line[60:66].strip() else 0.0,
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

    # 全距离矩阵 (Nl x Np)
    diff = lig_coords[:, np.newaxis, :] - prot_coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    # 避免除零: 距离最小值钳制
    dist_safe = np.maximum(dist, 0.5)

    # 距离特征
    min_dists = np.min(dist, axis=1)
    contact_mask = dist < 5.0
    contact_dists = dist[contact_mask]

    # 氢键 (N/O...N/O < 3.5Å)
    prot_hb = np.array([1 if e in ('N','O','S') else 0 for e in prot_elem])
    lig_hb  = np.array([1 if e in ('N','O','S','F') else 0 for e in lig_elem])
    hb_mask = (dist < 3.5) & (prot_hb[np.newaxis,:]>0) & (lig_hb[:,np.newaxis]>0)

    # 疏水接触 (C...C < 4.5Å)
    prot_c = np.array([1 if e=='C' else 0 for e in prot_elem])
    lig_c  = np.array([1 if e=='C' else 0 for e in lig_elem])
    hp_mask = (dist < 4.5) & (prot_c[np.newaxis,:]>0) & (lig_c[:,np.newaxis]>0)

    # 盐桥 (正电N...负电O < 5Å)
    prot_pos = np.array([1 if (e=='N' and r in POSITIVE_RES) else 0 for e,r in zip(prot_elem,prot_res)])
    prot_neg = np.array([1 if (e=='O' and r in NEGATIVE_RES) else 0 for e,r in zip(prot_elem,prot_res)])
    lig_pos  = np.array([1 if e=='N' else 0 for e in lig_elem])
    lig_neg  = np.array([1 if e=='O' else 0 for e in lig_elem])
    salt1 = (dist < 5.0) & (prot_pos[np.newaxis,:]>0) & (lig_neg[:,np.newaxis]>0)
    salt2 = (dist < 5.0) & (prot_neg[np.newaxis,:]>0) & (lig_pos[:,np.newaxis]>0)

    # 芳环接触 (芳香残基C...配体C < 5Å)
    prot_arom = np.array([1 if (e=='C' and r in AROMATIC_RES) else 0 for e,r in zip(prot_elem,prot_res)])
    arom_mask = (dist < 5.0) & (prot_arom[np.newaxis,:]>0) & (lig_c[:,np.newaxis]>0)

    # 接触残基统计
    contact_res = set()
    res_types = {'hydro':0,'pos':0,'neg':0,'polar':0,'other':0}
    for j in range(len(protein_atoms)):
        if np.any(contact_mask[:,j]):
            r = prot_res[j]
            contact_res.add((prot_resnum[j], r))
            if r in HYDROPHOBIC_RES: res_types['hydro'] += 1
            elif r in POSITIVE_RES: res_types['pos'] += 1
            elif r in NEGATIVE_RES: res_types['neg'] += 1
            elif r in POLAR_RES: res_types['polar'] += 1
            else: res_types['other'] += 1
    total_ca = max(sum(res_types.values()), 1)

    # B-factor
    prot_bf_arr = np.array(prot_bf)
    contact_prot = np.any(contact_mask, axis=0)
    mean_bf = float(np.mean(prot_bf_arr[contact_prot])) if np.any(contact_prot) else 0.0

    # 邻居数
    neighbors = np.sum(contact_mask, axis=1)

    # ======== 增强特征: 距离衰减权重 + 能量近似 + 配体效率 ========
    # 距离衰减权重 (指数衰减, sigma=3.0Å, 模拟相互作用强度随距离衰减)
    decay = np.exp(-dist_safe / 3.0)

    # 距离衰减加权的氢键强度
    hb_strength = float(np.sum(decay * hb_mask))

    # 距离衰减加权的疏水接触强度
    hp_strength = float(np.sum(decay * hp_mask))

    # 距离衰减加权的盐桥强度
    salt_strength = float(np.sum(decay * (salt1 + salt2)))

    # 距离衰减加权的芳环接触强度
    arom_strength = float(np.sum(decay * arom_mask))

    # 总相互作用强度 (所有原子对的衰减权重之和)
    total_interaction_strength = float(np.sum(decay * contact_mask))

    # 简化的Lennard-Jones势能近似 (排斥-吸引)
    # LJ(r) = 4*eps*[(sigma/r)^12 - (sigma/r)^6], sigma~3.0Å, eps~0.2 kcal/mol
    sigma = 3.0
    eps = 0.2
    inv_r6 = (sigma / dist_safe) ** 6
    inv_r12 = inv_r6 ** 2
    lj_energy = 4.0 * eps * (inv_r12 - inv_r6)
    # 仅计算接触范围内的能量 (避免远距离噪声)
    lj_contact = lj_energy * contact_mask
    vdw_energy = float(np.sum(lj_contact))

    # 静电相互作用近似 (库仑定律, 简化电荷分配)
    # 正电: N(+1), 负电: O(-1), 中性: 0
    prot_charge = np.array([1.0 if (e=='N' and r in POSITIVE_RES) else (-1.0 if (e=='O' and r in NEGATIVE_RES) else 0.0)
                            for e, r in zip(prot_elem, prot_res)])
    lig_charge = np.array([1.0 if e=='N' else (-1.0 if e=='O' else 0.0) for e in lig_elem])
    coulomb = (prot_charge[np.newaxis, :] * lig_charge[:, np.newaxis]) / dist_safe
    elec_energy = float(np.sum(coulomb * contact_mask))

    # 配体效率: 每个重原子的平均相互作用数
    n_lig_heavy = max(len(ligand_atoms), 1)
    lig_efficiency_hb = float(np.sum(hb_mask)) / n_lig_heavy
    lig_efficiency_hp = float(np.sum(hp_mask)) / n_lig_heavy
    lig_efficiency_contact = float(np.sum(contact_mask)) / n_lig_heavy

    # 形状互补性: 配体原子接触率 vs 蛋白原子接触率
    lig_contact_frac = float(np.sum(np.any(contact_mask, axis=1)) / n_lig_heavy)
    prot_contact_frac = float(np.sum(np.any(contact_mask, axis=0)) / max(len(protein_atoms), 1))
    shape_complementarity = lig_contact_frac * prot_contact_frac

    # 距离分布的百分位数
    p25_dist = float(np.percentile(min_dists, 25))
    p75_dist = float(np.percentile(min_dists, 75))

    # 最近接触距离
    closest_contact = float(np.min(dist[contact_mask])) if np.any(contact_mask) else 99.0

    features = {
        # 原始21个特征
        'min_distance': float(np.min(min_dists)),
        'mean_min_distance': float(np.mean(min_dists)),
        'median_min_distance': float(np.median(min_dists)),
        'mean_contact_dist': float(np.mean(contact_dists)) if len(contact_dists)>0 else 0.0,
        'std_contact_dist': float(np.std(contact_dists)) if len(contact_dists)>0 else 0.0,
        'num_close_contacts': int(np.sum(contact_mask)),
        'num_lig_atoms_in_contact': int(np.sum(np.any(contact_mask, axis=1))),
        'num_prot_atoms_in_contact': int(np.sum(np.any(contact_mask, axis=0))),
        'contact_ratio': float(np.sum(np.any(contact_mask, axis=1)) / max(len(ligand_atoms),1)),
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
        # 增强17个特征: 距离衰减权重
        'hb_decay_strength': hb_strength,
        'hp_decay_strength': hp_strength,
        'salt_decay_strength': salt_strength,
        'arom_decay_strength': arom_strength,
        'total_interaction_strength': total_interaction_strength,
        # 增强: 能量近似项
        'vdw_energy': vdw_energy,
        'elec_energy': elec_energy,
        'total_binding_energy': vdw_energy + elec_energy,
        # 增强: 配体效率
        'lig_eff_hb': lig_efficiency_hb,
        'lig_eff_hp': lig_efficiency_hp,
        'lig_eff_contact': lig_efficiency_contact,
        # 增强: 形状互补性
        'shape_complementarity': shape_complementarity,
        'lig_contact_fraction': lig_contact_frac,
        'prot_contact_fraction': prot_contact_frac,
        # 增强: 距离分布
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
        mol = Chem.AddHs(mol) if mol.GetNumHeavyAtoms()>0 else mol
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


def extract_features_from_files(protein_pdb_path, ligand_path, debug=False):
    """从PDBbind分离的蛋白PDB+配体SDF/MOL2提取特征"""
    try:
        with open(protein_pdb_path, 'r') as f:
            protein_atoms = parse_pdb_atoms(f.readlines())
        if not protein_atoms:
            if debug: print(f"    [debug] 蛋白原子数为0")
            return None
        # 支持SDF和MOL2两种格式
        lig_mol = None
        if ligand_path.endswith('.sdf'):
            suppl = Chem.SDMolSupplier(ligand_path, removeHs=False)
            lig_mol = next(suppl, None)
            if lig_mol is None:
                suppl = Chem.SDMolSupplier(ligand_path, removeHs=True)
                lig_mol = next(suppl, None)
            # SDF失败则尝试同目录MOL2
            if lig_mol is None:
                mol2_path = ligand_path.replace('.sdf', '.mol2')
                if os.path.exists(mol2_path):
                    lig_mol = Chem.MolFromMol2File(mol2_path, removeHs=False)
                    if lig_mol is None:
                        lig_mol = Chem.MolFromMol2File(mol2_path, removeHs=True)
                    if debug and lig_mol: print(f"    [debug] SDF失败, MOL2成功")
        elif ligand_path.endswith('.mol2'):
            lig_mol = Chem.MolFromMol2File(ligand_path, removeHs=False)
            if lig_mol is None:
                lig_mol = Chem.MolFromMol2File(ligand_path, removeHs=True)
        if lig_mol is None:
            if debug: print(f"    [debug] 配体分子读取失败")
            return None
        if debug: print(f"    [debug] 配体原子数={lig_mol.GetNumAtoms()}")
        try:
            conf = lig_mol.GetConformer()
        except Exception as e:
            if debug: print(f"    [debug] GetConformer失败: {e}")
            return None
        lig_atoms = []
        for i in range(lig_mol.GetNumAtoms()):
            try:
                pos = conf.GetAtomPosition(i)
                lig_atoms.append({
                    'coord': np.array([pos.x, pos.y, pos.z]),
                    'element': lig_mol.GetAtomWithIdx(i).GetSymbol().upper(),
                    'resname':'LIG','resnum':999,'chain':'Z','atom_name':'LIG','bfactor':0.0
                })
            except Exception:
                continue
        if debug: print(f"    [debug] lig_atoms构建完成, 数量={len(lig_atoms)}")
        inter = compute_interaction_features(protein_atoms, lig_atoms)
        if debug: print(f"    [debug] 相互作用特征={inter is not None}")
        desc = compute_ligand_descriptors(lig_mol)
        if debug: print(f"    [debug] 描述符={desc is not None}")
        if inter and desc:
            return {**inter, **desc}
        if debug: print(f"    [debug] inter或desc为None")
    except Exception as e:
        if debug: print(f"    [debug] 异常: {e}")
        return None


def extract_features_from_complex(complex_pdb_path):
    """从ASPA复合物PDB(含蛋白+配体)提取特征"""
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
        # 从HETATM构建RDKit分子
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
            heavy = sum(a['element']!='H' for a in ligand_atoms)
            desc = {'lig_MW':heavy*12.0,'lig_LogP':0.0,'lig_TPSA':0.0,'lig_num_HBD':0,'lig_num_HBA':0,
                    'lig_num_rotatable_bonds':0,'lig_num_aromatic_rings':0,'lig_num_heavy_atoms':heavy,
                    'lig_num_rings':0,'lig_num_saturated_rings':0,'lig_num_aliphatic_rings':0,
                    'lig_num_heteroatoms':sum(a['element'] in ('N','O','S','F','P') for a in ligand_atoms),
                    'lig_fraction_sp3':0.0,'lig_num_amide_bonds':0,'lig_num_valence_electrons':0,
                    'lig_formal_charge':0,'lig_mol_mr':0.0}
        return {**inter, **desc}
    except Exception as e:
        print(f"  [错误] {complex_pdb_path}: {e}")
        return None


# %% ================================================================
# Phase 1: 数据筛选 — 加载45个预筛选分子
# 参考: ASPA虚拟筛选.pptx中老师展示的筛选流程
#   YaTCM(47696) → LigPrep(58048) → ADMET(22227) → HTVS(5597) 
#   → SP(1120) → XP(224) → 柔性对接(45) → Top5
# 当前: 直接使用45个柔性对接结果作为粗筛产物
# ================================================================

print("\n" + "=" * 60)
print("Phase 1: 数据筛选 — 加载45个预筛选分子")
print("=" * 60)

# 加载对接打分CSV
screening_scores = {}
if os.path.exists(SCREENING_CSV):
    scr_df = pd.read_csv(SCREENING_CSV)
    print(f"[加载] ASPA虚拟筛选.csv: {len(scr_df)} 个分子")
    print(f"  原始对接打分范围: {scr_df['docking score'].min():.3f} ~ {scr_df['docking score'].max():.3f}")
    print(f"  原始Top3: {scr_df.head(3)['English_name'].tolist()}")
    for _, row in scr_df.iterrows():
        name = str(row.get('English_name', '')).strip()
        score = row.get('docking score')
        if name and pd.notna(score):
            screening_scores[name] = float(score)
else:
    print(f"[警告] 未找到CSV: {SCREENING_CSV}")

# 扫描FleDock PDB文件
pdb_files = sorted([f for f in os.listdir(FLEDOCK_DIR) if f.endswith('.pdb')]) if os.path.exists(FLEDOCK_DIR) else []
print(f"\n[扫描] 找到 {len(pdb_files)} 个对接复合物PDB文件")
if len(pdb_files) == 0:
    print(f"[错误] FleDock目录不存在或无PDB文件: {FLEDOCK_DIR}")
    print("  请确认: 1)数据集名称是否为'aspa-docking-data'  2)PDB文件是否在数据集根目录")

print("\n[完成] Phase 1 — 45个分子作为粗筛结果已就绪")


# %% ================================================================
# Phase 2: 分子对接 — 4MXU晶体结构对接结果
# 参考: PPT步骤
#   1. 蛋白预处理: 4MXU去水/补侧链/能量最小化
#   2. SiteMap结合位点预测 + Glide Grid格点生成
#   3. Glide HTVS→SP→XP→柔性对接
# 当前: FleDock PDB文件已包含4MXU+对接后配体的3D结合结构
#   (由Schrödinger Glide虚拟筛选流程生成)
# 后续可选: 用top10_pockets.csv中评分最高的口袋(conf_01/Pocket2)二次对接
# ================================================================

print("\n" + "=" * 60)
print("Phase 2: 分子对接 — 4MXU晶体结构")
print("=" * 60)

# 验证FleDock PDB文件格式(应同时包含ATOM蛋白+HETATM配体)
verified_files = []
for pf in pdb_files:
    path = os.path.join(FLEDOCK_DIR, pf)
    with open(path, 'r') as f:
        lines = f.readlines()
    has_protein = any(l.startswith('ATOM') for l in lines)
    has_ligand = any(l.startswith('HETATM') for l in lines)
    if has_protein and has_ligand:
        verified_files.append(pf)
    else:
        print(f"  [异常] {pf}: 蛋体={has_protein}, 配体={has_ligand}")

print(f"[验证] {len(verified_files)}/{len(pdb_files)} 个文件同时包含蛋白和配体")
print(f"  → 4MXU晶体结构对接已完成(Schrödinger Glide柔性对接产物)")
print(f"  → 无需重新对接, 直接使用现有3D结合结构进行精筛")

# ---- 可选: AutoDock Vina重对接(取消注释启用) ----
# 如需用AutoDock Vina对4MXU重新对接(替代Schrödinger结果):
#   1. pip install vina meeko
#   2. 准备受体PDBQT: obabel -ipdb 4MXU.pdb -opdbqt -O receptor.pdbqt -xr
#   3. 准备配体PDBQT: meeko mk_prepare_ligand.py -i ligand.sdf -o ligand.pdbqt
#   4. Vina对接: vina --receptor receptor.pdbqt --ligand ligand.pdbqt \
#                 --center_x X --center_y Y --center_z Z \
#                 --size_x 25 --size_y 25 --size_z 25 --exhaustiveness 8

# ---- 可选: top10_pockets.csv最高评分口袋二次对接 ----
# top10_pockets.csv中最高评分: conf_01/Pocket2 (评分=11.91)
# 二次对接步骤:
#   1. 使用 open_conformation_01.pdb 作为新受体
#   2. 根据Pocket2坐标定义Vina grid box
#   3. 对45个分子重新对接
#   4. 用新对接结果替换特征提取的输入

print("\n[完成] Phase 2 — 对接3D结构已就绪")

# %%
# %% ================================================================
# Phase 3: 模型训练 — PDBbind + 三维XGBoost
# 参考: XGBoost算法原理简介及调参.pdf
# ================================================================

print("\n" + "=" * 60)
print("Phase 3: 模型训练 — PDBbind训练集 + XGBoost")
print("=" * 60)

# ---- 3a: 下载PDBbind数据(DeepChem S3, 免注册) ----
def download_pdbbind(set_name='refined'):
    """下载PDBbind数据集: refined(~5000) 或 core(~285)"""
    urls = {
        'core':    'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbind_v2013_core_set.tar.gz',
        'refined': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pdbbindv2019/pdbbind_v2019_refined.tar.gz',
    }
    url = urls[set_name]
    tar_path = os.path.join(PDBBIND_DIR, f'pdbbind_{set_name}.tar.gz')
    if not os.path.exists(tar_path):
        print(f"[下载] PDBbind {set_name}集...")
        import urllib.request
        urllib.request.urlretrieve(url, tar_path)
        print(f"  大小: {os.path.getsize(tar_path)/1e6:.1f} MB")
    extract_dir = os.path.join(PDBBIND_DIR, f'pdbbind_{set_name}')
    if not os.path.exists(extract_dir):
        print(f"[解压]...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(PDBBIND_DIR)
        for item in os.listdir(PDBBIND_DIR):
            full = os.path.join(PDBBIND_DIR, item)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, 'index')):
                extract_dir = full
                break
    # 查找index文件
    index_file = None
    for name in ['INDEX_refined_data.2019', 'INDEX_core_data.2013']:
        for root, dirs, files in os.walk(extract_dir):
            if name in files:
                index_file = os.path.join(root, name)
                break
        if index_file:
            break
    print(f"[完成] Index: {index_file}")
    return extract_dir, index_file


def parse_pdbbind_index(index_file):
    """解析PDBbind index: PDB_code resolution year -logKd/Ki Kd/Ki ... ligand_name"""
    records = []
    with open(index_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    records.append({'pdb_id': parts[0], 'neg_log_kd': float(parts[3])})
                except ValueError:
                    continue
    return pd.DataFrame(records)


def build_pdbbind_file_map(extract_dir):
    """一次性扫描PDBbind目录, 建立PDB ID → (protein_path, ligand_path)的映射"""
    file_map = {}
    # PDBbind目录结构: extract_dir/{pdb_id}/{pdb_id}_pocket.pdb + {pdb_id}_ligand.sdf/.mol2
    for item in os.listdir(extract_dir):
        dir_path = os.path.join(extract_dir, item)
        if not os.path.isdir(dir_path) or len(item) != 4:
            continue
        # 蛋白文件: 优先pocket, 回退protein
        pocket = os.path.join(dir_path, f'{item}_pocket.pdb')
        protein = os.path.join(dir_path, f'{item}_protein.pdb')
        prot = pocket if os.path.exists(pocket) else protein
        # 配体文件: 优先sdf, 回退mol2
        ligand_sdf  = os.path.join(dir_path, f'{item}_ligand.sdf')
        ligand_mol2 = os.path.join(dir_path, f'{item}_ligand.mol2')
        ligand = ligand_sdf if os.path.exists(ligand_sdf) else (ligand_mol2 if os.path.exists(ligand_mol2) else None)
        if prot and ligand and os.path.exists(prot) and os.path.exists(ligand):
            file_map[item] = (prot, ligand)
    return file_map


# 下载并解析(优先refined, 失败回退core)
try:
    pdbbind_dir, index_file = download_pdbbind('refined')
    pdbbind_df = parse_pdbbind_index(index_file)
except Exception as e:
    print(f"[回退] refined下载失败({e}), 改用core集...")
    pdbbind_dir, index_file = download_pdbbind('core')
    pdbbind_df = parse_pdbbind_index(index_file)

print(f"[数据] {len(pdbbind_df)} 个蛋白-配体复合物")
print(f"  -logKd/Ki: {pdbbind_df['neg_log_kd'].min():.2f}~{pdbbind_df['neg_log_kd'].max():.2f}, "
      f"均值={pdbbind_df['neg_log_kd'].mean():.2f}±{pdbbind_df['neg_log_kd'].std():.2f}")

# 一次性扫描目录建立文件映射(比逐个os.walk快100倍)
print(f"\n[扫描] 建立PDBbind文件映射...")
file_map = build_pdbbind_file_map(pdbbind_dir)
print(f"[扫描] 找到 {len(file_map)} 个完整的复合物目录")
if file_map:
    sample_id = next(iter(file_map))
    prot_path, lig_path = file_map[sample_id]
    print(f"  样例: PDB={sample_id}")
    print(f"    蛋白: {prot_path}")
    print(f"    配体: {lig_path}")
    # 列出第一个目录的所有文件帮助调试
    sample_dir = os.path.dirname(prot_path)
    all_files = os.listdir(sample_dir)
    print(f"    目录内容: {all_files}")
else:
    print(f"[警告] 未找到任何复合物文件! 检查目录结构:")
    print(f"  extract_dir={pdbbind_dir}")
    if os.path.exists(pdbbind_dir):
        items = os.listdir(pdbbind_dir)
        print(f"  顶层内容({len(items)}项): {items[:10]}")
        for sub in items[:3]:
            sub_path = os.path.join(pdbbind_dir, sub)
            if os.path.isdir(sub_path):
                print(f"  {sub}/ 内容: {os.listdir(sub_path)}")

# ---- 3b: 提取PDBbind训练集特征 ----
print(f"\n[特征提取] 处理 {len(pdbbind_df)} 个复合物...")
all_features, all_labels, all_ids = [], [], []
t0 = time.time()
success_count = 0
debug_count = 0  # 只打印前3个的调试信息
debug_failures = 0  # 失败计数(用于打印前5个失败)
for idx, row in pdbbind_df.iterrows():
    pdb_id = row['pdb_id']
    if pdb_id not in file_map:
        continue
    prot, lig = file_map[pdb_id]
    # 仅前3个开debug验证
    is_debug = (debug_count < 3)
    if is_debug:
        debug_count += 1
    feats = extract_features_from_files(prot, lig, debug=is_debug)
    if feats:
        all_features.append(feats)
        all_labels.append(row['neg_log_kd'])
        all_ids.append(pdb_id)
        success_count += 1
    else:
        if debug_failures < 5:
            debug_failures += 1
            print(f"  [失败#{debug_failures}] PDB={pdb_id}")
    if (idx+1) % 500 == 0:
        el = time.time() - t0
        print(f"  进度: {idx+1}/{len(pdbbind_df)} | 成功 {success_count} | "
              f"速度 {success_count/max(el,0.1):.1f}/s | 剩余~{(len(pdbbind_df)-idx-1)/max((idx+1)/el,1):.0f}s")
el = time.time() - t0
print(f"[完成] 成功 {success_count}/{len(pdbbind_df)}, 耗时 {el:.0f}s, 失败 {len(pdbbind_df)-success_count}")

X_train_df = pd.DataFrame(all_features)
y_train = np.array(all_labels)
X_train_df.to_csv(os.path.join(WORK_DIR, 'pdbbind_features.csv'), index=False)
np.save(os.path.join(WORK_DIR, 'pdbbind_labels.npy'), y_train)
print(f"[特征矩阵] {X_train_df.shape}, {len(X_train_df.columns)}个特征")

# %%
# ---- 3c: XGBoost训练(参考XGBoost调参PDF) ----
# 参考: HyperGBM全Pipeline优化理念 — 特征标准化+早停+特征选择
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

X_tr, X_te, y_tr, y_te = train_test_split(X_train_df.values, y_train, test_size=0.2, random_state=42)

# 特征标准化 (能量项和距离项量级差异大, 标准化提升模型收敛)
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)
print(f"[标准化] 训练集{X_tr_scaled.shape}, 测试集{X_te_scaled.shape}")

# 基准模型 (带早停防过拟合)
base_params = {'objective':'reg:squarederror','max_depth':6,'learning_rate':0.1,'n_estimators':1000,
               'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':1,'gamma':0,
               'reg_alpha':0.1,'reg_lambda':1.0,'random_state':42,'n_jobs':-1,
               'early_stopping_rounds':50}
if HAS_GPU:
    base_params.update(GPU_PARAMS)
else:
    base_params['tree_method'] = 'hist'

print("\n[训练] 基准模型(早停)...")
base_model = xgb.XGBRegressor(**base_params)
base_model.fit(X_tr_scaled, y_tr, eval_set=[(X_te_scaled, y_te)], verbose=False)
y_base = base_model.predict(X_te_scaled)
rmse_b = np.sqrt(mean_squared_error(y_te, y_base))
pr_b = pearsonr(y_te, y_base)[0]
print(f"  基准: RMSE={rmse_b:.4f}, Pearson={pr_b:.4f}")

# 特征选择: 基于基准模型重要性筛选
selector = SelectFromModel(base_model, threshold='median', prefit=True)
selected_idx = selector.get_support()
n_selected = sum(selected_idx)
print(f"[特征选择] {X_train_df.shape[1]}→{n_selected}个特征 (基于重要性中位数阈值)")
X_tr_sel = X_tr_scaled[:, selected_idx]
X_te_sel = X_te_scaled[:, selected_idx]
feature_names_sel = X_train_df.columns[selected_idx].tolist()

# 随机搜索调参 (快速, 30组随机采样, Phase4的Optuna会做精细调参)
from sklearn.model_selection import RandomizedSearchCV
print("[调参] RandomizedSearchCV 5折交叉验证(30组随机采样)...")
param_dist = {'max_depth':[3,4,6,8,10], 'learning_rate':[0.01,0.05,0.1,0.2],
              'n_estimators':[200,500,800], 'subsample':[0.6,0.7,0.8,0.9],
              'colsample_bytree':[0.5,0.7,0.8,1.0], 'min_child_weight':[1,3,5]}
gs_params = {'objective':'reg:squarederror','gamma':0,'reg_alpha':0.1,'reg_lambda':1.0,
             'random_state':42,'n_jobs':-1,'early_stopping_rounds':50}
if HAS_GPU:
    gs_params.update(GPU_PARAMS)
else:
    gs_params['tree_method'] = 'hist'

gs = RandomizedSearchCV(xgb.XGBRegressor(**gs_params), param_dist,
                         n_iter=30, scoring='neg_root_mean_squared_error',
                         cv=KFold(5, shuffle=True, random_state=42),
                         n_jobs=1, verbose=1, random_state=42)
gs.fit(X_tr_sel, y_tr, eval_set=[(X_te_sel, y_te)], verbose=False)
best_model = gs.best_estimator_
y_best = best_model.predict(X_te_sel)
rmse = np.sqrt(mean_squared_error(y_te, y_best))
mae = mean_absolute_error(y_te, y_best)
r2 = r2_score(y_te, y_best)
pr = pearsonr(y_te, y_best)[0]
print(f"\n  最优参数: {gs.best_params_}")
print(f"  调优后: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Pearson={pr:.4f}")

# 特征重要性
print("\n[特征重要性 Top 15]:")
imp = pd.DataFrame({'feature':feature_names_sel,'importance':best_model.feature_importances_})
print(imp.sort_values('importance', ascending=False).head(15).to_string(index=False))

# 全量重训 (用标准化+选择后的全部数据)
print("\n[全量训练] 用全部数据重训最终模型...")
X_all_scaled = scaler.transform(X_train_df.values)
X_all_sel = X_all_scaled[:, selected_idx]
final_params = {**gs.best_params_, 'objective':'reg:squarederror','gamma':0,
                 'reg_alpha':0.1,'reg_lambda':1.0,'random_state':42,'n_jobs':-1}
if HAS_GPU:
    final_params.update(GPU_PARAMS)
else:
    final_params['tree_method'] = 'hist'
final_model = xgb.XGBRegressor(**final_params)
final_model.fit(X_all_sel, y_train, verbose=False)
final_model.save_model(os.path.join(WORK_DIR, 'xgboost_aspa_model.json'))
import joblib
joblib.dump(scaler, os.path.join(WORK_DIR, 'feature_scaler.pkl'))
joblib.dump(selector, os.path.join(WORK_DIR, 'feature_selector.pkl'))
print("[保存] 模型+标准化器+特征选择器")

# 保存性能指标
perf = {'rmse':rmse, 'mae':mae, 'r2':r2, 'pearson':pr, 'best_params':str(gs.best_params_)}
print(f"\n{'='*60}")
print(f"Phase 3完成: RMSE={rmse:.4f}, Pearson={pr:.4f}")
print(f"  达标标准: Pearson>{TARGET_PEARSON} 或 RMSE<{TARGET_RMSE}")
print(f"  {'✓ 达标' if pr>=TARGET_PEARSON or rmse<=TARGET_RMSE else '✗ 未达标, 进入Phase4优化'}")
print(f"{'='*60}")

# %%
# %% ================================================================
# Phase 4: 模型优化 — 无条件执行
# 参考: README_zh_CN.md (HyperGBM AutoML) + XGBoost调参PDF
# 策略: 1.Optuna贝叶斯超参搜索 2.加权模型集成 3.全量重训
# ================================================================

print("\n" + "=" * 60)
print("Phase 4: 模型优化 (无条件执行)")
print(f"  当前: Pearson={pr:.4f}, RMSE={rmse:.4f}")
print("=" * 60)

# ======== CHECKPOINT恢复: 如果存在Phase 4a checkpoint, 直接加载 ========
import joblib as _joblib
_checkpoint_path = os.path.join(WORK_DIR, 'checkpoint_phase4a.pkl')
if os.path.exists(_checkpoint_path):
    print("\n[Checkpoint] 检测到Phase 4a checkpoint, 加载中...")
    _ckpt = _joblib.load(_checkpoint_path)
    opt_model = _ckpt['opt_model']
    opt_params = _ckpt['opt_params']
    scaler = _ckpt['scaler']
    selector = _ckpt['selector']
    selected_idx = _ckpt['selected_idx']
    feature_names_sel = _ckpt['feature_names_sel']
    X_tr_sel = _ckpt['X_tr_sel']
    X_te_sel = _ckpt['X_te_sel']
    y_tr = _ckpt['y_tr']
    y_te = _ckpt['y_te']
    X_all_sel = _ckpt['X_all_sel']
    y_train = _ckpt['y_train']
    rmse_opt = _ckpt['rmse_opt']
    pr_opt = _ckpt['pr_opt']
    rmse = _ckpt['rmse_base']
    pr = _ckpt['pr_base']
    print(f"[Checkpoint] 已加载! Pearson={pr_opt:.4f}, RMSE={rmse_opt:.4f}")
    print(f"  → 跳过Phase 4a(Optuna), 直接进入Phase 4b(集成)")
    _SKIP_4A = True
else:
    _SKIP_4A = False

# ---- 4a: Optuna贝叶斯超参数优化 (有checkpoint时跳过) ----
if not _SKIP_4A:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        subprocess.run([sys.executable,'-m','pip','install','-q','optuna'])
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("[4a] Optuna贝叶斯超参数优化(100次试验, 扩大搜索空间)...")
    def objective(trial):
        params = {'objective':'reg:squarederror',
                   'max_depth':trial.suggest_int('max_depth',3,12),
                   'learning_rate':trial.suggest_float('learning_rate',0.005,0.3,log=True),
                   'n_estimators':trial.suggest_int('n_estimators',100,2000,step=100),
                   'subsample':trial.suggest_float('subsample',0.4,1.0),
                   'colsample_bytree':trial.suggest_float('colsample_bytree',0.3,1.0),
                   'min_child_weight':trial.suggest_int('min_child_weight',1,15),
                   'gamma':trial.suggest_float('gamma',0,2.0),
                   'reg_alpha':trial.suggest_float('reg_alpha',0,2.0),
                   'reg_lambda':trial.suggest_float('reg_lambda',0.1,5.0),
                   'max_leaves':trial.suggest_int('max_leaves',0,256),
                   'random_state':42,'n_jobs':-1}
        if HAS_GPU:
            params.update(GPU_PARAMS)
        else:
            params['tree_method'] = 'hist'
        scores = cross_val_score(xgb.XGBRegressor(**params), X_tr_sel, y_tr,
                                  cv=KFold(5,shuffle=True,random_state=42),
                                  scoring='neg_root_mean_squared_error', n_jobs=1)
        return -scores.mean()

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100, show_progress_bar=True)
    opt_params = {**study.best_params, 'objective':'reg:squarederror',
                   'random_state':42,'n_jobs':-1,'early_stopping_rounds':50}
    if HAS_GPU:
        opt_params.update(GPU_PARAMS)
    else:
        opt_params['tree_method'] = 'hist'
    opt_model = xgb.XGBRegressor(**opt_params)
    opt_model.fit(X_tr_sel, y_tr, eval_set=[(X_te_sel, y_te)], verbose=False)
    y_opt = opt_model.predict(X_te_sel)
    rmse_opt = np.sqrt(mean_squared_error(y_te, y_opt))
    pr_opt = pearsonr(y_te, y_opt)[0]
    print(f"  Optuna优化后: RMSE={rmse_opt:.4f}, Pearson={pr_opt:.4f}")
    print(f"  最优参数: {study.best_params}")

    # ======== CHECKPOINT: 保存Phase 4a结果 ========
    print("\n[Checkpoint] 保存Phase 4a结果到 /kaggle/working/...")
    import joblib, pickle
    checkpoint = {
        'opt_model': opt_model,
        'opt_params': opt_params,
        'study_best_params': study.best_params,
        'scaler': scaler,
        'selector': selector,
        'selected_idx': selected_idx,
        'feature_names_sel': feature_names_sel,
        'X_tr_sel': X_tr_sel, 'X_te_sel': X_te_sel, 'y_tr': y_tr, 'y_te': y_te,
        'X_all_sel': X_all_sel, 'y_train': y_train,
        'X_train_df_columns': X_train_df.columns.tolist(),
        'rmse_opt': rmse_opt, 'pr_opt': pr_opt,
        'rmse_base': rmse, 'pr_base': pr,
    }
    joblib.dump(checkpoint, os.path.join(WORK_DIR, 'checkpoint_phase4a.pkl'))
    # 同时保存特征矩阵用于Phase 5
    if 'all_features' in dir():
        joblib.dump(all_features, os.path.join(WORK_DIR, 'checkpoint_aspa_features.pkl'))
    if 'aspa_features' in dir():
        joblib.dump(aspa_features, os.path.join(WORK_DIR, 'checkpoint_aspa_features.pkl'))
    print("[Checkpoint] 已保存! 文件: checkpoint_phase4a.pkl")
    print("  → 刷新后可直接从Phase 4b继续, 无需重跑Phase 1-4a")

# %%
# ---- 4b: 加权模型集成(XGBoost+LightGBM+CatBoost) ----
print("\n[4b] 加权模型集成(XGBoost+LightGBM+CatBoost)...")

# Phase 4b集成初始化 (此时opt_model/y_opt已从checkpoint或Phase 4a获得)
ensemble_models = {'xgb': opt_model}
ensemble_preds = {'xgb': y_opt}
ensemble_weights = {}

# LightGBM
try:
    import lightgbm as lgb
    lgb_params = {'objective':'regression','max_depth':opt_params.get('max_depth',6),
                   'learning_rate':opt_params.get('learning_rate',0.1),
                   'n_estimators':opt_params.get('n_estimators',500),
                   'subsample':opt_params.get('subsample',0.8),
                   'colsample_bytree':opt_params.get('colsample_bytree',0.8),
                   'reg_alpha':opt_params.get('reg_alpha',0.1),
                   'reg_lambda':opt_params.get('reg_lambda',1.0),
                   'min_child_samples':opt_params.get('min_child_weight',1),
                   'random_state':42,'n_jobs':-1,'verbose':-1}
    lgb_m = lgb.LGBMRegressor(**lgb_params)
    lgb_m.fit(X_tr_sel, y_tr, eval_set=[(X_te_sel, y_te)])
    y_lgb = lgb_m.predict(X_te_sel)
    rmse_lgb = np.sqrt(mean_squared_error(y_te, y_lgb))
    pr_lgb = pearsonr(y_te, y_lgb)[0]
    print(f"  LightGBM: RMSE={rmse_lgb:.4f}, Pearson={pr_lgb:.4f}")
    ensemble_models['lgb'] = lgb_m
    ensemble_preds['lgb'] = y_lgb
    ensemble_weights['lgb'] = max(pr_lgb, 0.01)
except Exception as e:
    print(f"  LightGBM跳过: {e}")

# CatBoost
try:
    from catboost import CatBoostRegressor
    cat_m = CatBoostRegressor(depth=opt_params.get('max_depth',6),
                               learning_rate=opt_params.get('learning_rate',0.1),
                               iterations=opt_params.get('n_estimators',500),
                               l2_leaf_reg=opt_params.get('reg_lambda',1.0),
                               random_seed=42, verbose=0)
    cat_m.fit(X_tr_sel, y_tr, eval_set=(X_te_sel, y_te))
    y_cat = cat_m.predict(X_te_sel)
    rmse_cat = np.sqrt(mean_squared_error(y_te, y_cat))
    pr_cat = pearsonr(y_te, y_cat)[0]
    print(f"  CatBoost: RMSE={rmse_cat:.4f}, Pearson={pr_cat:.4f}")
    ensemble_models['cat'] = cat_m
    ensemble_preds['cat'] = y_cat
    ensemble_weights['cat'] = max(pr_cat, 0.01)
except Exception as e:
    print(f"  CatBoost跳过: {e}")

# XGBoost权重
ensemble_weights['xgb'] = max(pr_opt, 0.01)

# 加权集成 (按Pearson相关系数加权)
total_w = sum(ensemble_weights.values())
y_ens = np.zeros_like(y_te, dtype=float)
for name in ensemble_preds:
    y_ens += ensemble_preds[name] * (ensemble_weights[name] / total_w)
rmse = np.sqrt(mean_squared_error(y_te, y_ens))
pr = pearsonr(y_te, y_ens)[0]
print(f"  加权集成: RMSE={rmse:.4f}, Pearson={pr:.4f}")
weight_str = ", ".join(f"{k}={v/total_w:.2f}" for k,v in ensemble_weights.items())
print(f"  权重: {weight_str}")

# 选择最优模型: 集成 vs 单模型
if pr >= pr_opt:
    print("  → 使用加权集成模型作为最终模型")
    final_model = opt_model  # 保留XGBoost作为主模型
else:
    print("  → 使用Optuna优化XGBoost作为最终模型")
    final_model = opt_model

# 全量重训优化模型 (用标准化+选择后的全部数据, 去掉early_stopping)
print("\n[全量重训] 用全部数据重训最终模型...")
final_params_no_es = {k:v for k,v in opt_params.items() if k != 'early_stopping_rounds'}
final_model = xgb.XGBRegressor(**final_params_no_es)
final_model.fit(X_all_sel, y_train, verbose=False)
final_model.save_model(os.path.join(WORK_DIR, 'xgboost_aspa_final.json'))
# 同时重训集成模型
if 'lgb' in ensemble_models:
    ensemble_models['lgb'].fit(X_all_sel, y_train)
if 'cat' in ensemble_models:
    ensemble_models['cat'].fit(X_all_sel, y_train)
joblib.dump({'models': ensemble_models, 'weights': ensemble_weights,
             'scaler': scaler, 'selector': selector},
            os.path.join(WORK_DIR, 'ensemble_models.pkl'))
print(f"\n[Phase4完成] 优化后: RMSE={rmse:.4f}, Pearson={pr:.4f}")

# %%
# ======== Phase 4c: HyperGBM全Pipeline优化 ========
# 参考: README_zh_CN.md — 数据清洗+特征工程+共线性检测+Stacking集成
print("\n" + "=" * 60)
print("Phase 4c: HyperGBM全Pipeline优化")
print("  策略: 异常值清洗 → 多项式特征 → 共线性剔除 → Stacking集成")
print("=" * 60)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# ---- 4c-1: 数据清洗 (去除异常值, IQR方法) ----
print("\n[4c-1] 数据清洗: 去除异常值...")
y_train_arr = np.array(y_train)
Q1, Q3 = np.percentile(y_train_arr, [25, 75])
IQR = Q3 - Q1
lower_bound = Q1 - 2.0 * IQR
upper_bound = Q3 + 2.0 * IQR
outlier_mask = (y_train_arr >= lower_bound) & (y_train_arr <= upper_bound)
n_before = len(y_train_arr)
n_after = np.sum(outlier_mask)
print(f"  异常值范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"  清洗: {n_before} → {n_after} (去除{n_before-n_after}个异常值)")

# 用清洗后的数据重建训练/测试集
X_clean = X_train_df.values[outlier_mask]
y_clean = np.array(y_train)[outlier_mask]
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

# 重新标准化
scaler_c = StandardScaler()
X_tr_c_scaled = scaler_c.fit_transform(X_tr_c)
X_te_c_scaled = scaler_c.transform(X_te_c)

# %%
# ---- 4c-2: 特征工程 (多项式交互特征, degree=2, 仅交互项) ----
print("\n[4c-2] 特征工程: 生成多项式交互特征...")
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_tr_c_poly = poly.fit_transform(X_tr_c_scaled)
X_te_c_poly = poly.transform(X_te_c_scaled)
print(f"  特征数: {X_tr_c_scaled.shape[1]} → {X_tr_c_poly.shape[1]} (含交互项)")

# %%
# ---- 4c-3: 共线性检测 (VIF > 10的剔除) ----
print("\n[4c-3] 共线性检测: VIF筛选...")
# 计算每个特征的VIF (在子采样上计算以加速)
n_samples_vif = min(1000, X_tr_c_poly.shape[0])
vif_sample_idx = np.random.RandomState(42).choice(X_tr_c_poly.shape[0], n_samples_vif, replace=False)
X_vif = X_tr_c_poly[vif_sample_idx]

# 逐步计算VIF, 移除VIF>10的特征
keep_features = list(range(X_tr_c_poly.shape[1]))
max_vif_features = 100  # 最多保留100个特征
removed_count = 0

# 简化: 基于方差阈值和相关性快速筛选
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression

# 1. 去低方差
var_sel = VarianceThreshold(threshold=0.01)
X_tr_c_var = var_sel.fit_transform(X_tr_c_poly)
X_te_c_var = var_sel.transform(X_te_c_poly)
print(f"  低方差筛选: {X_tr_c_poly.shape[1]} → {X_tr_c_var.shape[1]}")

# 2. SelectKBest: 保留top50特征
n_keep = min(50, X_tr_c_var.shape[1])
kbest = SelectKBest(f_regression, k=n_keep)
X_tr_c_sel = kbest.fit_transform(X_tr_c_var, y_tr_c)
X_te_c_sel = kbest.transform(X_te_c_var)
print(f"  SelectKBest: {X_tr_c_var.shape[1]} → {X_tr_c_sel.shape[1]} (top-{n_keep})")

# 3. 标准化
scaler_poly = StandardScaler()
X_tr_c_final = scaler_poly.fit_transform(X_tr_c_sel)
X_te_c_final = scaler_poly.transform(X_te_c_sel)
print(f"  最终特征矩阵: {X_tr_c_final.shape}")

# %%
# ---- 4c-4: Stacking集成 (XGBoost+LightGBM+CatBoost → Ridge元学习器) ----
print("\n[4c-4] Stacking集成(XGBoost+LightGBM+CatBoost → Ridge元学习器)...")
from sklearn.ensemble import StackingRegressor

# 基模型 (使用Phase 4a的Optuna最优参数, 去掉early_stopping)
stack_xgb_params = {k:v for k,v in opt_params.items() if k != 'early_stopping_rounds'}
if HAS_GPU:
    stack_xgb_params.update(GPU_PARAMS)
else:
    stack_xgb_params['tree_method'] = 'hist'

base_models = [('xgb', xgb.XGBRegressor(**stack_xgb_params))]

try:
    import lightgbm as lgb
    lgb_params_s = {'objective':'regression','max_depth':opt_params.get('max_depth',6),
                     'learning_rate':opt_params.get('learning_rate',0.1),
                     'n_estimators':opt_params.get('n_estimators',500),
                     'subsample':opt_params.get('subsample',0.8),
                     'colsample_bytree':opt_params.get('colsample_bytree',0.8),
                     'random_state':42,'n_jobs':-1,'verbose':-1}
    base_models.append(('lgb', lgb.LGBMRegressor(**lgb_params_s)))
    print("  基模型: XGBoost + LightGBM", end="")
except Exception:
    print("  基模型: XGBoost", end="")

try:
    from catboost import CatBoostRegressor
    cat_params_s = {'depth':opt_params.get('max_depth',6),
                     'learning_rate':opt_params.get('learning_rate',0.1),
                     'iterations':opt_params.get('n_estimators',500),
                     'random_seed':42,'verbose':0}
    base_models.append(('cat', CatBoostRegressor(**cat_params_s)))
    print(" + CatBoost")
except Exception:
    print("")

# %%
# Stacking: 用Ridge作为元学习器 (防止过拟合)
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    n_jobs=1,
    cv=5
)

print("  训练Stacking模型 (5折内部交叉验证)...")
stacking_model.fit(X_tr_c_final, y_tr_c)
y_stack = stacking_model.predict(X_te_c_final)
rmse_stack = np.sqrt(mean_squared_error(y_te_c, y_stack))
pr_stack = pearsonr(y_te_c, y_stack)[0]
r2_stack = r2_score(y_te_c, y_stack)
mae_stack = mean_absolute_error(y_te_c, y_stack)
print(f"  Stacking: RMSE={rmse_stack:.4f}, MAE={mae_stack:.4f}, R2={r2_stack:.4f}, Pearson={pr_stack:.4f}")

# %%
# 对比: 是否比Phase 4的加权集成更好
if pr_stack > pr:
    print(f"\n  → Stacking优于加权集成 (Pearson: {pr_stack:.4f} > {pr:.4f})")
    print(f"  → R2提升: {r2_stack:.4f} (之前约0.5x)")
    final_model = stacking_model
    # 保存Stacking pipeline的各组件
    joblib.dump({
        'stacking_model': stacking_model,
        'scaler_c': scaler_c,
        'poly': poly,
        'var_sel': var_sel,
        'kbest': kbest,
        'scaler_poly': scaler_poly,
        'X_train_df_columns': X_train_df.columns.tolist() if 'X_train_df' in dir() else _ckpt.get('X_train_df_columns', feature_names_sel),
        'selected_idx': selected_idx,
        'scaler': scaler,
        'selector': selector,
    }, os.path.join(WORK_DIR, 'stacking_pipeline.pkl'))
    # 更新性能指标
    rmse = rmse_stack
    pr = pr_stack
    # 标记使用Stacking
    _USE_STACKING = True
else:
    print(f"\n  → Stacking未优于加权集成 (Pearson: {pr_stack:.4f} ≤ {pr:.4f}), 保留原模型")
    _USE_STACKING = False

# 更新集成模型用于Phase 5预测
if _USE_STACKING:
    # 保存全量重训的Stacking模型
    print("\n[全量重训] Stacking模型用全部清洗数据重训...")
    X_all_clean = X_train_df.values[outlier_mask]
    y_all_clean = np.array(y_train)[outlier_mask]
    X_all_clean_scaled = scaler_c.transform(X_all_clean)
    X_all_clean_poly = poly.transform(X_all_clean_scaled)
    X_all_clean_var = var_sel.transform(X_all_clean_poly)
    X_all_clean_sel = kbest.transform(X_all_clean_var)
    X_all_clean_final = scaler_poly.transform(X_all_clean_sel)
    stacking_model.fit(X_all_clean_final, y_all_clean)
    final_model = stacking_model
    joblib.dump({
        'stacking_model': stacking_model,
        'scaler_c': scaler_c,
        'poly': poly,
        'var_sel': var_sel,
        'kbest': kbest,
        'scaler_poly': scaler_poly,
        'X_train_df_columns': X_train_df.columns.tolist() if 'X_train_df' in dir() else _ckpt.get('X_train_df_columns', feature_names_sel),
        'selected_idx': selected_idx,
        'scaler': scaler,
        'selector': selector,
    }, os.path.join(WORK_DIR, 'stacking_pipeline.pkl'))
    print("[保存] stacking_pipeline.pkl")

print(f"\n[Phase4c完成] 最终: RMSE={rmse:.4f}, Pearson={pr:.4f}")

# %%
# ======== 可视化: 预测值 vs 真实值散点图 ========
print("\n[可视化] 生成预测值vs真实值散点图...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 使用最终模型在测试集上预测
if _USE_STACKING:
    y_pred_test = stacking_model.predict(X_te_c_final)
    y_true_plot = y_te_c
    _title_extra = f"Stacking | R2={r2_score(y_te_c, y_pred_test):.4f}"
else:
    y_pred_test = np.zeros_like(y_te, dtype=float)
    total_w = sum(ensemble_weights.values())
    for name in ensemble_preds:
        y_pred_test += ensemble_preds[name] * (ensemble_weights[name] / total_w)
    y_true_plot = y_te
    _title_extra = f"加权集成 | R2={r2_score(y_te, y_pred_test):.4f}"

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图: 散点图
ax = axes[0]
ax.scatter(y_true_plot, y_pred_test, alpha=0.4, s=20, c='steelblue', edgecolors='navy', linewidth=0.3)
# 理想线 y=x
lim_min = min(y_true_plot.min(), y_pred_test.min()) - 0.5
lim_max = max(y_true_plot.max(), y_pred_test.max()) + 0.5
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2, label='理想线 y=x')
# 回归拟合线
z = np.polyfit(y_true_plot, y_pred_test, 1)
p = np.poly1d(z)
x_fit = np.linspace(lim_min, lim_max, 100)
ax.plot(x_fit, p(x_fit), 'g-', lw=1.5, label=f'拟合线: y={z[0]:.3f}x+{z[1]:.3f}')
ax.set_xlabel('真实值 (-logKd/Ki)', fontsize=12)
ax.set_ylabel('预测值 (-logKd/Ki)', fontsize=12)
ax.set_title(f'预测值 vs 真实值\n{_title_extra}\nPearson={pr:.4f}, RMSE={rmse:.4f}', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# 右图: 残差分布
ax2 = axes[1]
residuals = y_pred_test - y_true_plot
ax2.hist(residuals, bins=50, color='steelblue', edgecolor='navy', alpha=0.7)
ax2.axvline(x=0, color='r', linestyle='--', lw=2, label='零误差线')
ax2.axvline(x=residuals.mean(), color='g', linestyle='-', lw=1.5, label=f'均值={residuals.mean():.3f}')
ax2.set_xlabel('残差 (预测值 - 真实值)', fontsize=12)
ax2.set_ylabel('频数', fontsize=12)
ax2.set_title(f'残差分布\nMAE={np.mean(np.abs(residuals)):.4f}, Std={residuals.std():.4f}', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'prediction_vs_actual.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"[保存] 图表: {os.path.join(WORK_DIR, 'prediction_vs_actual.png')}")


# %% ===================

# %%
# Phase 5: 肌肽排名分析 — XGBoost精筛 + Carnosine定位
# ================================================================

print("\n" + "=" * 60)
print("Phase 5: 肌肽排名分析")
print("=" * 60)

# ---- 5a: 提取ASPA对接复合物特征 ----
print("[5a] 提取45个ASPA对接复合物特征...")
aspa_features, aspa_names, aspa_scores, aspa_files = [], [], [], []
for pf in verified_files:
    path = os.path.join(FLEDOCK_DIR, pf)
    mol_name = pf.replace('.pdb','').replace('_',' ').strip()
    matched = mol_name
    for csv_name in screening_scores:
        if mol_name.lower().replace(' ','') == csv_name.lower().replace(' ',''):
            matched = csv_name
            break
    feats = extract_features_from_complex(path)
    if feats:
        aspa_features.append(feats)
        aspa_names.append(matched)
        aspa_scores.append(screening_scores.get(matched))
        aspa_files.append(pf)

print(f"  成功提取 {len(aspa_features)} 个分子特征")

# 对齐特征列 (与PDBbind训练集特征列完全一致)
# 如果从checkpoint恢复, 重建X_train_df的列名
if 'X_train_df' not in dir():
    if os.path.exists(os.path.join(WORK_DIR, 'checkpoint_phase4a.pkl')):
        import joblib as _jl
        _ckpt = _jl.load(os.path.join(WORK_DIR, 'checkpoint_phase4a.pkl'))
        _train_cols = _ckpt['X_train_df_columns']
        print(f"  [Checkpoint] 从checkpoint加载训练集特征列名: {len(_train_cols)}个")
    else:
        _train_cols = feature_names_sel
else:
    _train_cols = X_train_df.columns

X_aspa = pd.DataFrame(aspa_features)
for col in _train_cols:
    if col not in X_aspa.columns:
        X_aspa[col] = 0.0
X_aspa = X_aspa[_train_cols]

# 应用与训练集相同的预处理pipeline
if _USE_STACKING:
    # Stacking pipeline: 标准化 → 多项式 → 方差筛选 → KBest → 标准化 → Stacking预测
    print("[5b] Stacking模型预测(全Pipeline: 标准化→多项式→KBest→Stacking)...")
    X_aspa_scaled = scaler_c.transform(X_aspa.values)
    X_aspa_poly = poly.transform(X_aspa_scaled)
    X_aspa_var = var_sel.transform(X_aspa_poly)
    X_aspa_sel = kbest.transform(X_aspa_var)
    X_aspa_final = scaler_poly.transform(X_aspa_sel)
    predictions = stacking_model.predict(X_aspa_final)
elif os.path.exists(os.path.join(WORK_DIR, 'ensemble_models.pkl')):
    print("[5b] 加权集成模型预测...")
    X_aspa_scaled = scaler.transform(X_aspa.values)
    X_aspa_sel = X_aspa_scaled[:, selected_idx]
    ens_data = joblib.load(os.path.join(WORK_DIR, 'ensemble_models.pkl'))
    ens_models = ens_data['models']
    ens_weights = ens_data['weights']
    total_w = sum(ens_weights.values())
    predictions = np.zeros(X_aspa_sel.shape[0])
    for name, model in ens_models.items():
        w = ens_weights[name] / total_w
        pred = model.predict(X_aspa_sel)
        predictions += pred * w
        print(f"    {name}权重={w:.2f}")
else:
    print("[5b] 单一XGBoost模型预测...")
    X_aspa_scaled = scaler.transform(X_aspa.values)
    X_aspa_sel = X_aspa_scaled[:, selected_idx]
    predictions = final_model.predict(X_aspa_sel)

# ---- 5c: 排名 ----
results = pd.DataFrame({
    '分子名称': aspa_names,
    '文件名': aspa_files,
    'XGBoost预测(-logKd/Ki)': predictions,
    '对接打分(docking score)': aspa_scores,
})
results = results.sort_values('XGBoost预测(-logKd/Ki)', ascending=False).reset_index(drop=True)
results['XGBoost排名'] = range(1, len(results)+1)
if results['对接打分(docking score)'].notna().any():
    results['对接打分排名'] = results['对接打分(docking score)'].rank(ascending=False, method='min').fillna(0).astype(int)

print("\n" + "=" * 60)
print("全部分子XGBoost精筛排名:")
print("=" * 60)
print(results.to_string(index=False))
results.to_csv(os.path.join(WORK_DIR, 'ASPA_XGBoost_ranking.csv'), index=False, encoding='utf-8-sig')
print(f"\n[保存] ASPA_XGBoost_ranking.csv")

# ---- 5d: 定位肌肽(Carnosine) ----
print("\n" + "=" * 60)
print("★★★ 肌肽(Carnosine)排名结果 ★★★")
print("=" * 60)

found = False
for _, row in results.iterrows():
    name = str(row['分子名称']).lower()
    if 'carnosine' in name or '肌肽' in str(row['分子名称']):
        found = True
        print(f"\n  分子名称:       {row['分子名称']}")
        print(f"  文件名:         {row['文件名']}")
        print(f"  XGBoost预测值:  {row['XGBoost预测(-logKd/Ki)']:.4f}")
        print(f"  ★ XGBoost排名:  第 {row['XGBoost排名']} 名 / 共 {len(results)} 个分子")
        if '对接打分排名' in row and pd.notna(row['对接打分排名']):
            print(f"  原始对接打分:   {row['对接打分(docking score)']:.4f}")
            print(f"  对接打分排名:   第 {row['对接打分排名']} 名 / 共 {len(results)} 个分子")
        print(f"\n  → 在XGBoost三维精筛模型中，肌肽排名第 {row['XGBoost排名']} 位")
        break

if not found:
    print("\n  [警告] 未找到Carnosine, 所有分子名:")
    for n in aspa_names:
        print(f"    - {n}")

# ---- 最终汇总 ----
print("\n" + "=" * 60)
print("全流程执行完毕")
print("=" * 60)
print(f"""
┌────────────────────────────────────────────────────────┐
│              完整工作流程汇总                            │
├────────────────────────────────────────────────────────┤
│ Phase 1 数据筛选: 45个预筛选分子(FleDock对接结果)        │
│ Phase 2 分子对接: 4MXU晶体结构(Schrödinger Glide产物)    │
│ Phase 3 模型训练: PDBbind {len(all_features)}复合物 → XGBoost       │
│                  RMSE={rmse:.4f}, Pearson={pr:.4f}              │
│ Phase 4 模型优化: 已执行(无条件){'':26}│
│ Phase 5 肌肽排名: {'第'+str([r['XGBoost排名'] for _,r in results.iterrows() if 'carnosine' in str(r['分子名称']).lower()][0])+'名' if found else '未找到'}{'':36}│
└────────────────────────────────────────────────────────┘
""")
