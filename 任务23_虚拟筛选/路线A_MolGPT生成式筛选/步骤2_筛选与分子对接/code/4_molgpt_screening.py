#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASPA MolGPT支路虚拟筛选 — GNINA/Vina GPU自动选择版
=====================================================
MolGPT 5000 SMILES → 2D过滤 → QED预筛Top N → 3D构象 → GNINA/Vina对接
→ 复合物PDB → 55维特征 → XGBoost精排(复用方案B模型) → 排名 + 可视化

服务器目录约定:
  ~/ASPA_molgpt/
    data/                           输入数据 (Xftp手动上传)
      A3_MGPT_molgpt_only.csv       MolGPT 5000条SMILES (UTF-8 BOM)
      receptor_4MXU.pdb             ASPA 4MXU受体PDB
      ensemble_models.pkl (可选)    方案B已训XGBoost模型, 有则复用
    scripts/
      4_molgpt_screening.py         本脚本
    output_screening/               全部输出 (自动创建)

运行:
  cd ~/ASPA_molgpt
  python scripts/4_molgpt_screening.py
  # 可调对接数量:
  TOP_N_DOCK=1000 python scripts/4_molgpt_screening.py

对接引擎自动选择:
  1. GNINA (PATH中可找到, x86_64+CUDA) → GPU批量对接, 最快
  2. Vina (pip install vina) → CPU多线程, 可接受
  3. RDKit-empirical → 保底经验打分
"""

# ============================================================
# Phase 0: 环境检测
# ============================================================
import subprocess, sys, os, shutil, platform, math, time, tempfile, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')

print("=" * 60)
print("ASPA MolGPT支路虚拟筛选 | 对接引擎自动选择")
print("=" * 60)
print(f"[环境] Python {sys.version.split()[0]} | 架构 {platform.machine()}")

# ---- 对接引擎检测: gnina → vina → empirical ----
DOCK_ENGINE = None
GNINA_BIN = shutil.which('gnina')

if GNINA_BIN:
    DOCK_ENGINE = 'gnina'
    print(f"[引擎] 检测到 gnina: {GNINA_BIN}")
else:
    print("[引擎] 未检测到 gnina, 尝试 AutoDock Vina ...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vina'], timeout=600)
        from vina import Vina as _VinaProbe
        _probe = _VinaProbe(sf_name='vina', verbosity=0)
        DOCK_ENGINE = 'vina'
        print("[引擎] Vina 安装验证成功 → CPU多线程对接")
    except Exception as _e:
        print(f"[引擎] Vina 不可用 ({_e.__class__.__name__})")
        print("[引擎] 回退 RDKit-empirical 经验打分")
        DOCK_ENGINE = 'RDKit-empirical'

# ---- 依赖自检 ----
for _pkg in ['numpy', 'pandas', 'rdkit', 'xgboost', 'sklearn', 'scipy', 'matplotlib']:
    try:
        __import__(_pkg)
    except ImportError:
        print(f"[依赖] 缺少 {_pkg}, 尝试安装 ...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', _pkg])

# ---- 可视化后端 ----
import matplotlib
matplotlib.use('Agg')  # 无显示器环境用Agg
import matplotlib.pyplot as plt
HAS_PY3DMOL = False
try:
    import py3Dmol
    HAS_PY3DMOL = True
except ImportError:
    pass

# ---- RDKit ----
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ---- XGBoost + sklearn ----
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import joblib

# GPU检测 (XGBoost)
HAS_GPU = False
GPU_PARAMS = {}
try:
    xgb_ver = tuple(int(v) for v in xgb.__version__.split('.')[:2])
    if xgb_ver >= (2, 0):
        GPU_PARAMS = {'tree_method': 'hist', 'device': 'cuda'}
    else:
        GPU_PARAMS = {'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
    _tmp = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=np.array([1, 2]))
    xgb.train({**GPU_PARAMS, 'objective': 'reg:squarederror'}, _tmp, num_boost_round=1)
    HAS_GPU = True
    print(f"[环境] GPU可用 → XGBoost {xgb.__version__} {GPU_PARAMS}")
except Exception as e:
    HAS_GPU = False
    print(f"[环境] 仅CPU → XGBoost (GPU检测失败: {e})")

# ============================================================
# 配置
# ============================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 输出目录锚定: 脚本同级output_molgpt (不污染方案B的output_screening)
BASE_DIR = os.path.dirname(_SCRIPT_DIR)
WORK_DIR = os.path.join(BASE_DIR, 'output_molgpt')
PLOTS_DIR = os.path.join(WORK_DIR, 'plots')
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# 输入目录候选: 方案B的data/ 优先 (已有文件), 其次脚本同级data/
_CANDIDATE_DATA_DIRS = [
    os.path.join(os.path.expanduser('~/ASPA_planB'), 'data'),
    os.path.join(BASE_DIR, 'data'),
]
DATA_DIR = next((d for d in _CANDIDATE_DATA_DIRS if os.path.isdir(d)), _CANDIDATE_DATA_DIRS[0])

# 对接盒子: 4MXU催化Zn离子坐标 (Vina验证真实分-7.9 kcal/mol)
DOCK_CENTER = [-48.299, 14.804, -29.972]
DOCK_SIZE = (25, 25, 25)
VINA_EXHAUST = int(os.environ.get('VINA_EXHAUST', '8'))
VINA_CPU = int(os.environ.get('VINA_CPU', '8'))
VINA_TIMEOUT = int(os.environ.get('VINA_TIMEOUT', '300'))
TOP_N_DOCK = int(os.environ.get('TOP_N_DOCK', '500'))
# 预选策略: qed=纯QED TopN(旧, 偏向小而简单分子); strata=QED地板+MW分层(偏向300-450口袋填充区)
SELECT_MODE = os.environ.get('SELECT_MODE', 'qed')
# 排名策略: consensus=XGB+对接百分位融合(默认); xgb_only=仅按XGBoost预测排序
RANK_MODE = os.environ.get('RANK_MODE', 'consensus')
FINAL_TOP_N = int(os.environ.get('FINAL_TOP_N', '20'))   # 最终候选表输出数量

# 输入文件查找 (多目录回退)
def _find_file(label, candidates, search_dirs=None):
    if search_dirs is None:
        search_dirs = _CANDIDATE_DATA_DIRS
    tried = []
    for d in search_dirs:
        for fn in candidates:
            p = os.path.join(d, fn)
            tried.append(p)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"[缺输入] {label} 不存在。\n"
                            f"  搜索路径: {tried}")

MOLGPT_CSV = _find_file('MolGPT CSV', [
    'A3_MGPT_molgpt_only.csv',       # 5000条 (服务器已上传)
    'A3_MGPT_机器过滤.csv',
    'A3_MGPT_machine_filter.csv',
])
RECEPTOR_PDB = _find_file('受体PDB', [
    'receptor_4MXU.pdb',
    '4MXU_original.pdb',
    'ASPA-4MXU.pdb',
    '4MXU.pdb',
])

# 模型缓存查找: 已知路径 → 全盘glob搜索 ~/ASPA_planB 下的 ensemble_models.pkl
import glob as _glob
MODEL_CACHE = None
_model_search_dirs = [
    os.path.join(os.path.expanduser('~/ASPA_planB'), 'output_screening'),
    DATA_DIR,
    os.path.join(os.path.expanduser('~/ASPA_planB')),
    os.path.expanduser('~'),
]
for _d in _model_search_dirs:
    # 有限深度搜索, 避免递归扫描整个home目录过慢
    for _pattern in [os.path.join(_d, 'ensemble_models.pkl'),
                     os.path.join(_d, '*', 'ensemble_models.pkl'),
                     os.path.join(_d, '*', '*', 'ensemble_models.pkl'),
                     os.path.join(_d, 'output_screening', 'ensemble_models.pkl')]:
        _hits = sorted(set(_glob.glob(_pattern)))
        if _hits:
            MODEL_CACHE = _hits[0]
            break
    if MODEL_CACHE:
        break

print(f"\n路径配置:")
print(f"  工程根目录: {BASE_DIR}")
print(f"  数据目录:   {DATA_DIR}")
print(f"  输出目录:   {WORK_DIR}")
print(f"  MolGPT:     {MOLGPT_CSV}")
print(f"  受体PDB:    {RECEPTOR_PDB}")
print(f"  模型缓存:   {MODEL_CACHE if MODEL_CACHE else '未找到, 将仅用对接打分排名'}")
print(f"  对接引擎:   {DOCK_ENGINE}")
print(f"  对接数量:   {TOP_N_DOCK} (环境变量TOP_N_DOCK可调)")
print(f"  预选策略:   {SELECT_MODE} (环境变量SELECT_MODE: qed|strata)")
print(f"  排名策略:   {RANK_MODE} (环境变量RANK_MODE: xgb_only|consensus), 输出Top{FINAL_TOP_N}")
print(f"  对接中心:   {DOCK_CENTER}")
print(f"  盒子大小:   {DOCK_SIZE}")
print("=" * 60)


# ============================================================
# Phase 1: 特征提取函数 (移植自方案B, 保证口径一致)
# ============================================================
HYDROPHOBIC_RES = {'ALA', 'VAL', 'LEU', 'ILE', 'PHE', 'TRP', 'MET', 'PRO'}
POSITIVE_RES = {'ARG', 'LYS', 'HIP', 'HIS'}
NEGATIVE_RES = {'ASP', 'GLU'}
POLAR_RES = {'SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'}
AROMATIC_RES = {'PHE', 'TYR', 'TRP', 'HIS'}


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
    """计算蛋白-配体三维相互作用特征(38维)"""
    if not protein_atoms or not ligand_atoms:
        return None

    prot_coords = np.array([a['coord'] for a in protein_atoms])
    lig_coords = np.array([a['coord'] for a in ligand_atoms])
    prot_elem = [a['element'] for a in protein_atoms]
    lig_elem = [a['element'] for a in ligand_atoms]
    prot_res = [a['resname'] for a in protein_atoms]
    prot_resnum = [a['resnum'] for a in protein_atoms]
    prot_bf = [a['bfactor'] for a in protein_atoms]

    diff = lig_coords[:, np.newaxis, :] - prot_coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    dist_safe = np.maximum(dist, 0.5)

    min_dists = np.min(dist, axis=1)
    contact_mask = dist < 5.0
    contact_dists = dist[contact_mask]

    prot_hb = np.array([1 if e in ('N', 'O', 'S') else 0 for e in prot_elem])
    lig_hb = np.array([1 if e in ('N', 'O', 'S', 'F') else 0 for e in lig_elem])
    hb_mask = (dist < 3.5) & (prot_hb[np.newaxis, :] > 0) & (lig_hb[:, np.newaxis] > 0)

    prot_c = np.array([1 if e == 'C' else 0 for e in prot_elem])
    lig_c = np.array([1 if e == 'C' else 0 for e in lig_elem])
    hp_mask = (dist < 4.5) & (prot_c[np.newaxis, :] > 0) & (lig_c[:, np.newaxis] > 0)

    prot_pos = np.array([1 if (e == 'N' and r in POSITIVE_RES) else 0 for e, r in zip(prot_elem, prot_res)])
    prot_neg = np.array([1 if (e == 'O' and r in NEGATIVE_RES) else 0 for e, r in zip(prot_elem, prot_res)])
    lig_pos = np.array([1 if e == 'N' else 0 for e in lig_elem])
    lig_neg = np.array([1 if e == 'O' else 0 for e in lig_elem])
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
    """RDKit计算配体分子描述符(17维)"""
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
            desc = {'lig_MW': heavy * 12.0, 'lig_LogP': 0.0, 'lig_TPSA': 0.0, 'lig_num_HBD': 0,
                    'lig_num_HBA': 0, 'lig_num_rotatable_bonds': 0, 'lig_num_aromatic_rings': 0,
                    'lig_num_heavy_atoms': heavy, 'lig_num_rings': 0, 'lig_num_saturated_rings': 0,
                    'lig_num_aliphatic_rings': 0,
                    'lig_num_heteroatoms': sum(a['element'] in ('N', 'O', 'S', 'F', 'P') for a in ligand_atoms),
                    'lig_fraction_sp3': 0.0, 'lig_num_amide_bonds': 0, 'lig_num_valence_electrons': 0,
                    'lig_formal_charge': 0, 'lig_mol_mr': 0.0}
        return {**inter, **desc}
    except Exception as e:
        print(f"  [错误] {complex_pdb_path}: {e}")
        return None


print("特征提取函数已定义 (55维: 38相互作用 + 17描述符)")


# ============================================================
# Phase 2: 2D过滤 (Lipinski + Veber + 反应性 + PAINS + 重原子)
# ============================================================
print("\n" + "=" * 60)
print("Phase 2: 加载MolGPT库 + 2D过滤")
print("=" * 60)

# 8条反应性SMARTS (来源: genai_output/run_params.json)
REACTIVE_SMARTS = [
    '[OX2][OX2]',               # 过氧化物键
    '[CX3](=O)[Cl]',            # 酰氯
    '[#6][Sn]',                 # 有机锡
    '[#15;!$([#15]~[#6])]',     # 非碳磷
    'c1ccc2ccc3ccc4ccc1c2c34',  # 多环芳烃
    '[CX2]#[CX2]',              # 炔烃
    '[NX2]#[NX2]',              # 重氮
    '[#7;+]~[#7;-]',            # 偶氮氧化物
]
# PAINS简化过滤 (用SMARTS, 不用FilterCatalog避免ImportError)
PAINS_SMARTS = [
    '[#7]~[#7]',                # 肼/偶氮
    '[#6](=[#8])~[#8]~[#6]',   # 过氧酯
    '[#6]~[#8]~[#8]~[#6]',     # 过氧化物
]

_reactive_patts = [Chem.MolFromSmarts(s) for s in REACTIVE_SMARTS if Chem.MolFromSmarts(s)]
_pains_patts = [Chem.MolFromSmarts(s) for s in PAINS_SMARTS if Chem.MolFromSmarts(s)]


def passes_2d_filters(mol):
    """Lipinski违规≤1 + Veber + MW≥150 + 重原子≤100 + 非反应性 + 非PAINS
    返回 (ok, props, reason); reason 为英文短标签, 用于过滤分解图"""
    if mol is None:
        return False, {}, 'invalid'
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    heavy = mol.GetNumHeavyAtoms()
    props = {'MW': mw, 'LogP': logp, 'HBD': hbd, 'HBA': hba,
             'RotB': rotb, 'TPSA': tpsa, 'HeavyAtoms': heavy}
    # Lipinski违规≤1
    vio = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    if vio > 1:
        return False, props, 'Lipinski (>1 violation)'
    # Veber
    if rotb > 10 or tpsa > 140:
        return False, props, 'Veber (RotB>10/TPSA>140)'
    # MW下限
    if mw < 150:
        return False, props, 'MW < 150'
    # 重原子上限 (对齐ligprep_9 MAX_ATOMS=100)
    if heavy > 100:
        return False, props, 'Heavy atoms > 100'
    # 反应性SMARTS
    for patt in _reactive_patts:
        if mol.HasSubstructMatch(patt):
            return False, props, 'Reactive SMARTS'
    # PAINS
    for patt in _pains_patts:
        if mol.HasSubstructMatch(patt):
            return False, props, 'PAINS'
    return True, props, 'pass'


# 加载MolGPT CSV
print(f"[加载] {os.path.basename(MOLGPT_CSV)}")
mgpt_df = pd.read_csv(MOLGPT_CSV, encoding='utf-8-sig')
print(f"  CSV行数: {len(mgpt_df)}")
print(f"  列名: {list(mgpt_df.columns)}")

# 提取SMILES列 (兼容不同命名)
smiles_col = None
for c in mgpt_df.columns:
    if c.upper() in ('SMILES', 'CANONICAL_SMILES', 'SMILES'):
        smiles_col = c
        break
if smiles_col is None:
    smiles_col = mgpt_df.columns[0]
    print(f"  [注意] 未找到SMILES列, 使用第一列: {smiles_col}")
print(f"  SMILES列: {smiles_col}")

all_mols = []  # [(smiles, name, mol, props)]
n_fail = 0
for idx, row in mgpt_df.iterrows():
    smi = str(row[smiles_col]).strip()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        n_fail += 1
        continue
    name = f'MGPT_{idx + 1:05d}'
    all_mols.append((smi, name, mol, {}))
print(f"  有效分子: {len(all_mols)}/{len(mgpt_df)} (失败{n_fail})")

# 2D过滤
print("\n[2D过滤] Lipinski≤1 + Veber + MW≥150 + 重原子≤100 + 8反应性SMARTS + PAINS")
filtered_mols = []
filter_stats = defaultdict(int)
reject_reasons = defaultdict(int)   # 各淘汰原因计数 (过滤分解图用)
rejected_mw = []                    # 被拒分子MW (过滤前后MW对比图用)
rejected_logp = []
for smi, name, mol, _ in all_mols:
    ok, props, reason = passes_2d_filters(mol)
    if ok:
        qed = QED.qed(mol)
        filtered_mols.append((smi, name, mol, props, qed))
    else:
        filter_stats['rejected'] += 1
        reject_reasons[reason] += 1
        if props:
            rejected_mw.append(props.get('MW', np.nan))
            rejected_logp.append(props.get('LogP', np.nan))
print("  淘汰原因分解:")
for _r, _c in sorted(reject_reasons.items(), key=lambda kv: -kv[1]):
    print(f"    {_r:32s} {_c}")

n_filtered = len(filtered_mols)
print(f"[2D过滤] {len(all_mols)} → {n_filtered} (淘汰{len(all_mols) - n_filtered})")

# 去重 (canonical SMILES)
seen = set()
unique_mols = []
for smi, name, mol, props, qed in filtered_mols:
    canon = Chem.MolToSmiles(mol)
    if canon not in seen:
        seen.add(canon)
        unique_mols.append((smi, name, mol, props, qed))
print(f"[去重] {n_filtered} → {len(unique_mols)}")

# 图1: 化合物库分子性质三联分布 MW / LogP / QED (模版图1样式; 红虚线=实际使用的阈值)
try:
    mw_all   = [Descriptors.MolWt(m) for _, _, m, _ in all_mols]
    logp_all = [Descriptors.MolLogP(m) for _, _, m, _ in all_mols]
    qed_all  = [QED.qed(m) for _, _, m, _ in all_mols]
    mw_pass   = [x[3].get('MW', 0) for x in unique_mols]
    logp_pass = [x[3].get('LogP', 0) for x in unique_mols]
    qed_pass  = [x[4] for x in unique_mols]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # MW: 全部浅色 + 通过深色, Lipinski 500 红线 + 150 下界线
    ax[0].hist(mw_all, bins=55, color='#9ec9e8', alpha=0.8, label=f'All generated (n={len(mw_all)})')
    ax[0].hist(mw_pass, bins=55, color='#1f6fb4', alpha=0.75, label=f'Passed 2D filter (n={len(mw_pass)})')
    ax[0].axvline(500, color='red', ls='--', lw=1.5, label='Lipinski MW=500')
    ax[0].axvline(150, color='red', ls=':', lw=1.3, label='min MW=150')
    ax[0].set_xlabel('Molecular Weight (Da)'); ax[0].set_ylabel('Count')
    ax[0].set_title('MW Distribution'); ax[0].legend(fontsize=7.5)
    # LogP
    ax[1].hist(logp_all, bins=55, color='#b7e1b0', alpha=0.8, label=f'All (n={len(logp_all)})')
    ax[1].hist(logp_pass, bins=55, color='#2a9d3f', alpha=0.75, label=f'Passed (n={len(logp_pass)})')
    ax[1].axvline(5, color='red', ls='--', lw=1.5, label='Lipinski LogP=5')
    ax[1].set_xlabel('LogP (MolLogP)'); ax[1].set_ylabel('Count')
    ax[1].set_title('LogP Distribution'); ax[1].legend(fontsize=7.5)
    # QED (无硬阈值, 灰色中位线; QED实际用于Top500排序)
    ax[2].hist(qed_all, bins=55, color='#f3c98b', alpha=0.8, label=f'All (n={len(qed_all)})')
    ax[2].hist(qed_pass, bins=55, color='#e8890c', alpha=0.75, label=f'Passed (n={len(qed_pass)})')
    ax[2].axvline(float(np.median(qed_pass)), color='gray', ls='--', lw=1.3,
                  label=f'Passed median={np.median(qed_pass):.2f}')
    ax[2].set_xlabel('QED'); ax[2].set_ylabel('Count')
    ax[2].set_title('QED Distribution (used for Top-N ranking)'); ax[2].legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '01_library_properties.png'), dpi=150)
    plt.close()
    print(f"[可视化] 01_library_properties.png (MW/LogP/QED三联)")
except Exception as e:
    print(f"[可视化] 01性质分布图跳过: {e}")


# ============================================================
# Phase 3: QED预筛 Top N
# ============================================================
print("\n" + "=" * 60)
print(f"Phase 3: QED预筛 Top {TOP_N_DOCK}")
print("=" * 60)

def _select_strata(pool, n):
    """
    口袋导向分层预选:
      1) QED≥0.45地板 + MW 150~500, 保证类药性
      2) 前 n/2 名保留纯QED最优 (多为已对接缓存分子, 重跑秒过)
      3) 后 n/2 按MW区间配额补齐, 配额权重偏向300-450 (能填满4MXU口袋、相互作用多)
         区间内按 QED降序→重原子多者优先; 某区间不足时配额自动让给有余区间
    """
    elig = [m for m in pool if m[4] >= 0.45 and 150 <= m[3].get('MW', 0) <= 501]
    if len(elig) < n:
        elig = list(pool)
    chosen, seen = [], set()
    # 1) QED一半 (保留缓存)
    for m in sorted(elig, key=lambda x: x[4], reverse=True)[:max(1, n // 2)]:
        if m[1] not in seen:
            chosen.append(m); seen.add(m[1])
    # 2) MW分层补另一半: (下限, 上限, 配额权重)
    bins = [(150, 250, 1), (250, 300, 2), (300, 350, 3),
            (350, 400, 3), (400, 450, 2), (450, 501, 1)]
    total_w = sum(w for _, _, w in bins)
    need = n - len(chosen)
    bin_pools, quotas = [], []
    for lo, hi, w in bins:
        bp = sorted([m for m in elig if lo <= m[3].get('MW', 0) < hi and m[1] not in seen],
                    key=lambda x: (x[4], x[3].get('HeavyAtoms', 0)), reverse=True)
        bin_pools.append(bp)
        quotas.append(max(0, round(need * w / total_w)))
    take = [min(q, len(bp)) for q, bp in zip(quotas, bin_pools)]
    while sum(take) < need:   # 不足区间的配额让给有余区间
        deficit = need - sum(take)
        progressed = False
        for i in range(len(bin_pools)):
            room = len(bin_pools[i]) - take[i]
            if room > 0:
                step = min(deficit, room)
                take[i] += step; deficit -= step; progressed = True
                if deficit == 0:
                    break
        if not progressed:
            break
    for bp, k in zip(bin_pools, take):
        for m in bp[:k]:
            if m[1] not in seen:
                chosen.append(m); seen.add(m[1])
    if len(chosen) < n:   # 极端兜底: 全体按QED补
        for m in sorted(elig, key=lambda x: x[4], reverse=True):
            if len(chosen) >= n:
                break
            if m[1] not in seen:
                chosen.append(m); seen.add(m[1])
    return chosen[:n]

if SELECT_MODE == 'strata':
    print("[预筛] 策略=strata (QED地板0.45 + MW分层, 偏向300-450口袋填充区)")
    top_mols = _select_strata(unique_mols, TOP_N_DOCK)
else:
    unique_mols.sort(key=lambda x: x[4], reverse=True)
    top_mols = unique_mols[:TOP_N_DOCK]
print(f"[预筛] {len(unique_mols)} → {len(top_mols)}")
if top_mols:
    _mws = [m[3].get('MW', 0) for m in top_mols]
    print(f"  QED范围: {min(m[4] for m in top_mols):.3f} ~ {max(m[4] for m in top_mols):.3f}")
    print(f"  MW范围: {min(_mws):.0f} ~ {max(_mws):.0f}, 中位={np.median(_mws):.0f} "
          f"(旧纯QED模式中位约250-320, 偏小)")

# 保存预筛结果
prefilter_df = pd.DataFrame({
    'SMILES': [m[0] for m in top_mols],
    'Name': [m[1] for m in top_mols],
    'QED': [m[4] for m in top_mols],
    'MW': [m[3].get('MW', 0) for m in top_mols],
    'LogP': [m[3].get('LogP', 0) for m in top_mols],
    'HeavyAtoms': [m[3].get('HeavyAtoms', 0) for m in top_mols],
})
PREFILTER_CSV = os.path.join(WORK_DIR, f'prefilter_{SELECT_MODE}_top{TOP_N_DOCK}.csv')
prefilter_df.to_csv(PREFILTER_CSV, index=False, encoding='utf-8-sig')
print(f"[保存] {os.path.basename(PREFILTER_CSV)}")


# ============================================================
# Phase 4: 3D构象生成 (ETKDG + MMFF94)
# ============================================================
print("\n" + "=" * 60)
print("Phase 4: 3D构象生成 (RDKit ETKDG + MMFF94)")
print("=" * 60)

ligand_sdf_dir = os.path.join(WORK_DIR, 'ligands_3d')
os.makedirs(ligand_sdf_dir, exist_ok=True)

success_3d = 0
for i, (smi, name, mol, props, qed) in enumerate(top_mols):
    out_sdf = os.path.join(ligand_sdf_dir, f'lig_{i:05d}.sdf')
    if os.path.exists(out_sdf):
        # 检查已有文件是否有效
        check = next(Chem.SDMolSupplier(out_sdf, removeHs=False), None)
        if check is not None and check.GetNumConformers() > 0:
            success_3d += 1
            continue
        os.remove(out_sdf)

    mol_3d = Chem.AddHs(mol)
    # ETKDG版本兼容: v3 → v2 → 原版
    try:
        params = AllChem.ETKDGv3()
    except Exception:
        try:
            params = AllChem.ETKDGv2()
        except Exception:
            params = AllChem.ETKDG()
    try:
        AllChem.EmbedMolecule(mol_3d, params)
        try:
            AllChem.MMFFOptimizeMolecule(mol_3d)
        except Exception:
            pass
    except Exception:
        continue
    if mol_3d.GetNumConformers() == 0:
        continue
    writer = Chem.SDWriter(out_sdf)
    mol_3d.SetProp('_Name', name)
    mol_3d.SetProp('SMILES', smi)
    mol_3d.SetProp('QED', str(qed))
    writer.write(mol_3d)
    writer.close()
    success_3d += 1
    if (i + 1) % 100 == 0:
        print(f"  进度: {i + 1}/{len(top_mols)}, 成功{success_3d}")

print(f"[3D构象] 成功 {success_3d}/{len(top_mols)}")


# ============================================================
# Phase 5: 分子对接
# ============================================================
print("\n" + "=" * 60)
print(f"Phase 5: 分子对接 | 引擎 = {DOCK_ENGINE}")
print("=" * 60)

# 准备受体PDB (只保留ATOM)
receptor_clean = os.path.join(WORK_DIR, 'receptor_clean.pdb')
with open(RECEPTOR_PDB, 'r') as f:
    lines = f.readlines()
with open(receptor_clean, 'w') as f:
    for line in lines:
        if line.startswith('ATOM'):
            f.write(line)
    f.write('END\n')
print(f"[受体] 清洗后: {receptor_clean}")


# ---- 引擎1: GNINA批量对接 ----
def run_gnina_batch(receptor, ligand_dir, out_dir, center, size, gnina_bin, timeout=3600):
    """GNINA批量对接: 合并所有配体SDF → 一次调用 → 拆分输出
    修正关键: 用显式 --center_x/y/z + --size_x/y/z (不用autobox_ligand)
    """
    # 1. 合并所有SDF为一个大文件
    ligand_files = sorted([f for f in os.listdir(ligand_dir) if f.endswith('.sdf')])
    combined_sdf = os.path.join(out_dir, 'all_ligands.sdf')
    print(f"  [GNINA批量] 合并 {len(ligand_files)} 个配体 → {os.path.basename(combined_sdf)}")
    writer = Chem.SDWriter(combined_sdf)
    mol_name_map = {}  # name → original_filename
    for lf in ligand_files:
        mol = next(Chem.SDMolSupplier(os.path.join(ligand_dir, lf), removeHs=False), None)
        if mol is None or mol.GetNumConformers() == 0:
            continue
        name = lf.replace('.sdf', '')
        mol.SetProp('_Name', name)
        writer.write(mol)
        mol_name_map[name] = lf
    writer.close()
    n_input = len(mol_name_map)
    print(f"  [GNINA批量] 有效配体: {n_input}")

    # 2. GNINA一次调用 (显式盒子坐标, CNN rescore模式)
    out_sdf = os.path.join(out_dir, 'gnina_batch_out.sdf')
    cmd = [
        gnina_bin,
        '-r', receptor,
        '-l', combined_sdf,
        '--center_x', str(center[0]),
        '--center_y', str(center[1]),
        '--center_z', str(center[2]),
        '--size_x', str(size[0]),
        '--size_y', str(size[1]),
        '--size_z', str(size[2]),
        '--cnn_scoring', 'rescore',  # Vina搜索 + CNN重打分 (GPU加速)
        '--num_modes', '1',          # 只输出最佳pose
        '--seed', '42',
        '-o', out_sdf,
    ]
    print(f"  [GNINA批量] 调用: {' '.join(cmd[:8])} ...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    el = time.time() - t0
    print(f"  [GNINA批量] 完成, 耗时 {el:.0f}s ({el / 60:.1f}min)")

    # 3. 解析输出, 拆分为单个SDF
    results = {}  # name → (dock_score, cnn_score, cnn_affinity, docked_sdf_path)
    if os.path.exists(out_sdf):
        suppl = Chem.SDMolSupplier(out_sdf, removeHs=False)
        for mol in suppl:
            if mol is None:
                continue
            name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
            ds = float(mol.GetProp('minimizedAffinity')) if mol.HasProp('minimizedAffinity') else None
            cs = float(mol.GetProp('CNNscore')) if mol.HasProp('CNNscore') else None
            ca = float(mol.GetProp('CNNaffinity')) if mol.HasProp('CNNaffinity') else None
            if name in mol_name_map:
                orig_file = mol_name_map[name]
                docked_path = os.path.join(out_dir, orig_file.replace('.sdf', '_docked.sdf'))
                w = Chem.SDWriter(docked_path)
                w.write(mol)
                w.close()
                results[name] = (ds, cs, ca, docked_path)
    print(f"  [GNINA批量] 成功解析 {len(results)}/{n_input} 分子")
    # 标记失败的分子
    for name, lf in mol_name_map.items():
        if name not in results:
            results[name] = (None, None, None, None)
    return results, ligand_files


# ---- 引擎2: Vina逐分子对接 (PDBQT函数同方案B) ----
def _ad4_atom_type(atom):
    """简化AutoDock4原子类型"""
    sym = atom.GetSymbol()
    if sym == 'C':
        return 'A' if atom.GetIsAromatic() else 'C'
    nH = atom.GetTotalNumHs()
    if sym == 'N':
        return 'N' if nH > 0 else 'NA'
    if sym == 'O':
        return 'OA'
    if sym == 'S':
        return 'S' if nH > 0 else 'SA'
    if sym in ('F', 'Cl', 'Br', 'I', 'P', 'Fe', 'Mg', 'Zn', 'Mn', 'Ca'):
        return sym
    return 'C'


def _mol_to_pdbqt(mol, pdbqt_path, resname='LIG'):
    """RDKit分子(含3D构象) → 刚性配体PDBQT (Gasteiger电荷)"""
    m = Chem.Mol(mol)
    if m.GetNumHeavyAtoms() > 999:
        raise ValueError('配体重原子数超过999')
    Chem.AllChem.ComputeGasteigerCharges(m)
    conf = m.GetConformer()
    lines = ['ROOT\n']
    serial = 0
    for i in range(m.GetNumAtoms()):
        atom = m.GetAtomWithIdx(i)
        if atom.GetSymbol() == 'H':
            continue
        serial += 1
        p = conf.GetAtomPosition(i)
        q = float(atom.GetProp('_GasteigerCharge'))
        if not math.isfinite(q):
            q = 0.0
        tt = _ad4_atom_type(atom)
        lines.append(
            f"HETATM{serial:5d} X{i:03d} {resname:>3s} {1:4d}     "
            f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}{1.00:6.2f}{0.00:6.2f}{q:10.3f} {tt:<2s}\n")
    lines.append('ENDROOT\nTORSDOF 0\n')
    with open(pdbqt_path, 'w') as f:
        f.writelines(lines)


RECEPTOR_PDBQT = os.path.join(WORK_DIR, 'receptor_4MXU.pdbqt')


def prepare_receptor_pdbqt():
    """受体PDB→PDBQT: RDKit+Gasteiger电荷; 失败回退0电荷"""
    if os.path.exists(RECEPTOR_PDBQT):
        return
    ok = False
    try:
        prot = Chem.MolFromPDBFile(receptor_clean, removeHs=True, proximityBonding=True)
        if prot is not None and prot.GetNumConformers() > 0:
            Chem.AllChem.ComputeGasteigerCharges(prot)
            conf = prot.GetConformer()
            out = []
            serial = 0
            for i in range(prot.GetNumAtoms()):
                atom = prot.GetAtomWithIdx(i)
                serial += 1
                p = conf.GetAtomPosition(i)
                q = float(atom.GetProp('_GasteigerCharge')) if atom.HasProp('_GasteigerCharge') else 0.0
                if not math.isfinite(q):
                    q = 0.0
                out.append(
                    f"ATOM  {serial:5d} X{serial % 1000:03d} REC {1:4d}     "
                    f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}{1.00:6.2f}{0.00:6.2f}{q:10.3f} {_ad4_atom_type(atom):<2s}\n")
            out.append('END\n')
            with open(RECEPTOR_PDBQT, 'w') as f:
                f.writelines(out)
            ok = True
            print("[受体] PDBQT生成成功 (RDKit Gasteiger电荷)")
    except Exception as e:
        print(f"[受体] RDKit生成PDBQT失败({e}), 回退0电荷")
    if not ok:
        with open(receptor_clean, 'r') as f:
            pdb_lines = f.readlines()
        out = []
        serial = 0
        for line in pdb_lines:
            if not line.startswith('ATOM'):
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                elem = (line[76:78].strip() or 'C').upper()
            except Exception:
                continue
            tt = {'C': 'C', 'N': 'NA', 'O': 'OA', 'S': 'SA'}.get(elem, 'C')
            serial += 1
            out.append(
                f"ATOM  {serial:5d} {serial:04d} REC {1:4d}     "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{1.00:6.2f}{0.00:6.2f}{0.0:10.3f} {tt:<2s}\n")
        out.append('END\n')
        with open(RECEPTOR_PDBQT, 'w') as f:
            f.writelines(out)
        print("[受体] PDBQT生成成功 (0电荷近似)")


def _mol_heavy_center(mol):
    conf = mol.GetConformer()
    pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetSymbol() != 'H'])
    return pts.mean(axis=0)


def _apply_pdbqt_pose(template_mol, pose_pdbqt):
    """用Vina输出PDBQT中的坐标更新模板分子构象"""
    conf = template_mol.GetConformer()
    n_upd = 0
    n_heavy = sum(1 for a in template_mol.GetAtoms() if a.GetSymbol() != 'H')
    with open(pose_pdbqt, 'r') as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            name = line[12:16].strip()
            if not name.startswith('X'):
                continue
            try:
                idx = int(name[1:])
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                conf.SetAtomPosition(idx, (x, y, z))
                n_upd += 1
            except Exception:
                continue
    return n_upd >= max(n_heavy - 2, 1)


def run_vina_docking(ligand_sdf, output_sdf):
    """Vina对接单个配体, 返回docking_score(kcal/mol)"""
    from vina import Vina
    mol = next(Chem.SDMolSupplier(ligand_sdf, removeHs=False), None)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    base = output_sdf.replace('_docked.sdf', '')
    lig_pdbqt = base + '_lig.pdbqt'
    pose_pdbqt = base + '_vina.pdbqt'
    _mol_to_pdbqt(mol, lig_pdbqt)
    center = list(DOCK_CENTER)
    v = Vina(sf_name='vina', cpu=VINA_CPU, seed=42, verbosity=0)
    v.set_receptor(rigid_pdbqt_filename=RECEPTOR_PDBQT)
    v.set_ligand_from_file(lig_pdbqt)
    v.compute_vina_maps(center=[float(c) for c in center], box_size=list(DOCK_SIZE))
    v.dock(exhaustiveness=VINA_EXHAUST, n_poses=1)
    energies = np.array(v.energies())
    score = float(energies[0][0])
    if not math.isfinite(score):
        return None
    v.write_poses(pose_pdbqt, n_poses=1, overwrite=True)
    if _apply_pdbqt_pose(mol, pose_pdbqt):
        mol.SetDoubleProp('dock_score', score)
        w = Chem.SDWriter(output_sdf)
        w.write(mol)
        w.close()
    return score


def run_empirical_docking(ligand_sdf, output_sdf):
    """RDKit经验打分 (保底, 平移到盒子中心做接触统计)"""
    mol = next(Chem.SDMolSupplier(ligand_sdf, removeHs=False), None)
    if mol is None or mol.GetNumConformers() == 0:
        return None
    with open(receptor_clean, 'r') as f:
        prot_atoms = parse_pdb_atoms(f.readlines())
    if not prot_atoms:
        return None
    prot_coords = np.array([a['coord'] for a in prot_atoms])
    prot_elem = [a['element'] for a in prot_atoms]
    ref_center = np.array(DOCK_CENTER)
    conf = mol.GetConformer()
    lig_pts = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                        for i in range(mol.GetNumAtoms())])
    shift = ref_center - lig_pts.mean(axis=0)
    for i in range(mol.GetNumAtoms()):
        p = lig_pts[i] + shift
        conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
    heavy_idx = [i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetSymbol() != 'H']
    lc = lig_pts[heavy_idx] + shift
    le = [mol.GetAtomWithIdx(i).GetSymbol() for i in heavy_idx]
    diff = lc[:, np.newaxis, :] - prot_coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    n_contact = int(np.sum(dist < 4.5))
    n_clash = int(np.sum(dist < 2.6))
    prot_pol = np.array([1 if e in ('N', 'O', 'S') else 0 for e in prot_elem])
    lig_pol = np.array([1 if e in ('N', 'O', 'S') else 0 for e in le])
    n_hb = int(np.sum((dist < 3.5) & (prot_pol[np.newaxis, :] > 0) & (lig_pol[:, np.newaxis] > 0)))
    raw = 0.12 * n_contact + 1.0 * n_hb - 2.0 * n_clash
    score = -float(np.clip(raw, 1.5, 12.0))
    mol.SetDoubleProp('dock_score', score)
    w = Chem.SDWriter(output_sdf)
    w.write(mol)
    w.close()
    return score


# ---- 执行对接 ----
docking_dir = os.path.join(WORK_DIR, 'docking_results')
os.makedirs(docking_dir, exist_ok=True)

docking_results = []  # [(lig_file, dock_score, cnn_score, cnn_affinity, engine, docked_sdf)]
ligand_files = sorted([f for f in os.listdir(ligand_sdf_dir) if f.endswith('.sdf')])
print(f"[对接] 共 {len(ligand_files)} 个配体 | 引擎={DOCK_ENGINE}")

if DOCK_ENGINE == 'gnina':
    # GNINA批量: 一次调用所有配体
    gnina_results, _ = run_gnina_batch(
        receptor_clean, ligand_sdf_dir, docking_dir,
        DOCK_CENTER, DOCK_SIZE, GNINA_BIN)
    for lf in ligand_files:
        name = lf.replace('.sdf', '')
        if name in gnina_results:
            ds, cs, ca, docked = gnina_results[name]
        else:
            ds, cs, ca, docked = None, None, None, None
        docking_results.append((lf, ds, cs, ca, DOCK_ENGINE, docked))

elif DOCK_ENGINE == 'vina':
    # Vina逐分子
    prepare_receptor_pdbqt()
    t0 = time.time()
    for i, lf in enumerate(ligand_files):
        lig_path = os.path.join(ligand_sdf_dir, lf)
        out_sdf = os.path.join(docking_dir, lf.replace('.sdf', '_docked.sdf'))
        if os.path.exists(out_sdf):
            mol_d = next(Chem.SDMolSupplier(out_sdf), None)
            if mol_d is not None and mol_d.HasProp('dock_score'):
                ds = float(mol_d.GetProp('dock_score'))
                docking_results.append((lf, ds, None, None, DOCK_ENGINE, out_sdf))
                continue
        try:
            ds = run_vina_docking(lig_path, out_sdf)
        except Exception as _de:
            print(f"  [失败] {lf}: {_de.__class__.__name__}")
            ds = None
        if ds is not None and not math.isfinite(float(ds)):
            ds = None
        docking_results.append((lf, ds, None, None, DOCK_ENGINE, out_sdf if ds is not None else None))
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            speed = (i + 1) / max(el, 1)
            remain = (len(ligand_files) - i - 1) / max(speed, 0.1)
            n_ok = sum(1 for r in docking_results if r[1] is not None)
            print(f"  进度: {i + 1}/{len(ligand_files)} | 有效{n_ok} | 速度={speed:.2f}/s | 剩余~{remain / 60:.0f}min")
    el = time.time() - t0
    print(f"\n[Vina完成] {len(docking_results)}分子, 耗时{el:.0f}s ({el / 60:.1f}min)")

else:
    # 保底经验打分
    t0 = time.time()
    for i, lf in enumerate(ligand_files):
        lig_path = os.path.join(ligand_sdf_dir, lf)
        out_sdf = os.path.join(docking_dir, lf.replace('.sdf', '_docked.sdf'))
        if os.path.exists(out_sdf):
            mol_d = next(Chem.SDMolSupplier(out_sdf), None)
            if mol_d is not None and mol_d.HasProp('dock_score'):
                ds = float(mol_d.GetProp('dock_score'))
                docking_results.append((lf, ds, None, None, DOCK_ENGINE, out_sdf))
                continue
        try:
            ds = run_empirical_docking(lig_path, out_sdf)
        except Exception:
            ds = None
        docking_results.append((lf, ds, None, None, DOCK_ENGINE, out_sdf if ds is not None else None))
        if (i + 1) % 100 == 0:
            print(f"  进度: {i + 1}/{len(ligand_files)}")
    print(f"\n[经验打分完成] {len(docking_results)}分子, 耗时{time.time() - t0:.0f}s")

# 保存对接结果
dock_df = pd.DataFrame(docking_results,
                       columns=['lig_file', 'dock_score', 'cnn_score', 'cnn_affinity', 'dock_engine', 'docked_sdf'])
n_valid = int(dock_df['dock_score'].notna().sum())
dock_df.to_csv(os.path.join(WORK_DIR, 'docking_results.csv'), index=False)
print(f"\n[保存] docking_results.csv | 有效打分: {n_valid}")
if n_valid > 0:
    print(f"  打分范围: {dock_df['dock_score'].min():.2f} ~ {dock_df['dock_score'].max():.2f}")
    if dock_df['cnn_score'].notna().any():
        print(f"  CNNscore范围: {dock_df['cnn_score'].min():.3f} ~ {dock_df['cnn_score'].max():.3f}")

# QC图: 对接打分分布 (正式报告用三联评分布图见Phase10的04)
try:
    _has_cnn = dock_df['cnn_score'].notna().any()
    if _has_cnn:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax0 = ax[0]; ax1 = ax[1]
        valid_cnn = dock_df['cnn_score'].dropna()
        ax1.hist(valid_cnn, bins=40, color='mediumpurple', alpha=0.7)
        ax1.set_xlabel('CNN Score'); ax1.set_ylabel('Count')
        ax1.set_title(f'CNN Score Distribution (n={len(valid_cnn)})')
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(7, 4.5))
    valid_scores = dock_df['dock_score'].dropna()
    ax0.hist(valid_scores, bins=40, color='coral', alpha=0.75)
    ax0.set_xlabel('Docking Score (kcal/mol)'); ax0.set_ylabel('Count')
    ax0.set_title(f'Docking Score Distribution (n={len(valid_scores)})')
    ax0.axvline(float(valid_scores.median()), color='red', linestyle='--',
                label=f'median {valid_scores.median():.2f}')
    ax0.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'qc_docking_hist.png'), dpi=150)
    plt.close()
    print(f"[可视化] qc_docking_hist.png (QC速览图)")
except Exception as e:
    print(f"[可视化] 对接QC图跳过: {e}")


# ============================================================
# Phase 6: 复合物PDB构建 (受体ATOM + 对接pose HETATM)
# ============================================================
print("\n" + "=" * 60)
print("Phase 6: 构建复合物PDB")
print("=" * 60)

with open(receptor_clean, 'r') as f:
    receptor_lines = [l for l in f.readlines() if l.startswith('ATOM')]

complex_dir = os.path.join(WORK_DIR, 'complex_pdbs')
os.makedirs(complex_dir, exist_ok=True)


def sdf_to_pdb_hetatm(sdf_path, pdb_out):
    """SDF → PDB (HETATM记录)"""
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = next(suppl, None)
    if mol is None:
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=True)
        mol = next(suppl, None)
    if mol is None:
        return False
    try:
        Chem.MolToPDBFile(mol, pdb_out)
        return True
    except Exception:
        return False


success_complex = 0
for i, row in dock_df.iterrows():
    docked_sdf = row['docked_sdf']
    if not docked_sdf or not os.path.exists(docked_sdf):
        continue
    lig_file = row['lig_file']
    lig_pdb = os.path.join(complex_dir, lig_file.replace('.sdf', '_lig.pdb'))
    if not sdf_to_pdb_hetatm(docked_sdf, lig_pdb):
        continue
    complex_pdb = os.path.join(complex_dir, lig_file.replace('.sdf', '_complex.pdb'))
    with open(complex_pdb, 'w') as f:
        for line in receptor_lines:
            f.write(line)
        with open(lig_pdb, 'r') as lf:
            for line in lf:
                if line.startswith('HETATM') or line.startswith('ATOM'):
                    if line.startswith('ATOM'):
                        line = 'HETATM' + line[6:]
                    f.write(line)
        f.write('END\n')
    success_complex += 1

print(f"[复合物PDB] 成功 {success_complex}/{len(dock_df)}")


# ============================================================
# Phase 7: 特征提取 (55维)
# ============================================================
print("\n" + "=" * 60)
print("Phase 7: 提取55维三维特征")
print("=" * 60)

aspa_features = []
aspa_names = []
aspa_smiles = []
aspa_sources = []
aspa_dock_scores = []
aspa_cnn_scores = []
aspa_cnn_affs = []
aspa_engines = []

for _, row in dock_df.iterrows():
    lig_file = row['lig_file']
    docked_sdf = row['docked_sdf']
    if not docked_sdf or not os.path.exists(docked_sdf):
        continue
    complex_pdb = os.path.join(complex_dir, lig_file.replace('.sdf', '_complex.pdb'))
    if not os.path.exists(complex_pdb):
        continue
    feats = extract_features_from_complex(complex_pdb)
    if feats is None:
        continue
    # 从lig_file获取索引 → 查预筛表
    try:
        idx = int(lig_file.replace('lig_', '').replace('.sdf', ''))
        name = prefilter_df.iloc[idx]['Name']
        smi = prefilter_df.iloc[idx]['SMILES']
    except (ValueError, IndexError):
        name = lig_file
        smi = ''
    aspa_features.append(feats)
    aspa_names.append(name)
    aspa_smiles.append(smi)
    aspa_sources.append('MolGPT')
    aspa_dock_scores.append(row['dock_score'])
    aspa_cnn_scores.append(row['cnn_score'])
    aspa_cnn_affs.append(row['cnn_affinity'])
    aspa_engines.append(row['dock_engine'])

print(f"  成功提取 {len(aspa_features)} 个分子特征")


# ============================================================
# Phase 8: XGBoost精排 (复用方案B模型 或 仅用对接打分)
# ============================================================
print("\n" + "=" * 60)
print("Phase 8: XGBoost精排")
print("=" * 60)

HAS_MODEL = MODEL_CACHE is not None
predictions = None
train_columns = None

if HAS_MODEL:
    print(f"[模型] 加载: {MODEL_CACHE}")
    # 集成模型可能含 LightGBM/CatBoost, joblib.load 前确保库已安装
    import importlib
    for _dep, _pip in [('lightgbm', 'lightgbm'), ('catboost', 'catboost')]:
        try:
            importlib.import_module(_dep)
        except ImportError:
            print(f"  [依赖] 模型可能包含{_dep}, 正在安装 ...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', _pip], timeout=600)
    import traceback
    try:
        cache = joblib.load(MODEL_CACHE)
        models = cache['models']
        weights = cache['weights']
        scaler = cache['scaler']
        selected_idx = cache['selected_idx']
        train_columns = cache['X_train_df_columns']
        print(f"  [模型] 集成成员: {list(models.keys())}")

        # 对齐特征列
        X_aspa = pd.DataFrame(aspa_features)
        _missing = [c for c in train_columns if c not in X_aspa.columns]
        _extra = [c for c in X_aspa.columns if c not in train_columns]
        if _missing:
            print(f"  [特征对齐] 警告: {len(_missing)}个训练特征本次缺失(补0): {_missing}")
        if _extra:
            print(f"  [特征对齐] 提示: {len(_extra)}个本次特征不在训练集中(丢弃): {_extra}")
        if not _missing and not _extra:
            print(f"  [特征对齐] {len(train_columns)}个特征名完全一致")
        for col in _missing:
            X_aspa[col] = 0.0
        X_aspa = X_aspa[train_columns]

        # 预处理 + 预测
        X_scaled = scaler.transform(X_aspa.values)
        X_sel = X_scaled[:, selected_idx]

        predictions = np.zeros(X_sel.shape[0])
        total_w = sum(weights.values())
        for name_m, model in models.items():
            w = weights[name_m] / total_w
            try:
                pred = model.predict(X_sel)
            except Exception as _pe:
                # GPU训练的模型在无GPU环境加载时: XGBoost尝试显式切CPU再预测
                try:
                    if name_m == 'xgb':
                        model.set_params(device='cpu')
                    pred = model.predict(X_sel)
                    print(f"    [回退] {name_m} 已切换CPU预测")
                except Exception:
                    raise _pe
            predictions += pred * w
            print(f"    {name_m}权重={w:.2f}")
        print(f"\n  XGBoost预测范围: {predictions.min():.2f} ~ {predictions.max():.2f}")
    except Exception as e:
        print(f"  [错误] 模型加载/预测失败: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        print("  → 请确认该pkl由方案B(3_unified_screening.py)完整训练后生成, 且xgboost/lightgbm/catboost版本一致")
        HAS_MODEL = False

if not HAS_MODEL:
    # ---- 回退方案: ensemble_models.pkl 缺失时, 用方案B已保存的JSON最终模型
    #      + pdbbind_features.csv/labels.npy 按相同代码重建 scaler + median特征选择器 ----
    PLANB_OUT = os.path.join(os.path.expanduser('~/ASPA_planB'), 'output_screening')
    JSON_MODEL = os.path.join(PLANB_OUT, 'xgboost_aspa_final.json')
    FEAT_CSV = os.path.join(PLANB_OUT, 'pdbbind_features.csv')
    LABEL_NPY = os.path.join(PLANB_OUT, 'pdbbind_labels.npy')

    if os.path.exists(JSON_MODEL) and os.path.exists(FEAT_CSV) and os.path.exists(LABEL_NPY):
        print("[模型回退] 未找到 ensemble_models.pkl, 但发现方案B JSON模型 + PDBbind特征矩阵")
        print(f"  JSON: {JSON_MODEL}")
        print("  → 按方案B相同代码重建 StandardScaler + median特征选择器 (random_state=42)")
        print("  → 最终打分模型直接使用该JSON (与方案B全量重训后的final_model完全相同)")
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.feature_selection import SelectFromModel

            X_train_df = pd.read_csv(FEAT_CSV)
            y_train = np.load(LABEL_NPY)
            assert len(X_train_df) == len(y_train), \
                f'特征({len(X_train_df)})与标签({len(y_train)})行数不一致'

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train_df.values, y_train, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)

            # 基准模型参数 (与3_unified_screening.py Cell 8完全一致, 用于重建median选择器)
            base_params = {'objective': 'reg:squarederror', 'max_depth': 6, 'learning_rate': 0.1,
                           'n_estimators': 1000, 'subsample': 0.8, 'colsample_bytree': 0.8,
                           'min_child_weight': 1, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
                           'random_state': 42, 'n_jobs': -1, 'early_stopping_rounds': 50}
            base_params.update(GPU_PARAMS if HAS_GPU else {'tree_method': 'hist'})
            print("  [重建] 训练基准模型以复现median特征选择 ...")
            base_model = xgb.XGBRegressor(**base_params)
            base_model.fit(X_tr_scaled, y_tr, eval_set=[(X_te_scaled, y_te)], verbose=False)
            selector = SelectFromModel(base_model, threshold='median', prefit=True)
            selected_idx = selector.get_support()
            train_columns = X_train_df.columns.tolist()
            print(f"  [重建] 特征选择: {len(train_columns)} → {int(sum(selected_idx))}个")

            # 加载JSON最终模型 (方案B全量重训后的final_model)
            final_model = xgb.XGBRegressor()
            final_model.load_model(JSON_MODEL)
            print("  [模型] JSON最终模型加载成功 (xgb单模型, 权重=1.00)")

            # 对齐特征列
            X_aspa = pd.DataFrame(aspa_features)
            _missing = [c for c in train_columns if c not in X_aspa.columns]
            if _missing:
                print(f"  [特征对齐] 警告: {len(_missing)}个训练特征本次缺失(补0): {_missing}")
            else:
                print(f"  [特征对齐] {len(train_columns)}个特征名完全一致")
            for col in train_columns:
                if col not in X_aspa.columns:
                    X_aspa[col] = 0.0
            X_aspa = X_aspa[train_columns]
            X_scaled = scaler.transform(X_aspa.values.astype(float))
            X_sel = X_scaled[:, selected_idx]

            predictions = final_model.predict(X_sel)
            HAS_MODEL = True
            print(f"  XGBoost预测范围: {predictions.min():.2f} ~ {predictions.max():.2f}")
            print("  [说明] 此路径与ensemble_models.pkl的唯一差异: 缺少LGB/CatBoost集成成员;")
            print("         方案B的pkl生成后, 重跑本脚本会自动改用完整集成(含断点续跑, 不重对接)")
        except Exception as e:
            import traceback
            print(f"  [错误] JSON回退路径失败: {e.__class__.__name__}: {e}")
            traceback.print_exc()
            HAS_MODEL = False

if not HAS_MODEL:
    print("[模型] ensemble_models.pkl 和 xgboost_aspa_final.json 均未找到, 跳过XGBoost精排")
    print("  → 本次排名仅基于对接打分(Vina), 结果有效但缺少模型精排层")
    print("  → 请确认方案B的 output_screening 目录已完整生成")
    print("  → 模型就位后重跑本脚本: 对接结果有断点续跑会自动跳过, 仅补做XGBoost精排和排名")


# ============================================================
# Phase 9: 综合排名 + 输出
# ============================================================
print("\n" + "=" * 60)
print("Phase 9: 综合排名")
print("=" * 60)

results = pd.DataFrame({
    '分子名称': aspa_names,
    'SMILES': aspa_smiles,
    '来源': aspa_sources,
    'XGBoost预测': predictions if predictions is not None else [np.nan] * len(aspa_names),
    '对接打分': aspa_dock_scores,
    'CNNscore': aspa_cnn_scores,
    'CNNaffinity': aspa_cnn_affs,
    '对接引擎': aspa_engines,
})

# 排名 (排名百分位归一化, 解决不同引擎分数量纲不一致)
# 注意: rank(...)列约定"第1名=最好"; 百分位必须满足"最好=1.0",
# 不能直接用rank(pct=True) — 那会让第1名得到~0, 综合分反而选到最差分子。
_N = len(results)
def _good_pct(rank_series):
    """第1名→1.0, 末名→1/N (并列同名次取相同百分位)"""
    return (_N - rank_series + 1.0) / _N

# 1. XGBoost排名 (预测值越高越好)
if predictions is not None:
    results['XGBoost排名'] = results['XGBoost预测'].rank(ascending=False, method='min').astype(int)
    results['XGBoost百分位'] = _good_pct(results['XGBoost排名'])
else:
    results['XGBoost排名'] = np.nan
    results['XGBoost百分位'] = np.nan

# 2. 对接打分排名 (越负越好)
results['对接排名'] = results['对接打分'].rank(ascending=True, method='min')
results['对接百分位'] = _good_pct(results['对接排名'])

# 3. CNNscore排名 (越高越好, 仅GNINA引擎有值)
CNN_AVAILABLE = results['CNNscore'].notna().sum() > len(results) * 0.5
if CNN_AVAILABLE:
    results['CNN排名'] = results['CNNscore'].rank(ascending=False, method='min')
    results['CNN百分位'] = _good_pct(results['CNN排名'])
    rank_cols = ['XGBoost百分位', '对接百分位', 'CNN百分位']
else:
    results['CNN排名'] = np.nan
    results['CNN百分位'] = np.nan
    rank_cols = ['XGBoost百分位', '对接百分位'] if predictions is not None else ['对接百分位']

# 最终排名策略
if RANK_MODE == 'xgb_only':
    if predictions is None:
        raise RuntimeError("RANK_MODE=xgb_only 需要XGBoost模型, 但未加载到 ensemble_models.pkl")
    # 仅用XGBoost: 综合排名 = XGBoost排名, 综合分 = XGBoost百分位
    results['综合排名分'] = results['XGBoost百分位']
    results['综合排名'] = results['XGBoost排名'].astype(int)
    print("[排名] 策略=xgb_only: 仅按XGBoost预测(pK)从高到低排序, 对接分不参与排名")
else:
    # 各百分位均值 (0~1, 越高越好), NaN用中位数0.5填充
    valid_ranks = results[rank_cols].fillna(0.5)
    results['综合排名分'] = valid_ranks.mean(axis=1)
    # 先按综合分降序 + 次级键(XGB降序/对接升序)排序, 再用 method='first'
    # 得到连续无重复的 1..N 排名 (避免 method='min' 导致 #3 出现多次的并列重号)
    results = results.sort_values(
        ['综合排名分', 'XGBoost预测', '对接打分'],
        ascending=[False, False, True], na_position='last').reset_index(drop=True)
    results['综合排名'] = results['综合排名分'].rank(ascending=False, method='first').astype(int)
    print(f"[排名] 策略=consensus: {rank_cols} 百分位融合 (并列时按XGB降序/对接升序打破)")

# 排序 (综合排名已连续无重复, 分子名称仅作稳定次级键)
results = results.sort_values(['综合排名', '分子名称']).reset_index(drop=True)

# NaN安全格式化
def _fmt(v, spec):
    try:
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return '  -- '
        return format(v, spec)
    except Exception:
        return '  -- '

# 输出Top N
print(f"\n★ 排名 Top {min(FINAL_TOP_N, len(results))} ({RANK_MODE}) ★")
if RANK_MODE == 'xgb_only':
    print(f"{'#':>3} {'分子名称':16s} {'XGBoost预测pK':>12s} {'预测Kd(μM)':>10s}")
    print("-" * 50)
    for _, row in results.head(FINAL_TOP_N).iterrows():
        _pk = row['XGBoost预测']
        _kd = f"{10**(6 - _pk):.2f}" if math.isfinite(_pk) else '--'   # pK=6 → 1μM
        print(f"  #{int(row['综合排名']):3d} {str(row['分子名称'])[:16]:16s} {_pk:12.3f} {_kd:>10s}")
else:
    print(f"{'#':>3} {'分子名称':16s} {'XGB':>7s} {'Dock':>7s} {'CNN':>6s} {'综合':>6s} {'引擎'}")
    print("-" * 70)
    for _, row in results.head(FINAL_TOP_N).iterrows():
        print(f"  #{int(row['综合排名']):3d} {str(row['分子名称'])[:16]:16s} "
              f"{_fmt(row['XGBoost预测'], '.2f'):>7s} "
              f"{_fmt(row['对接打分'], '.2f'):>7s} "
              f"{_fmt(row['CNNscore'], '.3f'):>6s} "
              f"{row['综合排名分']:.3f}  {str(row['对接引擎'])}")

# 保存全量结果(保留全部列, 可追溯)
ranking_csv = os.path.join(WORK_DIR, 'ASPA_molgpt_ranking.csv')
results.to_csv(ranking_csv, index=False, encoding='utf-8-sig')

# 最终候选表列定义
final_n = min(FINAL_TOP_N, len(results))
if RANK_MODE == 'xgb_only':
    final_cols = ['综合排名', '分子名称', 'SMILES', '来源', 'XGBoost预测',
                  'XGBoost百分位', '对接打分']
    final_csv = os.path.join(WORK_DIR, f'final_top{final_n}_xgboost.csv')
    _csv_desc = '仅XGBoost排名'
else:
    # 综合模式: 保留两路分数/排名/百分位, 去掉Vina下全空的CNN列
    final_cols = ['综合排名', '综合排名分', '分子名称', 'SMILES', '来源',
                  'XGBoost预测', 'XGBoost排名', 'XGBoost百分位',
                  '对接打分', '对接排名', '对接百分位', '对接引擎']
    final_csv = os.path.join(WORK_DIR, f'final_top{final_n}_consensus.csv')
    _csv_desc = 'XGBoost+对接综合排名'
final_cols = [c for c in final_cols if c in results.columns]
results.head(final_n)[final_cols].to_csv(final_csv, index=False, encoding='utf-8-sig')
# 兼容旧文件名(部分报告引用final_top10_candidates.csv)
results.head(min(10, len(results)))[final_cols].to_csv(
    os.path.join(WORK_DIR, 'final_top10_candidates.csv'), index=False, encoding='utf-8-sig')
print(f"\n[保存] {os.path.basename(ranking_csv)} ({len(results)}分子全量排名, 全部列)")
print(f"[保存] {os.path.basename(final_csv)} (最终{final_n}候选, {_csv_desc})")


# ============================================================
# Phase 10: 可视化
# ============================================================
print("\n" + "=" * 60)
print("Phase 10: 可视化")
print("=" * 60)

# 名称 → prefilter行号/lig编号 映射 (名称编号≠行号时也正确; 修复旧版 split('_')-1 的隐患)
_name_to_idx = {str(n): i for i, n in enumerate(prefilter_df['Name'].tolist())}

# ---------- 图2: 筛选漏斗柱状图 (对标报告模版Screening Pipeline; 蓝渐变+柱顶数字) ----------
try:
    funnel_labels = ['Generated\n(valid SMILES)', '2D rule\nfilter', 'Dedup',
                     f'QED/strata\nTop{TOP_N_DOCK}', '3D\nconformers', 'Docked\n(Vina)',
                     f'Final\nTop{final_n}']
    funnel_counts = [len(all_mols), n_filtered, len(unique_mols), len(top_mols),
                     int(success_3d), int(n_valid), int(final_n)]
    _funnel_colors = plt.cm.Blues(np.linspace(0.45, 0.92, len(funnel_counts)))
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(funnel_labels))
    bars = ax.bar(x, funnel_counts, width=0.62, color=_funnel_colors,
                  edgecolor='#2b5b84', linewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(funnel_labels, fontsize=9)
    ax.set_ylabel('Number of Compounds', fontsize=11)
    ax.set_title('MolGPT Virtual Screening Pipeline', fontsize=13, fontweight='bold')
    ax.grid(axis='y', ls='--', alpha=0.35, zorder=0)
    _ymax = max(funnel_counts)
    ax.set_ylim(0, _ymax * 1.14)
    for b, v in zip(bars, funnel_counts):
        ax.text(b.get_x() + b.get_width() / 2, v + _ymax * 0.015,
                f'{v:,}', ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    # 阶段间保留率
    for i in range(1, len(funnel_counts)):
        _pct = 100.0 * funnel_counts[i] / max(funnel_counts[i - 1], 1)
        ax.text(x[i] - 0.5, _ymax * 1.07, f'{_pct:.1f}%', ha='center', va='top',
                fontsize=8, color='#555555')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '02_screening_funnel.png'), dpi=150)
    plt.close()
    print("[可视化] 02_screening_funnel.png")
except Exception as e:
    print(f"[可视化] 漏斗图跳过: {e}")

# ---------- 图3: 2D过滤原因分解 + 过滤前后MW对比 (对标模版图3) ----------
try:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={'width_ratios': [1.05, 1.4]})
    # 左: 各规则淘汰数量横向条形
    _items = sorted(reject_reasons.items(), key=lambda kv: kv[1])
    _rn = [k for k, _ in _items]
    _rc = [v for _, v in _items]
    _bars = ax[0].barh(range(len(_rn)), _rc, color='#e07b5f', alpha=0.85, zorder=3)
    ax[0].set_yticks(range(len(_rn)))
    ax[0].set_yticklabels(_rn, fontsize=8.5)
    ax[0].set_xlabel('Rejected Count')
    ax[0].set_title(f'2D Filter Rejection Breakdown (total rejected={sum(_rc):,})')
    ax[0].grid(axis='x', ls='--', alpha=0.35, zorder=0)
    _xmax = max(_rc) if _rc else 1
    for b, v in zip(_bars, _rc):
        ax[0].text(v + _xmax * 0.01, b.get_y() + b.get_height() / 2, str(v),
                   va='center', fontsize=8.5)
    ax[0].set_xlim(0, _xmax * 1.18)
    # 右: MW before/after过滤叠加
    _mw_all = [Descriptors.MolWt(m) for _, _, m, _ in all_mols]
    _mw_pass = [x[3].get('MW', 0) for x in unique_mols]
    ax[1].hist(_mw_all, bins=55, color='#9ec9e8', alpha=0.9,
               label=f'Before filter (n={len(_mw_all):,})')
    ax[1].hist(_mw_pass, bins=55, color='#1f6fb4', alpha=0.75,
               label=f'After 2D filter (n={len(_mw_pass):,})')
    ax[1].axvline(500, color='red', ls='--', lw=1.4, label='Lipinski MW=500')
    ax[1].axvline(150, color='red', ls=':', lw=1.2, label='min MW=150')
    ax[1].set_xlabel('Molecular Weight (Da)'); ax[1].set_ylabel('Count')
    ax[1].set_title('MW Distribution: Before vs After 2D Filtering')
    ax[1].legend(fontsize=8.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '03_filter_breakdown.png'), dpi=150)
    plt.close()
    print("[可视化] 03_filter_breakdown.png")
except Exception as e:
    print(f"[可视化] 过滤分解图跳过: {e}")

# ---------- 图4: 评分三联分布 对接分 / XGB预测 / 综合分 (中位数红线; 对标模版图4) ----------
if predictions is not None:
    try:
        _dock = results['对接打分'].astype(float).dropna()
        _xgb = results['XGBoost预测'].astype(float).dropna()
        _cons = results['综合排名分'].astype(float).dropna()
        _med_dock = float(_dock.median())
        _med_xgb = float(_xgb.median())
        _med_cons = float(_cons.median())
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        ax[0].hist(_dock, bins=40, color='coral', alpha=0.8)
        ax[0].axvline(_med_dock, color='red', ls='--', lw=1.5,
                      label=f'median={_med_dock:.2f}')
        ax[0].set_xlabel('Docking Score (kcal/mol)'); ax[0].set_ylabel('Count')
        ax[0].set_title(f'Vina Docking Score (n={len(_dock)})'); ax[0].legend(fontsize=8)
        ax[1].hist(_xgb, bins=40, color='mediumpurple', alpha=0.8)
        ax[1].axvline(_med_xgb, color='red', ls='--', lw=1.5,
                      label=f'median={_med_xgb:.2f}')
        ax[1].set_xlabel('XGBoost Predicted pK (-logKd/Ki)'); ax[1].set_ylabel('Count')
        ax[1].set_title(f'ML Interaction Score (n={len(_xgb)})'); ax[1].legend(fontsize=8)
        ax[2].hist(_cons, bins=40, color='#2ca02c', alpha=0.8)
        ax[2].axvline(_med_cons, color='red', ls='--', lw=1.5,
                      label=f'median={_med_cons:.2f}')
        ax[2].set_xlabel('Consensus Percentile Score'); ax[2].set_ylabel('Count')
        ax[2].set_title(f'Consensus Score (n={len(_cons)})'); ax[2].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '04_score_distributions.png'), dpi=150)
        plt.close()
        print("[可视化] 04_score_distributions.png")
    except Exception as e:
        print(f"[可视化] 三联评分分布图跳过: {e}")

    # ---------- 图5: XGB预测 vs 对接分散点 (Top20金色星标; 对标模版图5) ----------
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        valid = results[['XGBoost预测', '对接打分', '综合排名']].dropna()
        rest = valid[valid['综合排名'] > 20]
        topn = valid[valid['综合排名'] <= 20]
        ax.scatter(rest['对接打分'], rest['XGBoost预测'],
                   alpha=0.45, c='#7f9bb8', s=22, label=f'Other candidates (n={len(rest)})')
        ax.scatter(topn['对接打分'], topn['XGBoost预测'],
                   marker='*', s=320, c='#d4a017', edgecolors='#7a5c00',
                   linewidths=0.6, zorder=5, label=f'Top {len(topn)} (consensus)')
        for _, row in topn.head(10).iterrows():
            ax.annotate(f"#{int(row['综合排名'])}",
                        (row['对接打分'], row['XGBoost预测']),
                        fontsize=8, fontweight='bold', ha='left',
                        xytext=(4, 3), textcoords='offset points')
        ax.set_xlabel('Docking Score (kcal/mol, more negative = stronger)')
        ax.set_ylabel('XGBoost Prediction (-logKd/Ki, higher = stronger)')
        ax.set_title(f'XGBoost Prediction vs Docking Score (n={len(valid)})')
        ax.legend(loc='lower left', fontsize=9)
        ax.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '05_xgb_vs_dock_scatter.png'), dpi=150)
        plt.close()
        print("[可视化] 05_xgb_vs_dock_scatter.png")
    except Exception as e:
        print(f"[可视化] 散点图跳过: {e}")

    # 诊断图: XGB预测分布 vs PDBbind训练标签分布 (域漂移诊断)
    try:
        _lab_path = os.path.join(os.path.dirname(WORK_DIR.rstrip('/\\')),
                                 'output_screening', 'pdbbind_labels.npy')
        if not os.path.exists(_lab_path):
            _lab_path = os.path.join(WORK_DIR, 'pdbbind_labels.npy')
        y_lab = np.load(_lab_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(min(predictions.min(), y_lab.min()),
                           max(predictions.max(), y_lab.max()), 45)
        ax.hist(y_lab, bins=bins, alpha=0.55, color='gray', label=f'PDBbind train labels (n={len(y_lab)})', density=True)
        ax.hist(predictions, bins=bins, alpha=0.65, color='crimson', label=f'MolGPT predictions (n={len(predictions)})', density=True)
        ax.axvline(y_lab.mean(), color='gray', ls='--', lw=1.5, label=f'PDBbind mean={y_lab.mean():.2f}')
        ax.axvline(predictions.mean(), color='crimson', ls='--', lw=1.5, label=f'MolGPT mean={predictions.mean():.2f}')
        ax.set_xlabel('Affinity pK  (-logKd/Ki)')
        ax.set_ylabel('Density')
        ax.set_title('MolGPT Predictions vs PDBbind Label Distribution')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, '08_pred_vs_train_distribution.png'), dpi=150)
        plt.close()
        print("[可视化] 08_pred_vs_train_distribution.png")
        _qs = np.percentile(predictions, [5, 25, 50, 75, 95])
        print(f"  [诊断] MolGPT预测: mean={predictions.mean():.2f}, max={predictions.max():.2f}, "
              f"p5/25/50/75/95={np.round(_qs,2).tolist()}")
        print(f"  [诊断] PDBbind标签: mean={y_lab.mean():.2f}, range={y_lab.min():.2f}~{y_lab.max():.2f}")
    except Exception as e:
        print(f"[可视化] 训练标签分布图跳过: {e}")

# ---------- 图6: Top20综合排名横向柱 (红Top3/蓝4-10/浅蓝11-20; 对标模版图8) ----------
try:
    top20 = results.head(20).iloc[::-1]  # 反转使第1名在上
    fig, ax = plt.subplots(figsize=(10.5, 7))

    def _rank_color(r):
        if r <= 3:
            return '#d62728'        # 红 Top3
        elif r <= 10:
            return '#1f77b4'        # 蓝 4-10
        return '#aec7e8'            # 浅蓝 11-20

    colors = [_rank_color(r) for r in top20['综合排名']]
    if RANK_MODE == 'xgb_only':
        bar_vals = top20['XGBoost预测'].astype(float)
        x_label = 'XGBoost Predicted Affinity pK  (-logKd/Ki, higher = stronger)'
        title = f'Top 20 by XGBoost (total n={len(results)})'
    else:
        bar_vals = top20['综合排名分'].astype(float)
        x_label = 'Consensus Percentile Score (mean of XGB/Dock percentiles)'
        title = f'Top 20 Candidates by Consensus Ranking (total n={len(results)})'
    ax.barh(range(len(top20)), bar_vals, color=colors, alpha=0.88, zorder=3)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f"#{int(r)} {str(n)[:13]}" for r, n in zip(top20['综合排名'], top20['分子名称'])],
                       fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='x', ls='--', alpha=0.35, zorder=0)
    # 三色图例
    from matplotlib.patches import Patch as _Patch
    ax.legend(handles=[_Patch(color='#d62728', label='Top 3'),
                       _Patch(color='#1f77b4', label='Rank 4-10'),
                       _Patch(color='#aec7e8', label='Rank 11-20')],
              loc='lower right', fontsize=9)
    if RANK_MODE == 'xgb_only':
        _lo = float(bar_vals.min())
        ax.set_xlim(_lo - 0.15, float(bar_vals.max()) + 0.55)
        for i, (_, row) in enumerate(top20.iterrows()):
            _pk = float(row['XGBoost预测'])
            ax.text(_pk + 0.01, i, f"pK={_pk:.2f}  Kd~{10**(6-_pk):.2f} uM", va='center', fontsize=7.5)
    else:
        _lo = float(bar_vals.min())
        ax.set_xlim(_lo - 0.02, float(bar_vals.max()) + 0.12)
        for i, (_, row) in enumerate(top20.iterrows()):
            ax.text(float(row['综合排名分']) + 0.004, i,
                    f"XGB={_fmt(row['XGBoost预测'], '.1f')}  Dock={_fmt(row['对接打分'], '.1f')}",
                    va='center', fontsize=7.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '06_top20_ranking.png'), dpi=150)
    plt.close()
    print("[可视化] 06_top20_ranking.png")
except Exception as e:
    print(f"[可视化] Top20排名柱图跳过: {e}")

# ---------- 图7: Top20二维结构网格 (RDKit; 对标模版图9) ----------
try:
    from rdkit.Chem import Draw as _CustDraw
    _grid_mols, _grid_legends = [], []
    for _, row in results.head(20).iterrows():
        _m = Chem.MolFromSmiles(str(row['SMILES']))
        if _m is None:
            continue
        _grid_mols.append(_m)
        _pk = float(row['XGBoost预测']) if 'XGBoost预测' in row.index else float('nan')
        _dk = float(row['对接打分']) if '对接打分' in row.index else float('nan')
        _grid_legends.append(f"#{int(row['综合排名'])} {str(row['分子名称'])}\n"
                             f"XGB {_pk:.2f} | Dock {_dk:.1f}")
    if _grid_mols:
        _img = _CustDraw.MolsToGridImage(
            _grid_mols, molsPerRow=4,
            subImgSize=(380, 300),
            legends=_grid_legends)
        _img.save(os.path.join(PLOTS_DIR, '07_top20_structures.png'))
        print(f"[可视化] 07_top20_structures.png ({len(_grid_mols)}个二维结构)")
except Exception as e:
    print(f"[可视化] Top20二维结构图跳过: {e}")

# ---------- 图9(HTML): Top3复合物3D可视化 (py3Dmol) ----------
if HAS_PY3DMOL:
    try:
        top3_complexes = []
        for _, row in results.head(3).iterrows():
            _nm = str(row['分子名称'])
            _li = _name_to_idx.get(_nm)
            if _li is None:
                continue
            lig_file = f"lig_{_li:05d}.sdf"
            cpdb = os.path.join(complex_dir, lig_file.replace('.sdf', '_complex.pdb'))
            if os.path.exists(cpdb):
                top3_complexes.append((int(row['综合排名']), cpdb))
        if top3_complexes:
            html_parts = ['<html><head><meta charset="utf-8">'
                          '<style>body{margin:0;padding:0;}</style></head><body>']
            for rank, cpdb in top3_complexes:
                with open(cpdb, 'r') as f:
                    pdb_data = f.read()
                view = py3Dmol.view(width=800, height=500)
                view.addModel(pdb_data, 'pdb')
                view.setStyle({'cartoon': {'color': 'spectrum'}})
                view.setStyle({'resn': 'LIG'}, {'stick': {'colorscheme': 'yellowCarbon'}})
                view.setViewStyle({'style': 'outline'})
                view.zoomTo({'resn': 'LIG'})
                html_parts.append(f'<h3>Rank {rank}</h3>')
                html_parts.append(view._make_html())
            html_parts.append('</body></html>')
            html_path = os.path.join(PLOTS_DIR, '09_top3_complexes.html')
            with open(html_path, 'w') as f:
                f.write('\n'.join(html_parts))
            print(f"[可视化] 09_top3_complexes.html (浏览器打开可交互旋转)")
    except Exception as e:
        print(f"[可视化] 3D HTML跳过: {e}")
else:
    print("[可视化] py3Dmol未安装, 跳过3D交互可视化 (pip install py3Dmol可启用)")

# 复制Top5复合物PDB到plots目录方便查看 (用 prefilter_df 行号映射, 避免名称编号错位)
for _, row in results.head(5).iterrows():
    try:
        name = str(row['分子名称'])
        idx = _name_to_idx.get(name)
        if idx is None:
            continue
        lig_file = f'lig_{idx:05d}.sdf'
        cpdb = os.path.join(complex_dir, lig_file.replace('.sdf', '_complex.pdb'))
        if os.path.exists(cpdb):
            import shutil as _sh
            _sh.copy(cpdb, os.path.join(PLOTS_DIR, f'top{int(row["综合排名"])}_{name}.pdb'))
    except Exception:
        pass

# 清理旧版图名 (避免与新编号混淆)
for _old in ['01_qed_mw_distribution.png', '02_dock_score_distribution.png',
             '03_xgb_vs_dock_scatter.png', '04_top20_candidates.png',
             '05_top3_complexes.html', '06_pred_vs_train_distribution.png']:
    try:
        _op = os.path.join(PLOTS_DIR, _old)
        if os.path.exists(_op):
            os.remove(_op)
    except Exception:
        pass


# ============================================================
# 汇总
# ============================================================
print("\n" + "=" * 60)
print("全流程执行完毕")
print("=" * 60)
print(f"""
┌──────────────────────────────────────────────────────────┐
│           MolGPT支路虚拟筛选 汇总                         │
├──────────────────────────────────────────────────────────┤
│ Phase 1 MolGPT加载: {len(all_mols)} SMILES                        │
│ Phase 2 2D过滤+去重: → {len(unique_mols)}                          │
│ Phase 3 QED预筛: Top {TOP_N_DOCK:<5}                               │
│ Phase 4 3D构象: {success_3d}/{TOP_N_DOCK} 成功                      │
│ Phase 5 分子对接: {n_valid}有效打分, 引擎={DOCK_ENGINE:<16s}│
│ Phase 6 复合物PDB: {success_complex}个                            │
│ Phase 7 特征提取: {len(aspa_features)}个55维特征                   │
│ Phase 8 XGBoost: {'复用方案B模型' if HAS_MODEL else '跳过(仅对接打分)':<20s}        │
│ Phase 9 排名({RANK_MODE}): {len(results)}分子 → Top{final_n}{'':<7s}│
│ 可视化: {PLOTS_DIR}
└──────────────────────────────────────────────────────────┘
""")
print("关键输出文件:")
print(f"  1. {os.path.basename(final_csv)} (最终{final_n}候选, 仅XGBoost列)")
print(f"  2. ASPA_molgpt_ranking.csv (全量排名, 保留全部列可追溯)")
print(f"  3. prefilter_{SELECT_MODE}_top{TOP_N_DOCK}.csv (预筛结果)")
print(f"  4. docking_results.csv (对接打分)")
print(f"  5. plots/ (可视化图表)")
