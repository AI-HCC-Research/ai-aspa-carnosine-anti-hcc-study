"""
RF构象打分系统 - 特征提取模块 (步骤六)
======================================
功能：从构象 PDB + 原始 CavityPlus/HDBSCAN CSV 中提取 23 维口袋特征
输入：1) 构象 PDB 文件 (data/step6/conformations/open_conformation_XX.pdb)
      2) data/step6/raw_csv/ 下的原始 CSV (hdbscan_result.csv 等)
输出：扩充后的特征矩阵 (X) + 标签建议 (三源投票)
"""

import os
import math
import csv
import numpy as np
from collections import defaultdict


# ============================================================================
# 路径常量 (本文件位于 src/, 项目根为上一级, 数据在 data/step6/)
# ============================================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "step6")
CONF_DIR = os.path.join(DATA_DIR, "conformations")
CSV_DIR = os.path.join(DATA_DIR, "raw_csv")
CRYSTAL_PDB = os.path.join(DATA_DIR, "ASPA_2I3C.pdb")  # 如果不存在就跳过RMSD计算


# ============================================================================
# 1. PDB解析工具
# ============================================================================
def parse_pdb(path):
    """解析PDB文件，返回ATOM/HETATM原子列表"""
    atoms = []
    if not os.path.exists(path):
        return atoms
    with open(path, "r") as f:
        for line in f:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM"):
                try:
                    atom = {
                        "record": rec,
                        "serial": int(line[6:11].strip() or 0),
                        "atom_name": line[12:16].strip(),
                        "altloc": line[16:17].strip(),
                        "resname": line[17:20].strip(),
                        "chain": line[21:22].strip() or "A",
                        "resseq": int(line[22:26].strip() or 0),
                        "icode": line[26:27].strip(),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "occ": float(line[54:60] or 1.0),
                        "bfactor": float(line[60:66] or 0.0),
                        "element": line[76:78].strip() if len(line) > 76 else "",
                    }
                    if not atom["element"]:
                        atom["element"] = atom["atom_name"][0] if atom["atom_name"] else ""
                    atoms.append(atom)
                except (ValueError, IndexError):
                    continue
            elif rec == "END":
                break
    return atoms


def distance(a, b):
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2)


# ============================================================================
# 2. ASPA催化口袋定义（基于晶体2I3C的催化残基）
# ============================================================================
# ASPA活性位点：Zn²⁺ + HIS21 + HIS116 + ASP141 + GLU24 + 关键水HOH#329
# 口袋中心近似 = 这几个催化残基 CA 原子的平均坐标
CATALYTIC_RESIDUES = {
    21: "HIS",    # Zn 配体
    116: "HIS",   # Zn 配体
    141: "ASP",   # Zn 配体
    24: "GLU",    # 催化碱
    142: "SER",   # 底物结合
    164: "ASN",   # 底物结合
    273: "ARG",   # 底物结合
}


def get_pocket_center(atoms, chain="A"):
    """根据催化残基CA原子计算口袋中心"""
    pocket_cas = []
    for a in atoms:
        if (a["chain"] == chain and a["record"] == "ATOM"
                and a["atom_name"] == "CA" and a["resseq"] in CATALYTIC_RESIDUES):
            pocket_cas.append([a["x"], a["y"], a["z"]])
    if not pocket_cas:
        # fallback: 用晶体2I3C的Zn坐标（如果有的话）
        if os.path.exists(CRYSTAL_PDB):
            cat_atoms = parse_pdb(CRYSTAL_PDB)
            for a in cat_atoms:
                if a["resname"] == "ZN" and a["chain"] == chain:
                    return np.array([a["x"], a["y"], a["z"]])
        return np.array([0.0, 0.0, 0.0])
    return np.mean(pocket_cas, axis=0)


def get_pocket_residues(atoms, center, radius=12.0, chain="A"):
    """获取口袋半径内所有蛋白残基"""
    pocket_res = defaultdict(list)  # resseq -> [atom list]
    for a in atoms:
        if a["chain"] != chain or a["record"] != "ATOM":
            continue
        coord = np.array([a["x"], a["y"], a["z"]])
        d = np.linalg.norm(coord - center)
        if d <= radius:
            pocket_res[a["resseq"]].append(a)
    return pocket_res


# ============================================================================
# 3. 扩充特征计算（核心！4维 -> 23维）
# ============================================================================
def compute_extended_features(conf_id, atoms, raw_row=None):
    """
    对单个构象的单个口袋计算扩充特征
    conf_id: 'conf_01' 这样的编号
    atoms: 该构象的PDB atom list
    raw_row: 从hdbscan_result.csv读取的原始行字典（体积/深度/疏水/可药性/cluster）
    返回：dict of features
    """
    feat = {}

    # --- 原始4维特征 ---
    if raw_row:
        feat["pocket_volume"] = float(raw_row.get("口袋体积", 0.0))
        feat["pocket_depth"] = float(raw_row.get("口袋深度", 0.0))
        feat["hydrophobic_sasa"] = float(raw_row.get("疏水表面积", 0.0))
        feat["cavity_druggability"] = float(raw_row.get("可药性评分", 0.0))
        feat["cluster"] = int(raw_row.get("cluster", -1))
    else:
        feat["pocket_volume"] = 0.0
        feat["pocket_depth"] = 0.0
        feat["hydrophobic_sasa"] = 0.0
        feat["cavity_druggability"] = 0.0
        feat["cluster"] = -1

    # --- 衍生几何特征 ---
    feat["atom_density"] = feat["pocket_volume"] / feat["pocket_depth"] if feat["pocket_depth"] > 0 else 0.0
    # 开口估算：体积/深度² 作为开口横截面积近似
    feat["opening_area_est"] = feat["pocket_volume"] / (feat["pocket_depth"] ** 2) if feat["pocket_depth"] > 0 else 0.0
    # 疏水比：疏水表面积 / 体积
    feat["hydrophobic_ratio"] = feat["hydrophobic_sasa"] / feat["pocket_volume"] if feat["pocket_volume"] > 0 else 0.0

    # --- 口袋中心 + 残基级特征 ---
    center = get_pocket_center(atoms)
    pocket_res = get_pocket_residues(atoms, center, radius=12.0)
    resseqs = list(pocket_res.keys())

    # 计数各种残基类型
    hbd_count = 0  # 氢键供体: SER/THR/TYR/LYS/ASN/GLN/HIS
    hba_count = 0  # 氢键受体: ASP/GLU/ASN/GLN/SER/THR/TYR
    pos_charge = 0  # 正电: LYS/ARG/HIS(+)
    neg_charge = 0  # 负电: ASP/GLU
    aromatic = 0    # 芳香: PHE/TYR/TRP/HIS
    hydrophobic_count = 0  # 疏水: ALA/VAL/LEU/ILE/MET/PRO/PHE/TRP

    HBD_RES = {"SER", "THR", "TYR", "LYS", "ASN", "GLN", "HIS", "ARG", "CYS"}
    HBA_RES = {"ASP", "GLU", "ASN", "GLN", "SER", "THR", "TYR", "HIS"}
    POS_RES = {"LYS", "ARG"}
    NEG_RES = {"ASP", "GLU"}
    ARO_RES = {"PHE", "TYR", "TRP", "HIS"}
    HYDRO_RES = {"ALA", "VAL", "LEU", "ILE", "MET", "PRO", "PHE", "TRP"}

    his_count = 0
    asp_count = 0
    glu_count = 0
    catalytic_present = 0  # 催化残基在口袋内的完整度（0-7）

    for rseq, r_atoms in pocket_res.items():
        if not r_atoms:
            continue
        resname = r_atoms[0]["resname"]

        if resname in HBD_RES:
            hbd_count += 1
        if resname in HBA_RES:
            hba_count += 1
        if resname in POS_RES:
            pos_charge += 1
        if resname in NEG_RES:
            neg_charge += 1
        if resname in ARO_RES:
            aromatic += 1
        if resname in HYDRO_RES:
            hydrophobic_count += 1

        if resname == "HIS":
            his_count += 1
        if resname == "ASP":
            asp_count += 1
        if resname == "GLU":
            glu_count += 1

        # 催化残基检测
        if rseq in CATALYTIC_RESIDUES:
            catalytic_present += 1

    n_pocket_res = max(1, len(resseqs))
    feat["hbd_count"] = hbd_count
    feat["hba_count"] = hba_count
    feat["hb_total"] = hbd_count + hba_count
    feat["hb_density"] = (hbd_count + hba_count) / feat["pocket_volume"] if feat["pocket_volume"] > 0 else 0.0
    feat["pos_charge_count"] = pos_charge
    feat["neg_charge_count"] = neg_charge
    feat["charge_net"] = pos_charge - neg_charge
    feat["charge_total"] = pos_charge + neg_charge
    feat["aromatic_count"] = aromatic
    feat["hydrophobic_res_count"] = hydrophobic_count
    feat["hydrophobic_res_ratio"] = hydrophobic_count / n_pocket_res

    # --- Zn配位特征（ASPA特异性！）---
    # 思路：判断HIS21/HIS116/ASP141这三个Zn配体残基的空间位置是否合理
    # 如果三个都在口袋12Å内且相互距离在3-6Å之间 => 配位完好
    zn_coord_score = 0
    zn_ligands_found = []
    for rseq, target in [(21, "HIS"), (116, "HIS"), (141, "ASP")]:
        if rseq in pocket_res:
            # 取侧链配位原子（HIS: NE2/ND1, ASP: OD1/OD2）
            for at in pocket_res[rseq]:
                if (target == "HIS" and at["atom_name"] in ("NE2", "ND1", "CE1")) or \
                   (target == "ASP" and at["atom_name"] in ("OD1", "OD2", "CG")):
                    zn_ligands_found.append(at)
                    break

    if len(zn_ligands_found) >= 3:
        # 检查三个配位原子相互距离（正常三角构型: 2-7Å）
        d1 = distance(zn_ligands_found[0], zn_ligands_found[1])
        d2 = distance(zn_ligands_found[0], zn_ligands_found[2])
        d3 = distance(zn_ligands_found[1], zn_ligands_found[2])
        if 2.0 < d1 < 7.0 and 2.0 < d2 < 7.0 and 2.0 < d3 < 7.0:
            zn_coord_score = 3  # 三配体都有，几何合理
        else:
            zn_coord_score = 2  # 三配体都有，但几何有点变形
    elif len(zn_ligands_found) >= 2:
        zn_coord_score = 1
    else:
        zn_coord_score = 0

    feat["zn_coord_score"] = zn_coord_score
    feat["catalytic_completeness"] = catalytic_present  # 0-7

    # --- 催化残基关键计数 ---
    feat["his_in_pocket"] = his_count
    feat["asp_in_pocket"] = asp_count
    feat["glu_in_pocket"] = glu_count

    # --- 构象编号（方便调试，不作为特征）---
    feat["_conf_id"] = conf_id

    return feat


# ============================================================================
# 4. 批量特征提取入口
# ============================================================================
def load_raw_csv_data():
    """加载原始hdbscan_result.csv，返回45条口袋的list of dict"""
    csv_path = os.path.join(CSV_DIR, "hdbscan_result.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(CSV_DIR, "step5.csv")  # fallback用step5
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(r))
    if not rows:
        # 尝试GBK编码（Windows中文CSV常见）
        with open(csv_path, "r", encoding="gbk") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(dict(r))
    return rows


def conf_id_to_pdb_path(conf_id):
    """conf_01 -> data/step6/conformations/open_conformation_01.pdb"""
    try:
        num = int(conf_id.split("_")[-1])
    except (IndexError, ValueError):
        return None
    fname = f"open_conformation_{num:02d}.pdb"
    path = os.path.join(CONF_DIR, fname)
    return path if os.path.exists(path) else None


def extract_all_features():
    """
    主入口：对45个口袋（15构象×3口袋）提取扩充特征
    返回：list of dict
    """
    raw_rows = load_raw_csv_data()
    if not raw_rows:
        print("[FE] ❌ 找不到原始CSV数据！请确保 data/step6/raw_csv/hdbscan_result.csv 存在")
        return None

    print(f"[FE] ✅ 加载了 {len(raw_rows)} 条原始口袋数据")

    # 按构象ID缓存 PDB atom list，避免重复解析
    pdb_cache = {}

    results = []
    skipped = 0

    for i, raw in enumerate(raw_rows):
        # 处理列名兼容（不同CSV列名可能不同）
        conf_id = raw.get("构象编号") or raw.get("conf_id") or f"conf_{(i//3)+1:02d}"
        pocket_id = raw.get("口袋ID") or raw.get("pocket_id") or f"Pocket{(i%3)+1}"

        pdb_path = conf_id_to_pdb_path(conf_id)
        if pdb_path is None:
            skipped += 1
            continue

        if conf_id not in pdb_cache:
            pdb_cache[conf_id] = parse_pdb(pdb_path)

        atoms = pdb_cache[conf_id]
        if not atoms:
            skipped += 1
            continue

        feat = compute_extended_features(conf_id, atoms, raw_row=raw)
        feat["_pocket_id"] = pocket_id
        feat["_row_index"] = i
        results.append(feat)

    print(f"[FE] ✅ 提取了 {len(results)} 条扩充特征（跳过 {skipped} 条）")
    if results:
        print(f"[FE] 📐 特征维度: {len(results[0])-3} （不含3个内部标记列）")
    else:
        print("[FE] ❌ 无结果")

    return results


# ============================================================================
# 5. 三源标签融合（Level 1 的伪标签生成）
# ============================================================================
def generate_multi_source_labels(feature_rows):
    """
    三源投票生成训练标签 y（0/1 二分类）
    源1: CavityPlus Druggability >= 中位数  → 1
    源2: 启发式规则打分 >= 7/10  → 1
    源3: Zn配位 + 催化完整度综合判断  → 1
    最终：>=2票 标 y=1（好构象口袋）
    """
    if not feature_rows:
        return [], {}

    druggabilities = [r["cavity_druggability"] for r in feature_rows]
    median_drug = np.median(druggabilities)
    print(f"[LABEL] CavityPlus Druggability 中位数 = {median_drug:.3f}（源1阈值）")

    y_list = []
    vote_details = []
    n_pos = 0

    for r in feature_rows:
        # 源1: CavityPlus
        vote1 = 1 if r["cavity_druggability"] >= median_drug else 0

        # 源2: 启发式规则打分（0-10，>=7→1）
        s2 = 0
        if r["pocket_volume"] > 400:
            s2 += 2
        elif r["pocket_volume"] > 250:
            s2 += 1
        if r["hydrophobic_ratio"] > 0.5:
            s2 += 2
        elif r["hydrophobic_ratio"] > 0.35:
            s2 += 1
        if r["hb_total"] >= 8:
            s2 += 2
        elif r["hb_total"] >= 4:
            s2 += 1
        if r["zn_coord_score"] >= 2:
            s2 += 2
        if r["charge_net"] >= -1 and r["charge_net"] <= 2:
            s2 += 1
        if r["aromatic_count"] >= 2:
            s2 += 1
        vote2 = 1 if s2 >= 7 else 0

        # 源3: 锌配位 + 催化完整
        s3 = 0
        if r["zn_coord_score"] >= 2:
            s3 += 2
        if r["catalytic_completeness"] >= 5:
            s3 += 2
        if r["his_in_pocket"] >= 2:
            s3 += 1
        if r["asp_in_pocket"] >= 1:
            s3 += 1
        if r["pocket_depth"] >= 15:
            s3 += 1
        if r["hydrophobic_res_count"] >= 5:
            s3 += 1
        vote3 = 1 if s3 >= 5 else 0

        total_vote = vote1 + vote2 + vote3
        y = 1 if total_vote >= 2 else 0
        y_list.append(y)
        vote_details.append({"v1_Cavity": vote1, "v2_Rule": vote2, "v3_ZnCat": vote3,
                             "score2": s2, "score3": s3, "total_vote": total_vote})
        if y == 1:
            n_pos += 1

    print(f"[LABEL] ✅ 三源标签完成: 正样本 y=1 共 {n_pos}/{len(y_list)} ({n_pos/len(y_list)*100:.1f}%)")
    return y_list, {
        "median_druggability": median_drug,
        "n_positive": n_pos,
        "n_total": len(y_list),
        "vote_details": vote_details,
    }


if __name__ == "__main__":
    # 测试模式：直接运行查看输出
    rows = extract_all_features()
    if rows:
        y, info = generate_multi_source_labels(rows)
        feat_keys = [k for k in rows[0].keys() if not k.startswith("_")]
        print(f"\n✅ 特征列({len(feat_keys)}): {feat_keys}")
        print(f"✅ 标签分布: 正={info['n_positive']} 负={info['n_total']-info['n_positive']}")
