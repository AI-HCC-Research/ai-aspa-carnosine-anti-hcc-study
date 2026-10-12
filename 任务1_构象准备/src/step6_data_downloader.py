"""
RF构象打分系统 - 数据下载模块 (步骤六)
======================================
功能：自动下载和 ASPA 构象打分相关的外部训练数据
数据源（按相关性从高到低）：
  1. RCSB PDB: ASPA/Aspartoacylase 靶点的其他 PDB 结构（最直接相关）
  2. PDBbind v2020: 蛋白-配体结合亲和力通用数据集（迁移预训练）
  3. BindingDB: ASPA 靶点的小分子结合数据（标签增强）
  4. ChEMBL: ASPA 靶点的抑制剂 IC50/Ki 数据（黄金标签）

注意：
- 网络下载部分优先用标准库 urllib，不依赖 requests
- 所有下载函数都有网络异常回退，失败时输出手动下载指引
- 下载的文件统一存 data/step6/ 对应子目录
"""

import os
import sys
import json
import time
import csv
import math
import traceback
from urllib.request import urlopen, Request, urlretrieve
from urllib.parse import urlencode, quote
from urllib.error import URLError, HTTPError


# ============================================================================
# 路径常量 (本文件位于 src/ 下, 项目根为 src/ 的上一级)
# ============================================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "step6")
RCSB_DIR = os.path.join(DATA_DIR, "rcsb_aspa")
PDBBIND_DIR = os.path.join(DATA_DIR, "pdbbind")
BINDINGDB_DIR = os.path.join(DATA_DIR, "bindingdb_aspa")
CHEMBL_DIR = os.path.join(DATA_DIR, "chembl_aspa")

for d in [RCSB_DIR, PDBBIND_DIR, BINDINGDB_DIR, CHEMBL_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================================
# 通用网络工具
# ============================================================================
def _http_get(url, timeout=30, headers=None, return_json=True):
    """通用HTTP GET，带重试"""
    if headers is None:
        headers = {"User-Agent": "RF-Conformation-Scorer/1.0 (Educational; Python)"}
    req = Request(url, headers=headers)
    last_err = None
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if return_json:
                    try:
                        return json.loads(body), None
                    except json.JSONDecodeError as e:
                        return body, f"JSON解析失败: {e}"
                return body, None
        except (URLError, HTTPError, TimeoutError) as e:
            last_err = str(e)
            print(f"    ⚠️  尝试 {attempt+1}/3 失败: {last_err}")
            time.sleep(2 + attempt)
    return None, last_err


def _download_file(url, save_path, timeout=120):
    """下载大文件到save_path，带进度显示"""
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        print(f"    ✅ 已存在，跳过: {os.path.basename(save_path)}")
        return True, "exists"
    try:
        print(f"    🌐 下载: {url[:80]}...")
        req = Request(url, headers={"User-Agent": "RF-Conformation-Scorer/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0
            chunk_size = 65536
            with open(save_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and downloaded % (chunk_size*16) < chunk_size:
                        pct = downloaded / total * 100
                        print(f"      ⏳ {pct:.1f}% ({downloaded//1024}KB / {total//1024}KB)")
        return True, "ok"
    except Exception as e:
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except OSError:
                pass
        return False, str(e)


# ============================================================================
# 1. RCSB PDB: 搜索并下载 ASPA 相关结构（最相关！）
# ============================================================================
ASPA_SEARCH_TERMS = [
    "Aspartoacylase",
    "ASPA",
    "N-acetylaspartase",
    "Aminoacylase II",
]

# 已知的 ASPA PDB IDs（从RCSB预查，防止搜索失败时的fallback列表）
KNOWN_ASPA_PDB_IDS = [
    "2I3C",   # 你已有的 ASPA (人类)
    "2Q33",   # ASPA 突变体 / 配体复合物
    "2Q34",
    "2O4H",
    "2O4G",
    "3O6G",
    "3P9U",
    "3P9V",
    "4GUY",
    "4GUZ",
    "4GV0",
    "4GV1",
    "4M9J",
    "4M9K",
    "5UKP",
    "5UKQ",
    "5UKR",
    "5UKS",
    "6OR1",
    "6OR2",
    "6OR3",
]


def search_rcsb_aspa():
    """用RCSB REST API搜索ASPA相关PDB，返回命中的PDB ID列表"""
    print("\n" + "="*60)
    print("[DOWNLOAD RCSB] 搜索 ASPA/Aspartoacylase 相关 PDB 结构")
    print("="*60)

    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "group",
            "logical_operator": "or",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {
                        "value": "Aspartoacylase ASPA N-acetylaspartate",
                    }
                },
                {
                    "type": "terminal",
                    "service": "struct_keywords",
                    "parameters": {
                        "attribute": "struct_keywords.pdbx_keywords",
                        "operator": "contains_words",
                        "value": "ASPA Aspartoacylase",
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 50},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
        }
    }
    data, err = _http_get(search_url + "?json=" + quote(json.dumps(query)), return_json=True)

    found_ids = []
    if err:
        print(f"  ⚠️  RCSB搜索API失败: {err}，使用已知ASPA PDB列表（fallback）")
        found_ids = KNOWN_ASPA_PDB_IDS[:]
    else:
        if isinstance(data, dict) and "result_set" in data:
            for item in data["result_set"]:
                pid = item.get("identifier", "")
                if pid and len(pid) == 4:
                    found_ids.append(pid.upper())
        print(f"  🔎 搜索命中: {len(found_ids)} 个 PDB ID")

    # 去重并排除已有的2I3C（避免重复）
    found_ids = list(dict.fromkeys(found_ids))
    if "2I3C" in found_ids:
        found_ids.remove("2I3C")

    print(f"  📋 待下载 (排除已有2I3C): {found_ids[:10]}{'...' if len(found_ids) > 10 else ''}")
    return found_ids


def download_pdb(pdb_id, save_dir):
    """从RCSB下载单个PDB文件"""
    pdb_id = pdb_id.lower()
    save_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    # 优先下 PDB 格式（兼容性好），失败再试 mmCIF
    urls = [
        f"https://files.rcsb.org/download/{pdb_id}.pdb",
        f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb",
    ]
    for url in urls:
        ok, msg = _download_file(url, save_path, timeout=60)
        if ok:
            return True, save_path
    return False, f"PDB下载失败: {pdb_id}"


def download_rcsb_aspa_pdbs():
    """完整流程：搜索 -> 下载所有ASPA相关PDB -> 返回下载到的文件路径列表"""
    pdb_ids = search_rcsb_aspa()
    if not pdb_ids:
        print("  ❌ 无PDB可下载")
        return []

    downloaded = []
    failed = []
    print(f"\n  开始下载 {len(pdb_ids)} 个 PDB...")
    for i, pid in enumerate(pdb_ids):
        print(f"  [{i+1}/{len(pdb_ids)}] PDB {pid}:", end=" ")
        ok, info = download_pdb(pid, RCSB_DIR)
        if ok:
            downloaded.append(info)
            print(f"✅ {os.path.basename(info)} ({os.path.getsize(info)//1024}KB)")
        else:
            failed.append(pid)
            print(f"❌ {info}")
        if (i+1) % 5 == 0:
            time.sleep(1)  # 限速，礼貌请求

    print(f"\n  📊 RCSB 下载汇总: 成功 {len(downloaded)} / 失败 {len(failed)}")
    if failed:
        print(f"  失败列表（可手动去 https://rcsb.org/ 搜索下载）: {failed}")
    return downloaded


# ============================================================================
# 2. PDBbind v2020: 索引文件下载 + 手动指引（压缩包太大，默认半自动）
# ============================================================================
def download_pdbbind_index():
    """下载 PDBbind v2020 的索引文件（几十KB，包含5000+复合物的PDB ID和pKd数据）"""
    print("\n" + "="*60)
    print("[DOWNLOAD PDBbind] 获取 PDBbind v2020 数据集索引")
    print("="*60)

    # PDBbind 提供的索引文件（包含亲和力数据，最关键！）
    index_urls = [
        # 官方 v2020 refined set 索引（如果链接失效，自动fallback到v2016稳定版）
        "http://www.pdbbind.org.cn/download/INDEX_refined_data.2020",
        "http://www.pdbbind.org.cn/download/INDEX_refined_data.2016",
    ]

    index_file = None
    for url in index_urls:
        fname = os.path.basename(url)
        save_path = os.path.join(PDBBIND_DIR, fname)
        print(f"  尝试: {url}")
        ok, msg = _download_file(url, save_path, timeout=60)
        if ok:
            index_file = save_path
            break
        else:
            print(f"  ❌ {msg}")

    manual_guide = """
========================================================================
📋 如果 PDBbind 自动下载失败（pdbbind.org.cn 国内偶发抽风），请手动下载：

  ① 访问 http://www.pdbbind.org.cn/download.php
  ② 下载 "PDBbind v2020 refined set" (压缩包，约300MB)
  ③ 解压后将里面的 "INDEX_refined_data.2020" 文件复制到：
       data/step6/pdbbind/INDEX_refined_data.2020
  ④ （可选，推荐）将解压后所有 PDB ID 文件夹复制到 data/step6/pdbbind/pdb_files/
       这样可以进行完整的 Level 2 迁移预训练

  💡 如果 v2020 下载慢，可以先下 v2016（小一点，~150MB，也足够用）
========================================================================
"""
    if index_file is None:
        print(manual_guide)
        return None

    print(f"  ✅ 索引文件已下载: {index_file}")
    print(f"  📐 文件大小: {os.path.getsize(index_file)//1024} KB")
    return index_file


def parse_pdbbind_index(index_file):
    """解析PDBbind索引文件，返回 list of dict: [{pdb_id, pkd, resolution, ...}, ...]"""
    if index_file is None or not os.path.exists(index_file):
        return []
    rows = []
    # PDBbind索引格式：前6行是注释，以#开头
    # 数据行格式: PDBID  Resolution  Year  PDBbind_ID  Binding Type  pKd/Ki  ...
    with open(index_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                pdb_id = parts[0].strip().lower()
                if len(pdb_id) != 4:
                    continue
                resol = float(parts[1]) if parts[1] != "NMR" and parts[1] != "N/A" else None
                year = int(parts[2]) if parts[2].isdigit() else None
                # pKd 字段：可能是 -logKd, -logKi, -logIC50，统一取数字部分
                pkd_raw = parts[5]
                try:
                    pkd = float(pkd_raw)
                except ValueError:
                    # 格式可能带前缀如 ~7.2 或 >6.0
                    num_str = ""
                    for c in pkd_raw:
                        if c.isdigit() or c in ".-":
                            num_str += c
                    pkd = float(num_str) if num_str else None
                if pkd is None:
                    continue
                rows.append({
                    "pdb_id": pdb_id,
                    "resolution": resol,
                    "year": year,
                    "pkd": pkd,
                    "binding_type": parts[4] if len(parts) > 4 else "",
                })
            except Exception:
                continue
    print(f"  📊 解析得到 {len(rows)} 条 PDBbind 亲和力数据")
    if rows:
        pkd_arr = [r["pkd"] for r in rows]
        print(f"     pKd 范围: [{min(pkd_arr):.2f}, {max(pkd_arr):.2f}], 均值={sum(pkd_arr)/len(pkd_arr):.2f}")
    return rows


# ============================================================================
# 3. BindingDB ASPA 靶点数据下载
# ============================================================================
def download_bindingdb_aspa():
    """从 BindingDB REST API 下载 ASPA 靶点的小分子结合数据"""
    print("\n" + "="*60)
    print("[DOWNLOAD BindingDB] 搜索 ASPA 靶点结合数据")
    print("="*60)

    # BindingDB API：按Target Name搜索
    search_terms = ["Aspartoacylase", "ASPA"]
    all_rows = []

    for term in search_terms:
        url = ("https://www.bindingdb.org/bind/TargetNameService?"
               + urlencode({"name": term, "format": "json", "rows": 100}))
        print(f"  搜索 TargetName = '{term}'...")
        data, err = _http_get(url, timeout=45, return_json=True)
        if err or not data:
            print(f"    ⚠️  失败: {err}")
            continue
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "results" in data:
            items = data["results"]
        else:
            items = []
        print(f"    命中 {len(items)} 条")
        for it in items:
            all_rows.append({
                "target_name": it.get("targetName", ""),
                "uniprot": it.get("uniprotId", ""),
                "pdb_id": it.get("pdbCode", ""),
                "ligand_smiles": it.get("smiles", ""),
                "ligand_name": it.get("ligandName", ""),
                "ki": it.get("ki", ""),
                "kd": it.get("kd", ""),
                "ic50": it.get("ic50", ""),
            })

    save_path = os.path.join(BINDINGDB_DIR, "aspa_bindingdb.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 保存: {save_path} ({len(all_rows)} 条)")

    if not all_rows:
        manual = """
📋 BindingDB 手动下载指引:
  ① 打开 https://www.bindingdb.org/
  ② Target Name 搜索 "Aspartoacylase"
  ③ 勾选结果，Export → JSON
  ④ 保存到 data/step6/bindingdb_aspa/aspa_bindingdb.json
"""
        print(manual)
    return all_rows


# ============================================================================
# 4. ChEMBL ASPA 靶点数据下载
# ============================================================================
def download_chembl_aspa():
    """从 ChEMBL REST API 下载 ASPA 靶点活性数据"""
    print("\n" + "="*60)
    print("[DOWNLOAD ChEMBL] 搜索 ASPA 靶点 IC50/Ki 数据")
    print("="*60)

    # 第一步：搜索 ASPA 靶点的 ChEMBL ID
    search_url = "https://www.ebi.ac.uk/chembl/api/data/target/search.json?" + \
        urlencode({"q": "Aspartoacylase ASPA"})
    print("  搜索 ChEMBL Target ID...")
    data, err = _http_get(search_url, timeout=45, return_json=True)
    target_ids = []
    if err or not data or "targets" not in data:
        print(f"  ⚠️  靶点搜索失败: {err}，使用预定义 ChEMBL Target ID fallback")
        # ASPA (人类) 的 ChEMBL ID 预查值（若失效可手动更新）
        target_ids = ["CHEMBL3712", "CHEMBL2331066"]
    else:
        for t in data["targets"]:
            tid = t.get("target_chembl_id", "")
            if tid:
                target_ids.append(tid)
        print(f"  命中 Target ID: {target_ids}")

    # 第二步：按 Target ID 拉取活性数据（每个target取前200条）
    activities = []
    for tid in target_ids:
        act_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json?" + \
            urlencode({"target_chembl_id": tid, "limit": 200})
        print(f"  拉取 {tid} 活性数据...", end=" ")
        data, err = _http_get(act_url, timeout=60, return_json=True)
        if err or not data or "activities" not in data:
            print(f"失败: {err}")
            continue
        acts = data["activities"]
        print(f"{len(acts)} 条")
        for a in acts:
            activities.append({
                "target_chembl_id": tid,
                "molecule_chembl_id": a.get("molecule_chembl_id", ""),
                "canonical_smiles": a.get("canonical_smiles", ""),
                "standard_type": a.get("standard_type", ""),  # IC50 / Ki / Kd
                "standard_units": a.get("standard_units", ""),  # nM / uM
                "standard_value": a.get("standard_value", None),
                "pchembl_value": a.get("pchembl_value", None),  # 标准化的 pIC50 / pKi
                "assay_chembl_id": a.get("assay_chembl_id", ""),
            })

    save_path = os.path.join(CHEMBL_DIR, "aspa_chembl_activities.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(activities, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 保存: {save_path} ({len(activities)} 条)")

    if not activities:
        manual = """
📋 ChEMBL 手动下载指引:
  ① 打开 https://www.ebi.ac.uk/chembl/
  ② Targets → 搜索 "Aspartoacylase" → 选择人类 ASPA
  ③ Activities → Download → JSON
  ④ 保存到 data/step6/chembl_aspa/aspa_chembl_activities.json
"""
        print(manual)
    return activities


# ============================================================================
# 5. 统一入口：下载所有可用数据
# ============================================================================
def download_all_related_data(skip_large=False):
    """
    一键下载所有和ASPA相关的训练数据
    skip_large: 跳过 PDBbind 完整压缩包（300MB+），只下索引文件（推荐默认 True）
    返回：下载汇总 dict
    """
    summary = {}

    print("\n" + "#"*60)
    print("# 🚀 开始下载 ASPA RF 训练相关数据")
    print("#"*60)

    # Step 1: RCSB ASPA PDB （最相关，必做）
    try:
        rc_list = download_rcsb_aspa_pdbs()
        summary["rcsb_pdb_count"] = len(rc_list)
        summary["rcsb_pdb_files"] = rc_list
    except Exception as e:
        print(f"  ❌ RCSB 模块异常: {e}")
        summary["rcsb_pdb_count"] = 0
        summary["rcsb_error"] = str(e)

    # Step 2: PDBbind 索引文件（迁移预训练用）
    try:
        idx_file = download_pdbbind_index()
        summary["pdbbind_index_file"] = idx_file
        if idx_file:
            pd_rows = parse_pdbbind_index(idx_file)
            summary["pdbbind_index_count"] = len(pd_rows)
        else:
            summary["pdbbind_index_count"] = 0
    except Exception as e:
        print(f"  ❌ PDBbind 模块异常: {e}")
        summary["pdbbind_error"] = str(e)

    # Step 3: BindingDB ASPA 数据
    try:
        bd_rows = download_bindingdb_aspa()
        summary["bindingdb_rows"] = len(bd_rows)
    except Exception as e:
        print(f"  ❌ BindingDB 模块异常: {e}")
        summary["bindingdb_error"] = str(e)

    # Step 4: ChEMBL ASPA 数据
    try:
        ch_rows = download_chembl_aspa()
        summary["chembl_rows"] = len(ch_rows)
    except Exception as e:
        print(f"  ❌ ChEMBL 模块异常: {e}")
        summary["chembl_error"] = str(e)

    # 汇总打印
    print("\n" + "="*60)
    print("📊 数据下载汇总")
    print("="*60)
    print(f"  RCSB ASPA PDB:        {summary.get('rcsb_pdb_count', 0)} 个结构")
    print(f"  PDBbind 索引条目:      {summary.get('pdbbind_index_count', 0)} 条复合物")
    print(f"  BindingDB ASPA 条目:   {summary.get('bindingdb_rows', 0)} 条")
    print(f"  ChEMBL ASPA 活性数据:  {summary.get('chembl_rows', 0)} 条")
    print("\n" + "="*60)

    return summary


if __name__ == "__main__":
    # 直接运行该脚本 = 启动下载
    summary = download_all_related_data(skip_large=True)
    summary_path = os.path.join(DATA_DIR, "download_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n✅ 下载汇总已保存: {summary_path}")
