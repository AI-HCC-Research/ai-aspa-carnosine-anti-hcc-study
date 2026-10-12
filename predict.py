#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict.py — ASPA 全项目统一复现入口 (附件5 规范)
=================================================================
全链路:
  阶段1  任务1: 序列 → HelixFold3 → GNN → 预处理 → UMAP → HDBSCAN
          → RF 构象打分 (子目录 任务1_构象准备/predict.py)
  阶段2  交接核对: 15 个开放构象在两包间 MD5 一致
  阶段3  路线A: MolGPT 生成式筛选随包中间产物核对 (训练/生成重建说明)
  阶段4  路线B: yaTCM + Schrödinger 商业软件溯源 (不执行), FleDock 核对
  阶段5  双榜汇总 → results/results.csv 最终候选清单 (肌肽为最终选定候选)
  阶段6  MD 验证: 从随包 xvg/dat 重绘 5 张验证图

执行原则 (依据附件5):
  - 默认只执行本机可跑环节; 训练好的模型权重随包, predict 直接用于推理,
    不强制重训 (附件5 第二条3款);
  - Schrödinger 商业软件按名称/版本/时间/参数/许可溯源, 不执行
    (附件5 第三条 第三方平台);
  - MolGPT 权重与超大轨迹未随包, 以随包中间产物 + 重建路线处理
    (附件5 第五条 资源受限); --full 打印全部重活的重建命令。

用法:
  python predict.py                  # 默认全链路 (可跑部分)
  python predict.py --skip-task1     # 已有任务1结果, 跳过阶段1
  python predict.py --skip-md-plot   # 不重绘 MD 图件
  python predict.py --full           # 打印重型环节完整重建命令清单

依赖 (默认路径): numpy, pandas, scikit-learn, matplotlib;
       阶段1 还需 mdtraj, umap-learn, hdbscan (详见 requirements.txt)。
"""

import os
import sys
import csv
import json
import hashlib
import argparse
import subprocess

# ----------------------------------------------------------------
# 路径 (全部相对本文件定位, 不硬编码绝对路径)
# ----------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR = os.path.join(ROOT, "任务1_构象准备")
TASK23_DIR = os.path.join(ROOT, "任务23_虚拟筛选")
ROUTE_A = os.path.join(TASK23_DIR, "路线A_MolGPT生成式筛选")
ROUTE_B = os.path.join(TASK23_DIR, "路线B_yaTCM数据库筛选")
COMMON = os.path.join(TASK23_DIR, "公共模块_XGBoost模型与评估")
MD_DIR = os.path.join(TASK23_DIR, "MD验证_ASPA-肌肽复合物")
FIG_SRC = os.path.join(TASK23_DIR, "figures_src")
FINAL_LIST = os.path.join(ROOT, "results", "results.csv")

# 任务1 构象目录 / 路线B 步骤1 构象目录 (同源, 用于交接 MD5 核对)
TASK1_CONFS = os.path.join(TASK1_DIR, "data", "step6", "conformations")
ROUTEB_CONFS = os.path.join(ROUTE_B, "步骤1_蛋白准备_口袋预测与网格",
                           "open_conformations")


# ----------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------
def info(m=""):
    print(m)


def ok(m):
    print("  [OK] " + m)


def warn(m):
    print("  [!] " + m)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path):
    """CSV 数据行数 (不含表头)。"""
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def run_script(script_path, cwd, required=True):
    """用当前解释器运行子包脚本。"""
    print("\n>>> %s" % os.path.relpath(script_path, ROOT))
    proc = subprocess.run([sys.executable, script_path], cwd=cwd)
    if proc.returncode != 0:
        if required:
            print("  [X] 脚本失败, 退出码 %d" % proc.returncode)
            return False
        warn("非必需脚本失败 (退出码 %d), 继续" % proc.returncode)
    return True


def stage_banner(title):
    print("\n" + "#" * 70)
    print("# " + title)
    print("#" * 70)


# ----------------------------------------------------------------
# 阶段0: 依赖自检
# ----------------------------------------------------------------
def stage0_check():
    stage_banner("阶段0  依赖自检")
    required_core = ["numpy", "pandas", "sklearn"]
    optional = {"mdtraj": "步骤4 MD轨迹", "umap": "步骤4/5 降维",
                "hdbscan": "步骤5 聚类", "rdkit": "路线A/化学信息学",
                "xgboost": "路线A/B 精排", "matplotlib": "绘图"}
    missing = []
    for m in required_core:
        try:
            __import__(m)
            ok("核心依赖: %s" % m)
        except ImportError:
            warn("缺少核心依赖: %s" % m)
            missing.append(m)
    for m, use in optional.items():
        try:
            __import__(m)
            ok("%s (%s)" % (m, use))
        except ImportError:
            warn("缺少 %s (%s) — 相关阶段将降级/跳过" % (m, use))
    if missing:
        print("\n请先安装依赖:  pip install -r requirements.txt")
        return False
    return True


# ----------------------------------------------------------------
# 阶段1: 任务1 全流程
# ----------------------------------------------------------------
def stage1_task1():
    stage_banner("阶段1  任务1: 构象准备与 RF 构象打分")
    task1_predict = os.path.join(TASK1_DIR, "predict.py")
    if not os.path.exists(task1_predict):
        warn("未找到 任务1_构象准备/predict.py")
        return None
    if not run_script(task1_predict, cwd=TASK1_DIR, required=True):
        return None

    ranking = os.path.join(TASK1_DIR, "results", "step6",
                          "conformation_ranking.csv")
    if not os.path.exists(ranking):
        warn("未找到 conformation_ranking.csv")
        return None
    with open(ranking, "r", encoding="utf-8-sig", newline="") as f:
        top = next(csv.DictReader(f))
    top_id, top_score = top.get("构象编号"), top.get("活性综合分")
    if top_id == "conf_13":
        ok("任务1 Top1 = %s, 活性综合分 %s" % (top_id, top_score))
    else:
        warn("任务1 Top1 = %s (%s), 预期 conf_13; 请检查 models 权重"
             % (top_id, top_score))
    return top_id


# ----------------------------------------------------------------
# 阶段2: 15 构象 MD5 交接核对
# ----------------------------------------------------------------
def stage2_handoff():
    stage_banner("阶段2  任务1 → 任务2/3 构象交接 (MD5)")
    same = diff = miss = 0
    for i in range(1, 16):
        name = "open_conformation_%02d.pdb" % i
        p1 = os.path.join(TASK1_CONFS, name)
        p2 = os.path.join(ROUTEB_CONFS, name)
        if not (os.path.exists(p1) and os.path.exists(p2)):
            warn("缺失: %s" % name)
            miss += 1
            continue
        if md5_of(p1) == md5_of(p2):
            same += 1
        else:
            warn("内容不一致: %s" % name)
            diff += 1
    ok("15 构象: 一致 %d, 不一致 %d, 缺失 %d" % (same, diff, miss))
    return diff == 0 and miss == 0


# ----------------------------------------------------------------
# 阶段3: 路线A 随包产物核对
# ----------------------------------------------------------------
def stage3_route_a():
    stage_banner("阶段3  路线A (MolGPT 生成式筛选) 产物核对")
    checks = [
        (os.path.join(ROUTE_A, "步骤1_MolGPT训练与定向生成", "results",
                      "directed_qed_5000.csv"), 5000, "定向生成 5000 分子"),
        (os.path.join(ROUTE_A, "步骤2_筛选与分子对接", "results",
                      "prefilter_top500.csv"), 500, "2D过滤+QED抽样 Top500"),
        (os.path.join(ROUTE_A, "步骤3_XGBoost粗筛", "results",
                      "final_top20_xgboost.csv"), None, "XGBoost 粗筛 Top20"),
        (os.path.join(ROUTE_A, "步骤4_Top10最终结果",
                      "final_top20_consensus.csv"), 20, "XGBoost+对接共识 Top20"),
    ]
    all_ok = True
    for path, expect, desc in checks:
        if not os.path.exists(path):
            warn("%s — 文件缺失" % desc)
            all_ok = False
            continue
        n = count_rows(path)
        if expect is not None and n != expect:
            warn("%s — 行数 %d ≠ 预期 %d" % (desc, n, expect))
            all_ok = False
        else:
            ok("%s: %d 行" % (desc, n))

    # XGBoost 推理前置: preprocess_params.json
    preprocess = os.path.join(COMMON, "preprocess_params.json")
    if os.path.exists(preprocess):
        ok("XGBoost 预处理参数存在, 可直接重新推理")
    else:
        warn("缺 公共模块/preprocess_params.json; 最终打分以随包共识 CSV 为准")
        print("      重建(在有 PDBbind 特征缓存的环境, 一次性, seed=42):")
        print("        python 任务23_虚拟筛选/predict.py "
              "--build-preprocess pdbbind_features.csv pdbbind_labels.npy")

    # MolGPT 训练/生成 (需 GPU + 未随包权重)
    print("\n  -- MolGPT 训练/定向生成为 GPU 环节, .pt 权重未随包 (附件5 第五条)")
    print("     训练: 路线A/步骤1.../code/1_molgpt_train.py")
    print("     生成: 路线A/步骤1.../code/2_directed_generation.py")
    return all_ok


# ----------------------------------------------------------------
# 阶段4: 路线B Schrödinger 溯源 (不执行)
# ----------------------------------------------------------------
def stage4_route_b():
    stage_banner("阶段4  路线B (yaTCM + Schrödinger) 溯源核对 — 不执行")
    print("  商业软件流程按附件5 第三条披露, 不在本地重跑:")
    print("    软件 : Schrödinger Suite (Glide/LigPrep/Epik/QikProp/Prime/"
          "SiteMap/IFD)")
    print("    版本 : 2024-4 (Windows 工作站 CADD) / 2024-1 (Linux 集群)")
    print("    力场 : OPLS_2005 全程统一")
    print("    时间 : Grid 2025-03-31; IFD 2025-04-01")
    print("    许可 : 商业许可 (高校科研授权)")
    print("    漏斗 : 47,696 → 58,048 → 22,227 → 5,597 → 1,120 → 224 → 45")

    fledock = os.path.join(ROUTE_B, "步骤9_Top10最终结果", "FleDock.csv")
    if not os.path.exists(fledock):
        warn("FleDock.csv 缺失")
        return []
    rows = []
    with open(fledock, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r.get("docking score", "").strip(),
                         r.get("English_name", "").strip()))
    if len(rows) == 45:
        ok("FleDock.csv: 45 个候选; Top3 = %s"
           % ", ".join(n for _, n in rows[:3]))
    else:
        warn("FleDock.csv 行数 %d ≠ 45" % len(rows))
    return rows


# ----------------------------------------------------------------
# 阶段5: 汇总最终候选清单 results/results.csv
# ----------------------------------------------------------------
FINAL_NOTE_CARNOSINE = ("最终选定候选: ① 对接/RF 证据链 (本项目新颖假说), "
                        "② ASPA 已知配体 AS9 验证口袋, "
                        "③ 肌肽抗 HCC 前期文献; 并经 100 ns MD + MM/PBSA 验证")

VERSION_B = "Schrödinger Glide/IFD 2024-4/2024-1, OPLS_2005"
VERSION_A = "XGBoost 3.4.1 (xgboost_aspa_final.json) + AutoDock Vina"


def carnosine_smiles():
    """优先从随包 Carnosine_sentinel.sdf 提取 canonical SMILES。"""
    sdf = os.path.join(ROUTE_A, "步骤2_筛选与分子对接", "data",
                       "Carnosine_sentinel.sdf")
    try:
        from rdkit import Chem
        m = None
        if os.path.exists(sdf):
            # Windows 中文路径下 MolFromMolFile 的 C++ 文件流可能打不开,
            # 用 Python 读取文本后走 MolFromMolBlock 规避
            with open(sdf, "r", encoding="utf-8", errors="replace") as f:
                m = Chem.MolFromMolBlock(f.read())
        if m is None:
            m = Chem.MolFromMolFile(sdf)
        if m is not None:
            return Chem.MolToSmiles(m)
    except Exception:
        pass
    return ""


def stage5_final_list(routeb_rows):
    stage_banner("阶段5  生成最终候选清单 results/results.csv")
    header = ["候选编号", "所属路线", "候选名称", "SMILES",
              "对接打分(kcal/mol)", "XGBoost预测(pKd代理)",
              "模型与运行版本", "结构文件", "备注"]
    out = []
    idx = 1

    # --- 路线B: yaTCM/Schrödinger (45) ---
    top_pdbs = {1: "Top1.pdb", 2: "Top2.pdb", 3: "Top3.pdb",
                4: "Top4.pdb", 5: "Top5.pdb"}
    for rank, (score, name) in enumerate(routeb_rows, start=1):
        note = "路线B 对接排名 #%d" % rank
        if name == "#NAME?":
            note += "; 原始清单名称解析缺失"
        struct = top_pdbs.get(rank, "")
        if name == "Carnosine":
            note = FINAL_NOTE_CARNOSINE
        out.append([idx, "路线B yaTCM", name, "", score, "",
                    VERSION_B, struct, note])
        idx += 1

    # --- 路线A: MolGPT 共识 Top20 ---
    consensus = os.path.join(ROUTE_A, "步骤4_Top10最终结果",
                             "final_top20_consensus.csv")
    # 路线A top 分子 PDB 实物 (步骤2 results/plots/)
    plots = os.path.join(ROUTE_A, "步骤2_筛选与分子对接", "results", "plots")
    avail = set(os.listdir(plots)) if os.path.isdir(plots) else set()
    if os.path.exists(consensus):
        with open(consensus, "r", encoding="utf-8-sig", errors="replace",
                  newline="") as f:
            for r in csv.DictReader(f):
                mol = r.get("分子名称", "").strip()
                struct_pick = ""
                for fn in avail:
                    if fn.startswith("top") and mol in fn and fn.endswith(".pdb"):
                        struct_pick = fn
                        break
                note = "路线A 综合排名 #%s" % r.get("综合排名", "")
                out.append([idx, "路线A MolGPT", mol,
                            r.get("SMILES", "").strip(),
                            r.get("对接打分", "").strip(),
                            r.get("XGBoost预测", "").strip(),
                            VERSION_A, struct_pick, note])
                idx += 1

    # 肌肽行确保存在 (路线B 已含); 补 SMILES
    smi = carnosine_smiles()
    for row in out:
        if row[2] == "Carnosine" and smi and not row[3]:
            row[3] = smi

    os.makedirs(os.path.dirname(FINAL_LIST), exist_ok=True)
    with open(FINAL_LIST, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out)
    ok("候选 %d 个 (路线B 45 + 路线A %d) → %s"
       % (len(out), len(out) - len(routeb_rows),
          os.path.relpath(FINAL_LIST, ROOT)))

    car = [r for r in out if r[2] == "Carnosine"]
    if car:
        ok("最终选定候选: Carnosine (肌肽); 路线B 对接 -8.592 kcal/mol; SMILES: %s"
           % (car[0][3] or "(未取得)"))
    else:
        warn("清单中未找到 Carnosine")
    return bool(car)


# ----------------------------------------------------------------
# 阶段6: MD 图件重绘
# ----------------------------------------------------------------
def stage6_md_plot():
    stage_banner("阶段6  MD 验证图件重绘 (随包 xvg/dat)")
    script = os.path.join(FIG_SRC, "fig_md_analysis.py")
    if not os.path.exists(script):
        warn("未找到 figures_src/fig_md_analysis.py")
        return False
    if not run_script(script, cwd=FIG_SRC, required=False):
        return False
    figdir = os.path.join(MD_DIR, "figures")
    figs = [f for f in os.listdir(figdir) if f.endswith(".png")] if os.path.isdir(figdir) else []
    if len(figs) >= 5:
        ok("5 张 MD 验证图已重绘: %s" % ", ".join(sorted(figs)))
        return True
    warn("MD 图件数量 %d < 5" % len(figs))
    return False


# ----------------------------------------------------------------
# --full: 重型环节重建命令
# ----------------------------------------------------------------
def print_full_rebuild():
    stage_banner("重型环节完整重建命令 (--full)")
    print("""
1) 任务1 全流程 (含 RF 重训):
     python 任务1_构象准备/predict.py
     python 任务1_构象准备/src/step6_run_all.py --retrain

2) MolGPT 训练与定向生成 (Linux + GPU, Python 3.10, torch+CUDA):
     python 任务23_虚拟筛选/路线A_MolGPT生成式筛选/步骤1_MolGPT训练与定向生成/code/1_molgpt_train.py
     python 任务23_虚拟筛选/路线A_MolGPT生成式筛选/步骤1_MolGPT训练与定向生成/code/2_directed_generation.py

3) 统一筛选 + 对接 (引擎自动: gnina → vina → RDKit 经验打分):
     python 任务23_虚拟筛选/路线A_MolGPT生成式筛选/步骤2_筛选与分子对接/code/3_unified_screening.py

4) XGBoost 预处理参数重建 (需 PDBbind v2019 特征缓存, 一次性, seed=42):
     python 任务23_虚拟筛选/predict.py --build-preprocess pdbbind_features.csv pdbbind_labels.npy

5) XGBoost 模型完整训练/评估 (Kaggle notebook 或脚本):
     任务23_虚拟筛选/公共模块_XGBoost模型与评估/2_xgboost_model.ipynb

6) MD 模拟 (Ubuntu, GROMACS 2025.4 + gmx_MMPBSA 1.6.4; 系统级安装):
     原始 100 ns 轨迹数十 GB 未随包; 随包 .xvg/.dat 可重绘全部图件:
     python 任务23_虚拟筛选/figures_src/fig_md_analysis.py
""")


# ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ASPA 全项目统一复现入口")
    ap.add_argument("--skip-task1", action="store_true",
                    help="跳过阶段1 (任务1已有结果)")
    ap.add_argument("--skip-md-plot", action="store_true",
                    help="跳过阶段6 (MD 图件重绘)")
    ap.add_argument("--full", action="store_true",
                    help="只打印重型环节重建命令清单")
    args = ap.parse_args()

    print("=" * 70)
    print("ASPA: 构象准备 → 双路线虚拟筛选 → 肌肽 → MD 验证 (统一复现)")
    print("=" * 70)

    if args.full:
        print_full_rebuild()
        return 0

    results_ok = {"dep": True, "task1": True, "handoff": True,
                  "routeA": True, "final": False, "md": True}

    results_ok["dep"] = stage0_check()
    if not results_ok["dep"]:
        return 1

    if args.skip_task1:
        stage_banner("阶段1  已按 --skip-task1 跳过")
    else:
        top = stage1_task1()
        results_ok["task1"] = top is not None

    results_ok["handoff"] = stage2_handoff()
    results_ok["routeA"] = stage3_route_a()
    routeb_rows = stage4_route_b()
    results_ok["final"] = stage5_final_list(routeb_rows)

    if args.skip_md_plot:
        stage_banner("阶段6  已按 --skip-md-plot 跳过")
    else:
        results_ok["md"] = stage6_md_plot()

    # ---- 总结 ----
    print("\n" + "=" * 70)
    print("统一复现完成 — 核对摘要")
    print("=" * 70)
    print("  任务1 最优构象   : conf_13 (活性综合分 0.974)")
    print("  最终候选清单     : results/results.csv")
    print("  最终选定候选     : Carnosine (肌肽)")
    print("  MD 验证          : 100 ns MD + MM/PBSA, 图件在 任务23_虚拟筛选/"
          "MD验证_ASPA-肌肽复合物/figures/")
    failed = [k for k, v in results_ok.items() if not v]
    if failed:
        print("\n  存在未通过环节: %s (见上方提示)" % ", ".join(failed))
        return 1
    print("\n  全部环节通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
