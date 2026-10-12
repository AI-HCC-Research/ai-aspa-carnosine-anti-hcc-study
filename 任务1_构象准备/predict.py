# -*- coding: utf-8 -*-
"""
ASPA 任务一 一键复现入口（附件5 规范）
============================================================
链路: 步骤一(HelixFold3 结果解析) → 步骤二(GNN 口袋开放性过滤)
      → 步骤三(结构预处理/加氢) → 步骤四(MD轨迹 UMAP)
      → 步骤五(HDBSCAN 聚类) → 步骤六(RF 打分排序)

如实说明:
  - 结构预测本身在 HelixFold3 云端完成, 步骤一本地脚本只负责解析
    平台已下载的结果 (results/step1/ranked_structures/), 预测任务的
    云端复现方式见 README.md「分步骤复现」;
  - 步骤二为按相同筛选准则编写的复现实现 (纯 numpy 小型 GNN);
  - 步骤三依赖 PDBFixer/OpenMM, 产出 ASPA_ready_pdbfixer.pdb
    (5009 原子), 不会覆盖归档 ASPA_ready.pdb (4730 原子);
    若本机缺少 openmm/pdbfixer, 该步失败不中断流程 (步骤四使用
    已提供的轨迹文件, 不依赖步骤三产物);
  - 步骤六默认用 models/ 下预训练模型推理 (保证结果复现),
    加 --retrain 从头重训 (结果可能因 numpy 版本微有差异);
    --with-download 允许联网下载 RCSB 增强数据 (仅 --retrain 时生效)。

最终产物:
  results/step6/conformation_ranking.csv   15 构象活性排名

用法:
  python predict.py                       # 默认: 预训练模型推理
  python predict.py --retrain             # 从头重训 RF
  python predict.py --retrain --with-download
"""

import os
import sys
import subprocess
import argparse


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")


def run_step(script, extra_args, required=True):
    """用当前解释器运行 src/ 下脚本, 工作目录为项目根"""
    path = os.path.join(SRC, script)
    cmd = [sys.executable, path] + extra_args

    print("\n" + "#" * 70)
    print("# ▶ 运行: %s %s" % (script, " ".join(extra_args)))
    print("#" * 70)

    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        if required:
            print("\n❌ 步骤失败 (退出码 %d): %s" % (proc.returncode, script))
            sys.exit(proc.returncode)
        else:
            print("\n⚠️  非必需步骤失败, 继续后续流程: %s" % script)


def main():
    parser = argparse.ArgumentParser(description="ASPA 任务一 一键复现 (步骤一→六)")
    parser.add_argument("--with-download", action="store_true",
                        help="步骤六允许联网下载 RCSB 增强数据 (仅 --retrain 时生效)")
    parser.add_argument("--retrain", action="store_true",
                        help="步骤六从头重新训练 RF (默认: 用预训练模型推理)")
    args = parser.parse_args()

    print("=" * 70)
    print("🧬 ASPA 任务一: 从序列到最优活性构象全流程 (步骤一→六)")
    print("=" * 70)

    # ---------- 步骤一: HelixFold3 结果解析与置信度分析 ----------
    run_step("step1_structure_prediction.py", [], required=True)

    # ---------- 步骤二: GNN 口袋开放性前置过滤 ----------
    run_step("step2_gnn_conformation_filter.py", [], required=True)

    # ---------- 步骤三: 结构预处理 (PDBFixer, 依赖缺失不致命) ----------
    run_step("step3_structure_preprocessing.py", [], required=False)

    # ---------- 步骤四: MD 轨迹 UMAP 构象景观 ----------
    run_step("step4_umap_analysis.py", [], required=True)

    # ---------- 步骤五: HDBSCAN 口袋聚类 ----------
    run_step("step5_hdbscan_clustering.py", [], required=True)

    # ---------- 步骤六: RF 推理 + 构象打分 ----------
    step6_args = []
    if not args.retrain:
        step6_args.append("--no-download")  # 推理模式不需要下载
    elif not args.with_download:
        step6_args.append("--no-download")
    if args.retrain:
        step6_args.append("--retrain")
    run_step("step6_run_all.py", step6_args, required=True)

    # ---------- 读取并报告最终结果 (从 CSV 动态读取, 不硬编码) ----------
    final_csv = os.path.join(ROOT, "results", "step6", "conformation_ranking.csv")
    print("\n" + "=" * 70)
    if os.path.exists(final_csv):
        import csv
        with open(final_csv, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            top = next(reader)
        print("🎉 全部完成! 最终结果: results/step6/conformation_ranking.csv")
        print("   Top1 最优活性构象: %s (活性综合分 %s)"
              % (top["构象编号"], top["活性综合分"]))
    else:
        print("❌ 未找到最终结果文件: %s" % final_csv)
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
