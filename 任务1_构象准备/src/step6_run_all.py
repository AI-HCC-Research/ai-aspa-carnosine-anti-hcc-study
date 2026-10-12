"""
ASPA RF 构象打分系统 - 一键运行入口 (步骤六)
======================================
用法：
  python src/step6_run_all.py                  # 用预训练模型推理 (默认, 保证结果复现)
  python src/step6_run_all.py --retrain        # 从头重新训练 Level1 + Level2
  python src/step6_run_all.py --retrain --no-download  # 重训但跳过外部下载
  python src/step6_run_all.py --download-only  # 只下载数据不训练
"""

import os
import sys

# --- 路径准备: 本文件位于 src/, 同目录模块与可选 _pkgs 依赖 ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "step6")

_LOCAL_PKGS = os.path.join(SRC_DIR, "_pkgs")
if os.path.isdir(_LOCAL_PKGS) and _LOCAL_PKGS not in sys.path:
    sys.path.insert(0, _LOCAL_PKGS)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# ----------------------------------------------------------------

import argparse


def main():
    parser = argparse.ArgumentParser(description="ASPA RF 构象打分系统")
    parser.add_argument("--no-download", action="store_true",
                        help="跳过外部数据下载，只用本地数据训练（断网也能跑 baseline）")
    parser.add_argument("--download-only", action="store_true",
                        help="只下载所有相关数据，不训练模型")
    parser.add_argument("--retrain", action="store_true",
                        help="从头重新训练 Level1 + Level2 (默认: 用预训练模型推理)")
    args = parser.parse_args()

    if args.download_only:
        print("📥 只运行数据下载模块...")
        from step6_data_downloader import download_all_related_data
        summary = download_all_related_data(skip_large=True)
        summary_path = os.path.join(DATA_DIR, "download_summary.json")
        import json
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n✅ 下载完成！汇总保存在: {summary_path}")
        return 0

    # 否则跑完整流程
    from step6_rf_trainer import run_full_pipeline
    ok = run_full_pipeline(run_download=not args.no_download,
                           retrain=args.retrain)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
