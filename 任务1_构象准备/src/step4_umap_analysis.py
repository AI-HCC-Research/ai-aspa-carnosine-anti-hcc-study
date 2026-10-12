# -*- coding: utf-8 -*-
"""
步骤四: ASPA-肌肽复合物 MD 轨迹 UMAP 构象空间分析
============================================================
技术路线: MD 轨迹(.xtc) + 拓扑(.gro)
          -> mdtraj 提取每帧 CA 原子坐标并展平
          -> UMAP 2D 降维 (n_neighbors=15, min_dist=0.1, random_state=42)
          -> matplotlib 构象景观图 (每个点=一个模拟帧, 颜色=帧索引/时间)

输入 (默认):
  data/step4/biobb.MDsetup.xtc   GROMACS 轨迹
  data/step4/biobb.MDsetup.gro   拓扑
输出:
  results/step4/UMAP_ASPA_trajectory.png  构象景观图
  results/step4/umap_embedding.csv        每帧 UMAP1/UMAP2 坐标 (供复核)

运行:
  python src/step4_umap_analysis.py
  python src/step4_umap_analysis.py --xtc <轨迹> --gro <拓扑>
"""

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _ascii_path(path):
    """返回可被 mdtraj C 扩展接受的 ASCII 路径.
    Windows 上 mdtraj 的 xtc/gro 解析器只能处理 ASCII 路径; 若路径含中文/非
    ASCII 字符, 则在系统临时目录 (纯 ASCII) 生成同名副本供 mdtraj 读取.
    其他平台直接返回原路径."""
    if sys.platform != 'win32':
        return path
    try:
        path.encode('ascii')
        return path
    except UnicodeEncodeError:
        pass
    # GetShortPathName 在中文 Windows 上不一定输出 ASCII 短名, 先试一次
    import ctypes
    from ctypes import wintypes
    buf = ctypes.create_unicode_buffer(1024)
    get_short = ctypes.windll.kernel32.GetShortPathNameW
    get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    get_short.restype = wintypes.DWORD
    if get_short(path, buf, 1024) > 0:
        try:
            buf.value.encode('ascii')
            return buf.value
        except UnicodeEncodeError:
            pass
    # 退化为在系统临时目录 (纯 ASCII) 生成副本; 保留原扩展名,
    # mdtraj 靠扩展名识别格式, 不能丢后缀
    import shutil
    import tempfile
    tmp = tempfile.gettempdir()
    dst = os.path.join(tmp, 'aspa_' + os.path.basename(path))
    shutil.copyfile(path, dst)
    return dst


def run_umap(xtc_path, gro_path, out_png, out_csv):
    """加载轨迹 -> CA 展平 -> UMAP 2D -> 图 + 坐标表"""
    import mdtraj as md
    from umap import UMAP

    print("[step4] 加载轨迹: %s (拓扑: %s)" % (xtc_path, gro_path))
    # mdtraj 的 C 扩展仅接受 ASCII 路径; 中文路径下转成 8.3 短名
    xtc_ascii = _ascii_path(xtc_path)
    gro_ascii = _ascii_path(gro_path)
    traj = md.load(xtc_ascii, top=gro_ascii)

    ca_indices = traj.topology.select("name CA")
    coords = traj.xyz[:, ca_indices, :]
    n_frames = coords.shape[0]
    coords_flat = coords.reshape(n_frames, -1)
    print("[step4] 帧数=%d, CA 原子=%d, 展平特征=%d 维"
          % (n_frames, len(ca_indices), coords_flat.shape[1]))

    # 固定随机种子, 保证可复现
    reducer = UMAP(n_neighbors=15, min_dist=0.1,
                   n_components=2, random_state=42)
    embedding = reducer.fit_transform(coords_flat)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # 画构象景观图: 颜色 = 帧索引(时间), 紫(早期) -> 黄(晚期)
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1],
                     c=range(n_frames), cmap='viridis',
                     s=10, alpha=0.7)
    plt.colorbar(sc, label='Frame Index (Time)')
    plt.title('UMAP Projection of ASPA MD Trajectory', fontsize=14)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

    # 保存坐标表供复核/再绘图
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'umap1', 'umap2'])
        for i in range(n_frames):
            writer.writerow([i, embedding[i, 0], embedding[i, 1]])

    print("[step4] ✅ 构象景观图: %s" % out_png)
    print("[step4] ✅ 坐标表: %s" % out_csv)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    parser = argparse.ArgumentParser(description="步骤四 MD 轨迹 UMAP 分析")
    parser.add_argument('--xtc',
                        default=os.path.join(root, 'data', 'step4', 'biobb.MDsetup.xtc'),
                        help='GROMACS 轨迹文件 (.xtc)')
    parser.add_argument('--gro',
                        default=os.path.join(root, 'data', 'step4', 'biobb.MDsetup.gro'),
                        help='拓扑文件 (.gro)')
    parser.add_argument('--out',
                        default=os.path.join(root, 'results', 'step4', 'UMAP_ASPA_trajectory.png'),
                        help='输出图片路径')
    parser.add_argument('--embedding',
                        default=os.path.join(root, 'results', 'step4', 'umap_embedding.csv'),
                        help='输出 UMAP 坐标表路径')
    args = parser.parse_args()

    run_umap(args.xtc, args.gro, args.out, args.embedding)


if __name__ == '__main__':
    main()
