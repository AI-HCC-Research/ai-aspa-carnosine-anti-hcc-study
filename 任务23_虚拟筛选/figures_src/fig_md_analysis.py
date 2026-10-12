#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fig_md_analysis.py — ASPA-肌肽复合物分子动力学(MD)验证结果绘图
=================================================================
功能: 读取 GROMACS 100 ns 轨迹分析输出(.xvg)与 MM/PBSA 结合自由能结果(.dat),
      重绘申报材料中的 MD 验证组图:
        1) RMSD 曲线(蛋白骨架/配体)     — pic.rmsd.xvg, pic.rmsd2.xvg, pic.rmsd3.xvg
        2) RMSF 残基波动               — pic.rmsf.xvg
        3) SASA / 回旋半径Rg / 氢键数  — pic.area.xvg, pic.gyrate.xvg, pic.hbnum.xvg
        4) MM/PBSA 平均结合自由能      — FINAL_RESULTS_MMPBSA.dat
        5) 残基能量分解                — FINAL_DECOMP_MMPBSA.dat

数据来源: ASPA-Carnosine 复合物 100 ns GROMACS 模拟 (AMBER力场流程),
         MM/PBSA 由 gmx_MMPBSA 计算。原始文件位于 ../md_analysis/data/

运行:
    cd 提交包_ASPA代码
    python figures_src/fig_md_analysis.py
输出:
    md_analysis/figures/md_rmsd.png / md_rmsf.png / md_hbonds.png
    md_analysis/figures/md_binding_energy.png / md_residue_decomp.png

依赖: numpy, matplotlib (无需 RDKit/torch)
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- 路径 ----
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '..', 'MD验证_ASPA-肌肽复合物', 'data')
OUT_DIR = os.path.join(HERE, '..', 'MD验证_ASPA-肌肽复合物', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
# 中文显示(Windows: 黑体; Linux/Mac 自动回退)
for _f in ['SimHei', 'Microsoft YaHei', 'Noto Sans CJK SC', 'Arial Unicode MS']:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams['font.sans-serif'] = [_f]
        break
    except Exception:
        continue


def read_xvg(path):
    """读取 GROMACS .xvg 文件, 返回 (x, y) 或 (x, [y1,y2,...])。
    跳过以 '#'/'@' 开头的注释与图例行。"""
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('#', '@')):
                continue
            parts = line.split()
            try:
                rows.append([float(v) for v in parts])
            except ValueError:
                continue
    arr = np.array(rows)
    return arr[:, 0], [arr[:, i] for i in range(1, arr.shape[1])]


def smooth(y, window=50):
    """滑动平均平滑(仅用于展示趋势)"""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode='valid')


# ============================================================
# 图1: RMSD (蛋白骨架 + 配体)
# ============================================================
def plot_rmsd():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    series = [
        ('pic.rmsd.xvg',  'Protein backbone', '#1f77b4'),
        ('pic.rmsd2.xvg', 'Ligand (Carnosine)', '#d62728'),
        ('pic.rmsd3.xvg', 'Complex', '#2ca02c'),
    ]
    plotted = False
    for fname, label, color in series:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  [跳过] 缺少 {fname}")
            continue
        t, ys = read_xvg(path)
        y = ys[0]
        t_ns = t / 1000.0 if t.max() > 1000 else t  # ps -> ns 自适应
        ax.plot(t_ns, y, color=color, alpha=0.25, lw=0.6)
        ys_s = smooth(y)
        ax.plot(t_ns[:len(ys_s)], ys_s, color=color, lw=1.8, label=label)
        plotted = True
    if not plotted:
        return
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('RMSD (nm)')
    ax.set_title('MD Simulation — RMSD (100 ns)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'md_rmsd.png'), dpi=300)
    plt.close(fig)
    print("  [输出] md_rmsd.png")


# ============================================================
# 图2: RMSF 残基波动
# ============================================================
def plot_rmsf():
    path = os.path.join(DATA_DIR, 'pic.rmsf.xvg')
    if not os.path.exists(path):
        print("  [跳过] 缺少 pic.rmsf.xvg")
        return
    res, ys = read_xvg(path)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(res, ys[0], color='#1f77b4', lw=1.5)
    ax.fill_between(res, ys[0], alpha=0.2, color='#1f77b4')
    ax.set_xlabel('Residue index')
    ax.set_ylabel('RMSF (nm)')
    ax.set_title('MD Simulation — Per-residue RMSF')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'md_rmsf.png'), dpi=300)
    plt.close(fig)
    print("  [输出] md_rmsf.png")


# ============================================================
# 图3: SASA / Rg / 氢键数 三联图
# ============================================================
def plot_sasa_rg_hbond():
    panels = [
        ('pic.area.xvg',   'SASA',               'SASA (nm$^2$)',      '#2ca02c'),
        ('pic.gyrate.xvg', 'Radius of gyration', 'Rg (nm)',            '#ff7f0e'),
        ('pic.hbnum.xvg',  'Hydrogen bonds',     'Number of H-bonds',  '#9467bd'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=150)
    any_plot = False
    for ax, (fname, title, ylabel, color) in zip(axes, panels):
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f'缺少\n{fname}', ha='center', va='center', transform=ax.transAxes)
            continue
        t, ys = read_xvg(path)
        y = ys[0]
        t_ns = t / 1000.0 if t.max() > 1000 else t
        ax.plot(t_ns, y, color=color, alpha=0.25, lw=0.6)
        ys_s = smooth(y)
        ax.plot(t_ns[:len(ys_s)], ys_s, color=color, lw=1.8)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        any_plot = True
    if not any_plot:
        return
    fig.suptitle('MD Simulation — SASA / Rg / H-bonds', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'md_hbonds.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [输出] md_hbonds.png (SASA/Rg/H-bonds 三联图)")


# ============================================================
# 图4: MM/PBSA 平均结合自由能
# ============================================================
def parse_mmpbsa_results(path):
    """解析 gmx_MMPBSA FINAL_RESULTS_MMPBSA.dat, 提取能量分项 (kcal/mol)"""
    items = {}
    cur_section = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if re.match(r'^[A-Z][A-Za-z /()_-]+:$', s):
                cur_section = s.rstrip(':')
                continue
            m = re.match(r'^([A-Za-z][A-Za-z0-9 _/+-]*?)\s+(-?\d+\.?\d*)\s+', s)
            if m:
                key = m.group(1).strip()
                try:
                    items[key] = float(m.group(2))
                except ValueError:
                    pass
    return items


def plot_binding_energy():
    path = os.path.join(DATA_DIR, 'FINAL_RESULTS_MMPBSA.dat')
    if not os.path.exists(path):
        print("  [跳过] 缺少 FINAL_RESULTS_MMPBSA.dat")
        return
    items = parse_mmpbsa_results(path)
    # 常见能量分项关键词
    want = ['VDWAALS', 'EEL', 'EGB', 'ESURF', 'GGAS', 'GSOLV', 'TOTAL',
            'DELTA G gas', 'DELTA G solv', 'DELTA TOTAL']
    labels, values = [], []
    for k, v in items.items():
        ku = k.upper()
        if any(w.upper() in ku for w in ['VDWAALS', 'EEL', 'EGB', 'ESURF', 'GGAS', 'GSOLV', 'TOTAL', 'DELTA']):
            labels.append(k)
            values.append(v)
    if not labels:  # 解析失败时全部画出
        labels, values = list(items.keys()), list(items.values())
    if not labels:
        print("  [警告] MM/PBSA 结果解析为空")
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    colors = ['#d62728' if v < 0 else '#1f77b4' for v in values]
    ax.bar(range(len(labels)), values, color=colors, alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Energy (kcal/mol)')
    ax.set_title('MM/PBSA Binding Free Energy — ASPA-Carnosine')
    ax.axhline(0, color='k', lw=0.8)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'md_binding_energy.png'), dpi=300)
    plt.close(fig)
    print("  [输出] md_binding_energy.png")


# ============================================================
# 图5: 残基能量分解 (FINAL_DECOMP_MMPBSA.dat)
# ============================================================
def plot_residue_decomp():
    path = os.path.join(DATA_DIR, 'FINAL_DECOMP_MMPBSA.dat')
    if not os.path.exists(path):
        print("  [跳过] 缺少 FINAL_DECOMP_MMPBSA.dat")
        return
    res_names, energies = [], []
    # 只解析 DELTAS 区块的 Total Energy Decomposition (每残基结合贡献);
    # 前面的 Complex/Receptor/Ligand 区块是绝对能量, 不能用于此图
    in_deltas = in_total = False
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if s == 'DELTAS:':
                in_deltas = True
                continue
            if in_deltas and 'Total Energy Decomposition:' in s:
                in_total = True
                continue
            if in_total and 'Sidechain Energy Decomposition:' in s:
                break
            if not in_total or not s or s.startswith(('Residue,', ',Avg.')):
                continue
            parts = s.split(',')
            if len(parts) < 17:
                continue
            # 标签形如 R:A:LYS:103 (类型:链:残基名:编号), 简化为 LYS 103
            rk = parts[0].strip().split(':')
            res_key = '%s %s' % (rk[2], rk[3]) if len(rk) >= 4 else parts[0].strip()
            try:
                # 第17列(索引16) = TOTAL Avg.; 旧代码误取索引15 (Non-Polar Std.Err≈0)
                total_val = float(parts[16].strip())
            except ValueError:
                continue
            res_names.append(res_key)
            energies.append(total_val)
    if not energies:
        print("  [警告] 残基分解能量解析为空, 请检查 FINAL_DECOMP_MMPBSA.dat 格式")
        return
    # 取贡献最强的前 20 个残基(最负=最强结合贡献)
    order = np.argsort(energies)[:20]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    labels = [res_names[i] for i in order]
    vals = [energies[i] for i in order]
    colors = ['#d62728' if v < 0 else '#1f77b4' for v in vals]
    ax.barh(range(len(labels)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Energy contribution (kcal/mol)')
    ax.set_title('MM/PBSA Per-residue Energy Decomposition (Top 20)')
    ax.axvline(0, color='k', lw=0.8)
    ax.grid(alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'md_residue_decomp.png'), dpi=300)
    plt.close(fig)
    print("  [输出] md_residue_decomp.png")


if __name__ == '__main__':
    print("=" * 60)
    print("MD 验证结果绘图 — ASPA-Carnosine 100 ns GROMACS + MM/PBSA")
    print(f"  数据目录: {os.path.abspath(DATA_DIR)}")
    print(f"  输出目录: {os.path.abspath(OUT_DIR)}")
    print("=" * 60)
    plot_rmsd()
    plot_rmsf()
    plot_sasa_rg_hbond()
    plot_binding_energy()
    plot_residue_decomp()
    print("\n完成。输出与申报材料中 md_*.png 对应的全部图件。")
