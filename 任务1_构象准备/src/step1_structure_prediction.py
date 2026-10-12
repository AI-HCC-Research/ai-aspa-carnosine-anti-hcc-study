# -*- coding: utf-8 -*-
"""
ASPA 蛋白 HelixFold3 结构预测结果解析与置信度分析
============================================================
步骤一: 解析 HelixFold3 平台输出的 5 个候选构象置信度指标,
        汇总为 summary.csv, 并生成 pLDDT 分析图.

说明:
  结构预测在 HelixFold3 在线平台 (PaddleHelix) 完成, 本脚本不负责
  预测本身, 负责对平台下载的原始结果 (all_results.json + PDB)
  进行解析、汇总与可视化, 输出与提交包内结果文件一致.

输入: results/step1/ranked_structures/rank1~rank5/
  - all_results.json       (平台输出的逐原子/逐残基置信度)
  - predicted_structure.pdb (预测结构, 用于绘制 pLDDT 着色图)

输出: results/step1/ 文件夹下
  1. summary.csv                          (5 候选全局指标汇总)
  2. step1_predicted_structure_pLDDT.png  (rank1 结构 pLDDT 着色图)
  3. step1_residue_plddt_curve.png        (逐残基 pLDDT 曲线)

运行方式:
  python src/step1_structure_prediction.py

依赖:
  pip install -r requirements.txt
"""

import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# pLDDT 官方配色 (橙->黄->浅蓝->深蓝 = 低->高置信)
PLDDT_CMAP = LinearSegmentedColormap.from_list(
    'pLDDT', [(0.0, '#FF7D45'), (0.35, '#F5C83F'),
              (0.7, '#65CBF3'), (1.0, '#0053D6')])

# ============================================================
# 路径设置 (本文件位于 src/; 平台下载结果在 results/step1/)
# ============================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RANK_DIR = os.path.join(PROJECT_ROOT, 'results', 'step1', 'ranked_structures')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step1')

N_MODELS = 5  # HelixFold3 每次任务输出 5 个候选构象

# ============================================================
# 第1步: 解析 5 个候选的 all_results.json, 汇总全局指标
# ============================================================
print('=' * 60)
print('第1步: 解析 5 个候选构象的全局置信度指标')
print('=' * 60)

rows = []
for rank in range(1, N_MODELS + 1):
    json_path = os.path.join(RANK_DIR, f'rank{rank}', 'all_results.json')
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)
    row = {
        'model_id': f'1-{rank}',
        'has_clash': d['global_has_clash'],
        'global_ptm': d['global_ptm'],
        'global_iptm': d['global_iptm'] if d['global_iptm'] is not None else '',
        'global_plddt': d['global_plddt'],
        'ranking_confidence': d['ranking_confidence'],
    }
    rows.append(row)
    print(f"rank{rank} (model {row['model_id']}): "
          f"pTM={row['global_ptm']}, pLDDT={row['global_plddt']}, "
          f"clash={row['has_clash']}")

summary_path = os.path.join(OUTPUT_DIR, 'summary.csv')
with open(summary_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f'\n已写出: {summary_path}')

# ============================================================
# 第2步: 逐残基 pLDDT 统计 (token_plddt, 长度=残基数)
# ============================================================
print('\n' + '=' * 60)
print('第2步: 逐残基 pLDDT 统计 (rank1)')
print('=' * 60)

with open(os.path.join(RANK_DIR, 'rank1', 'all_results.json'),
          encoding='utf-8') as f:
    d1 = json.load(f)
token_plddt = np.array(d1['token_plddt'], dtype=float)
print(f'残基数: {len(token_plddt)}')
print(f'pLDDT 均值: {token_plddt.mean():.2f}  '
      f'最小: {token_plddt.min():.2f}  最大: {token_plddt.max():.2f}')
print(f'pLDDT >= 90 的残基占比: '
      f'{(token_plddt >= 90).mean() * 100:.1f}%')
print(f'pLDDT >= 70 的残基占比: '
      f'{(token_plddt >= 70).mean() * 100:.1f}%')

fig, ax = plt.subplots(figsize=(9, 3.6), dpi=200)
ax.plot(np.arange(1, len(token_plddt) + 1), token_plddt,
        lw=1.6, color='#0053D6')
ax.axhline(90, color='#2E7D32', ls='--', lw=1, alpha=0.7)
ax.axhline(70, color='#F5A623', ls='--', lw=1, alpha=0.7)
ax.text(len(token_plddt) * 0.99, 90.8, '90 (极高置信)',
        ha='right', fontsize=9, color='#2E7D32')
ax.text(len(token_plddt) * 0.99, 70.8, '70 (可信)',
        ha='right', fontsize=9, color='#B07C10')
ax.set_xlabel('残基序号', fontsize=11)
ax.set_ylabel('pLDDT', fontsize=11)
ax.set_ylim(0, 105)
ax.set_title('HelixFold3 预测 ASPA 结构 逐残基置信度 (rank1)',
             fontsize=12)
ax.grid(ls='--', alpha=0.35)
fig.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, 'step1_residue_plddt_curve.png')
fig.savefig(curve_path, dpi=200)
plt.close(fig)
print(f'已写出: {curve_path}')

# ============================================================
# 第3步: rank1 结构按逐残基 pLDDT 着色 (3D 散点)
# ============================================================
print('\n' + '=' * 60)
print('第3步: 绘制 rank1 结构 pLDDT 着色图')
print('=' * 60)


def parse_ca(pdb_path):
    """读取 PDB 中 CA 原子坐标与 B-factor(即逐残基 pLDDT)"""
    xyz, bf = [], []
    with open(pdb_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                xyz.append([float(line[30:38]), float(line[38:46]),
                            float(line[46:54])])
                bf.append(float(line[60:66]))
    return np.array(xyz), np.array(bf)


pdb1 = os.path.join(RANK_DIR, 'rank1', 'predicted_structure.pdb')
xyz, plddt = parse_ca(pdb1)
print(f'CA 原子数: {len(xyz)} (应等于残基数 313)')

fig = plt.figure(figsize=(6.2, 5.6), dpi=200)
fig.subplots_adjust(left=0.00, right=0.80, bottom=0.13, top=0.90)
ax = fig.add_subplot(111, projection='3d')
ax.add_collection3d(Line3DCollection(
    [xyz[i:i + 2] for i in range(len(xyz) - 1)],
    colors='#888888', linewidths=1.0, alpha=0.5))
sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=plddt,
                cmap=PLDDT_CMAP, vmin=40, vmax=100, s=10,
                depthshade=True)
mins, maxs = xyz.min(0), xyz.max(0)
ctr = (mins + maxs) / 2
r = (maxs - mins).max() / 2
ax.set_xlim(ctr[0] - r, ctr[0] + r)
ax.set_ylim(ctr[1] - r, ctr[1] + r)
ax.set_zlim(ctr[2] - r, ctr[2] + r)
ax.set_box_aspect((1, 1, 1))
ax.set_xlabel('X (Å)', fontsize=10, labelpad=2)
ax.set_ylabel('Y (Å)', fontsize=10, labelpad=2)
ax.set_zlabel('Z (Å)', fontsize=10, labelpad=4)
ax.tick_params(labelsize=8, pad=2)
ax.set_title('HelixFold3 预测 ASPA 结构（按逐残基 pLDDT 着色）',
             fontsize=12, pad=14)
cax = fig.add_axes([0.86, 0.22, 0.045, 0.56])
cb = fig.colorbar(sc, cax=cax)
cb.set_label('逐残基 pLDDT', fontsize=10)
cb.ax.tick_params(labelsize=8)
ax.view_init(elev=18, azim=-70)
struct_path = os.path.join(OUTPUT_DIR,
                           'step1_predicted_structure_pLDDT.png')
fig.savefig(struct_path, dpi=200)
plt.close(fig)
print(f'已写出: {struct_path}')

print('\n全部完成 ✓')
