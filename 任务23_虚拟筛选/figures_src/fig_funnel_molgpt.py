# -*- coding: utf-8 -*-
"""MolGPT 虚拟筛选 2D 漏斗图 - 中文优化版"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

stages = [
    ("生成分子库",               5000),
    ("二维过滤与去重",             3754),
    ("QED 抽样 Top500",            500),
    ("三维构象与\nVina 对接",       499),
    ("最终 Top20",                 20),
]
names = [s[0] for s in stages]
vals = np.array([s[1] for s in stages], dtype=float)

colors = ['#B8D4E8', '#8FBCD4', '#A8D5BA', '#F5C6A5', '#F0A8B8']

fig, ax = plt.subplots(figsize=(11, 7.5))

n = len(stages)
maxv = vals.max()
shrink = 0.08   # 每层底部至少比顶部窄 8%，保证侧边有斜度

def width_at(val):
    return 0.12 + (val / maxv) * 0.82

# 预计算每层的 w_top 和 w_bottom：衔接处宽度连续，且每层有最小收窄
w_top_list = [0.0] * n
w_bot_list = [0.0] * n
w_top_list[0] = width_at(vals[0])
for i in range(n):
    if i < n - 1:
        # 底部宽度 = min(下一层应有宽度, 本层顶部×(1-shrink))
        w_bot_list[i] = min(width_at(vals[i+1]), w_top_list[i] * (1 - shrink))
        # 下一层顶部 = 本层底部，保证衔接连续无台阶
        w_top_list[i+1] = w_bot_list[i]
    else:
        w_bot_list[i] = w_top_list[i] * 0.45

for i in range(n):
    y_top = n - i
    y_bottom = n - i - 1
    w_top = w_top_list[i]
    w_bottom = w_bot_list[i]

    x_left_top = 0.5 - w_top / 2
    x_right_top = 0.5 + w_top / 2
    x_left_bot = 0.5 - w_bottom / 2
    x_right_bot = 0.5 + w_bottom / 2

    poly = Polygon([
        (x_left_top, y_top),
        (x_right_top, y_top),
        (x_right_bot, y_bottom),
        (x_left_bot, y_bottom),
    ], closed=True, facecolor=colors[i], edgecolor='none')
    ax.add_patch(poly)

    label_y = (y_top + y_bottom) / 2
    ax.text(0.5, label_y + 0.15, f"{int(vals[i]):,}", ha='center', va='center',
            fontsize=17, color='#2c3e50', fontweight='bold')
    ax.text(0.5, label_y - 0.25, names[i], ha='center', va='center',
            fontsize=10, color='#2c3e50')

ax.set_xlim(0.02, 0.98)
ax.set_ylim(-0.8, n + 0.5)
ax.axis('off')

out = r"d:\ASPA-肌肽\文献调研_肌肽ASPA\_molgpt_funnel.png"
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
print(f"SAVED: {out}")
