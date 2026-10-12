# -*- coding: utf-8 -*-
"""科研图规范 v2：A 对接 / B XGBoost Top5；条内白字署名、条端精确数值、规范双语图注"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

zh = fm.FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.linewidth': 0.9,
    'xtick.major.width': 0.9,
    'xtick.major.size': 3.5,
    'axes.unicode_minus': False,
})

HIT  = '#B22222'   # 肌肽
BAR  = '#4A7AB8'   # 其余分子（与综合排名图同款蓝色）
INK  = '#1A1A1A'
SUB  = '#5A5A5A'

# (主名, 分数)
dock = [
    ('Boehmeriasin A',   -9.076),
    ('N-Acetyl-L-cysteine', -8.710),
    ('Carnosine',        -8.592),
    ('Isocitric acid',   -8.435),
    ('Creatine',         -8.423),
]
xgb = [
    ('HYKOP',              6.4736),
    ('2-hydroxy-1,2,3-propane-\ntricarboxylic acid-2-ethyl ester', 6.3949),
    ('Taurine',            6.3512),
    ('Carnosine',          6.3312),
    ('Lysopine',           6.3069),
]

fig = plt.figure(figsize=(13.33, 6.2), dpi=300)
fig.patch.set_facecolor('white')
axL = fig.add_axes([0.072, 0.40, 0.375, 0.49])
axR = fig.add_axes([0.575, 0.40, 0.375, 0.49])

def panel(ax, data, neg, xlim, xticks, title, xlabel):
    ys = list(range(len(data)))[::-1]
    vals = [d[1] for d in data]
    colors = [HIT if 'Carnosine' in d[0] else BAR for d in data]
    ax.barh(ys, vals, height=0.72, color=colors, edgecolor='white', linewidth=0.8, zorder=3)
    ax.set_yticks([])
    ax.set_ylim(-0.72, len(data) - 0.28)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.tick_params(axis='x', labelsize=10, colors=INK, pad=3)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#555555')
    ax.axvline(0, color='#555555', lw=0.9)
    for y, (name, v) in zip(ys, data):
        hit = 'Carnosine' in name
        # —— 条内英文名（白字，靠零端，垂直居中）——
        if neg:
            ax.text(-0.22, y, name, va='center', ha='right', fontsize=10.2,
                    color='white', fontweight='bold' if hit else 'normal',
                    fontproperties=zh, zorder=4, linespacing=1.2)
        else:
            ax.text(0.22, y, name, va='center', ha='left', fontsize=10.2,
                    color='white', fontweight='bold' if hit else 'normal',
                    fontproperties=zh, zorder=4, linespacing=1.2)
    ax.set_xlabel(xlabel, fontsize=10, color=SUB, labelpad=6)
    ax.set_title(title, fontsize=12.5, color=INK, fontweight='bold', pad=10, loc='left')

panel(axL, dock, True, (-10.3, 0.35), [0, -2, -4, -6, -8, -10],
      'A   Physics-based flexible docking   ( n = 45 candidates )',
      'Docking score (kcal/mol)       more negative  →  stronger binding')
panel(axR, xgb, False, (0, 7.9), [0, 1, 2, 3, 4, 5, 6, 7],
      'B   XGBoost binding-affinity prediction   ( n = 45 candidates )',
      'Predicted pK = −log Kd/Ki       larger  →  stronger binding')

# ---- 中文图注 ----
zh_lines = [
 ('图注：45 个候选分子经（A）柔性对接与（B）XGBoost 亲和力预测两套独立方法排序，各取 Top 5；', 9.2, SUB),
 ('肌肽为唯一同时进入两套榜单 Top 5 的分子（对接第 3/45，XGBoost 第 4/45），双方法共识显著降低单一打分函数的假阳性风险。', 9.2, SUB),
 ('数据来源：FleDock.csv（01_虚拟筛选与柔性对接流程数据/10_最终排名）；XGBoost训练流程与精筛报告_v4.docx。', 7.8, '#999999'),
]

y = 0.262
for line, fs, col in zh_lines:
    fig.text(0.012, y, line, fontsize=fs, color=col, ha='left', va='top', fontproperties=zh)
    y -= 0.030

out = r'd:\ASPA-肌肽\文献调研_肌肽ASPA\figures\C0_双模型排名交叉验证.png'
plt.savefig(out, dpi=300, facecolor='white')
print('saved', out)
