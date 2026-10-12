#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6_plot_lowdata_fig.py —— A图独立出图脚本(原图风格, 不训练)
与 1_molgpt_train.py Cell 7.5 绘图逻辑保持一致.

用法:
  python3 6_plot_lowdata_fig.py [csv路径]
    · 默认读 ../output_molgpt/lowdata_results.csv
    · 图片输出到 CSV 同目录 lowdata_efficiency.png
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CANDIDATES = [
    r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
]
_CANDIDATES += glob.glob('/usr/share/fonts/**/NotoSansCJK*.*', recursive=True)
_CANDIDATES += glob.glob('/usr/share/fonts/**/wqy*.*', recursive=True)
for _fp in _CANDIDATES:
    if os.path.exists(_fp):
        try:
            font_manager.fontManager.addfont(_fp)
        except Exception:
            pass
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
                                   'WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, '..', 'output_molgpt', 'lowdata_results.csv')
CSV_PATH = os.path.abspath(CSV_PATH)
assert os.path.exists(CSV_PATH), f'找不到CSV: {CSV_PATH}'
OUT_PATH = os.path.join(os.path.dirname(CSV_PATH), 'lowdata_efficiency.png')

df_ld = pd.read_csv(CSV_PATH)
VARIANTS = [('molgpt_baseline', 'MolGPT baseline'),
            ('molgpt_graphbias', 'MolGPT + graph-bias (ours)')]
ratios = sorted(df_ld['ratio'].unique())
n_seed = int(df_ld.groupby(['variant', 'ratio']).size().max())
print(f'[数据] {CSV_PATH}  |  {len(df_ld)}个点, 档位: '
      + ', '.join(f'{r*100:.0f}%' for r in ratios) + f', 每点最多{n_seed}种子')

def fit_saturation(x, y):
    best = None
    for a in np.linspace(y.max() + 0.002, 0.999, 400):
        z = np.log(np.clip(a - y, 1e-5, None))
        A = np.vstack([np.ones_like(x), -x]).T
        coef, *_ = np.linalg.lstsq(A, z, rcond=None)
        lb, c = coef
        b = np.exp(lb)
        sse = ((a - b * np.exp(-c * x) - y) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, a, b, c)
    return best[1], best[2], best[3]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
colors = {'molgpt_baseline': '#4C72B0', 'molgpt_graphbias': '#DD8452'}
curve = {}
for variant, label in VARIANTS:
    xs, ym, ye = [], [], []
    for r in ratios:
        sub = df_ld[(df_ld['variant'] == variant) &
                    (np.isclose(df_ld['ratio'], r))]['validity']
        if len(sub) == 0:
            continue
        xs.append(r * 100)
        ym.append(float(sub.mean()))
        ye.append(float(sub.std(ddof=0)) if len(sub) > 1 else 0.0)
    xs, ym, ye = np.array(xs), np.array(ym), np.array(ye)
    ax.errorbar(xs, ym, yerr=ye, marker='o', capsize=5, linewidth=2,
                markersize=8, color=colors[variant], label=label)
    curve[variant] = dict(zip(xs, ym))
    a, b, c = fit_saturation(xs / 100.0, ym)
    xe = np.linspace(100, 150, 100)
    ax.plot(xe, a - b * np.exp(-c * (xe / 100)), color=colors[variant],
            ls='--', lw=2, alpha=0.7)

v_base100 = curve['molgpt_baseline'][100.0]
ax.axhline(v_base100, color='#4C72B0', linestyle=':', alpha=0.6,
           label=f'baseline@100% = {v_base100:.1%}')
ax.text(152, v_base100 - 0.035, 'dashed: extrapolation (not measured)',
        color='#777777', fontsize=9, ha='right')
ax.set_xlabel('Training data fraction (%)', fontsize=12)
ax.set_ylabel('Validity (fixed eval seed 999)', fontsize=12)
ax.set_title('Data Efficiency: Graph-Bias Prior in the Low-Data Regime', fontsize=13)
ax.set_xlim(0, 155)
ax.set_xticks(sorted(curve['molgpt_baseline'].keys()) + [125, 150])
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f'[出图] {OUT_PATH}')
