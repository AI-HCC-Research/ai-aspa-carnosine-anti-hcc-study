# -*- coding: utf-8 -*-
"""
ASPA 蛋白结构标准化预处理 (加氢质子化)
============================================================
步骤三: 将 GNN 选中的起始构象 (ASPA_best_start.pdb) 预处理为
        可直接用于分子动力学模拟的标准结构.

预处理内容 (与归档结果说明逐项对应):
  1. 移除平台元信息头: 删除 HelixFold3 的 HEADER/TITLE/AUTHOR/
     REMARK 及 MODEL/ENDMDL 封装, 仅保留 ATOM/TER/END 记录;
  2. 链标识规范化: 统一为标准 PDB 链格式 (链 A);
  3. 添加氢原子 (质子化): 在生理 pH (7.4) 条件下补全氢原子
     (PDBFixer/OpenMM 标准流程), 重原子坐标保持不变;
  4. 输出校验: 标准产物 2514 -> 5009 原子, 313 个残基齐全,
     C 端含 OXT.

如实说明 (与团队讨论后确认的记录):
  归档的 results/step3/ASPA_ready.pdb (4730 原子) 缺 279 个 α 氢
  (HA), 且个别芳香环带非标准氢 (TRP 多 HD2 / HIS 多 HE2), 非标准
  工具产物, 应为当时定制脚本按简化几何规则加氢所致。本脚本为
  PDBFixer 标准流程复现, 产物 (5009 原子) 化学上更完整;
  为不改动归档结果, 本脚本另存为 ASPA_ready_pdbfixer.pdb,
  归档 ASPA_ready.pdb 保持原样, 两处差异已如实记录于 结果说明.txt.

输入: data/step3/ASPA_best_start.pdb
输出: results/step3/ASPA_ready_pdbfixer.pdb  (+ 终端打印校验报告)

运行方式:
  python src/step3_structure_preprocessing.py

依赖:
  pip install -r requirements.txt   (openmm, pdbfixer)
"""

import os
from pdbfixer import PDBFixer
from openmm.app import PDBFile

# ============================================================
# 路径设置 (本文件位于 src/; 输入在 data/step3/, 输出到 results/step3/)
# ============================================================
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
INPUT_PDB = os.path.join(PROJECT_ROOT, 'data', 'step3', 'ASPA_best_start.pdb')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'results', 'step3')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 另存新文件名: 不覆盖归档的 ASPA_ready.pdb (4730 原子)
OUTPUT_PDB = os.path.join(OUTPUT_DIR, 'ASPA_ready_pdbfixer.pdb')

PH = 7.4   # 生理 pH

# ============================================================
# 第1步: 加载结构 (PDBFixer 自动丢弃 HEADER/REMARK 等平台头)
# ============================================================
print('=' * 60)
print('第1步: 加载 ASPA_best_start.pdb')
print('=' * 60)

fixer = PDBFixer(filename=INPUT_PDB)
n_heavy_in = sum(1 for a in fixer.topology.atoms()
                 if a.element.symbol != 'H')
residues = list(fixer.topology.residues())
print(f'输入: {n_heavy_in} 个重原子, {len(residues)} 个残基')

# ============================================================
# 第2步: 补全缺失原子/残基检查 + 生理pH加氢
# ============================================================
print('\n' + '=' * 60)
print(f'第2步: pH={PH} 条件下添加氢原子 (质子化)')
print('=' * 60)

fixer.findMissingResidues()
print(f'缺失残基: {fixer.missingResidues} (应为空 {{}} = 序列完整)')
fixer.findMissingAtoms()
fixer.addMissingAtoms()          # 序列完整时不新增重原子
fixer.addMissingHydrogens(pH=PH)

n_total = sum(1 for _ in fixer.topology.atoms())
n_h = sum(1 for a in fixer.topology.atoms() if a.element.symbol == 'H')
print(f'加氢后: 总原子 {n_total} (重原子 {n_total - n_h} + 氢 {n_h})')

# ============================================================
# 第3步: 写出标准 PDB (仅 ATOM/TER/END 记录)
# ============================================================
print('\n' + '=' * 60)
print('第3步: 写出 ASPA_ready_pdbfixer.pdb')
print('=' * 60)

if os.path.exists(OUTPUT_PDB):
    # Windows 下 zip 解出的文件可能带只读属性, 先解除以便覆写
    os.chmod(OUTPUT_PDB, 0o666)
with open(OUTPUT_PDB, 'w') as f:
    PDBFile.writeFile(fixer.topology, fixer.positions, f)
print(f'已写出: {OUTPUT_PDB}')

# ============================================================
# 第4步: 输出校验 (标准工具口径 + 与归档文件的差异说明)
# ============================================================
print('\n' + '=' * 60)
print('第4步: 输出校验')
print('=' * 60)

n_atom = 0
res_ids = set()
has_oxt = False
with open(OUTPUT_PDB) as f:
    for line in f:
        if line.startswith('ATOM'):
            n_atom += 1
            res_ids.add((line[21].strip(), int(line[22:26])))
            if line[12:16].strip() == 'OXT':
                has_oxt = True

print(f'总原子数: {n_atom}  (PDBFixer 标准加氢预期 5009)')
print(f'残基数:   {len(res_ids)}  (预期 313)')
print(f'C端 OXT:  {"有" if has_oxt else "无"}  (预期 有)')
ok = (n_atom == 5009 and len(res_ids) == 313 and has_oxt)
print(f'\n标准校验: {"通过 ✓" if ok else "未通过 ✗"}')
print()
print('与归档 ASPA_ready.pdb (4730 原子) 的差异说明:')
print('  归档文件缺 279 个 α 氢 (HA), 且个别芳香环带非标准氢')
print('  (TRP 多 HD2 / HIS 多 HE2), 并非标准工具产物, 应为当时')
print('  定制脚本按简化几何规则加氢所致; 本脚本为标准流程,')
print('  产物化学上更完整 (含全部 HA), 差异属预期、已如实记录.')
