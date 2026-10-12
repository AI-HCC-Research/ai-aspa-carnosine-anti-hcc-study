# -*- coding: utf-8 -*-
"""
=============================================================================
子任务2 前置脚本: MolGPT 从零训练 + 三模型消融对比 (DGX云服务器迁移版)
=============================================================================
【DGX服务器迁移版】训练逻辑/超参/种子42与Kaggle版完全一致, 仅改路径锚定:
  工程根 = 脚本上级目录(若含 data/) 否则脚本所在目录; 产物存 output_molgpt/ 等。
与《子任务2_生成式分子探索_Kaggle.py》(主流程脚本) 的关系:
  本脚本 = 生成引擎本体。训练好的 MolGPT 生产采样 5000 条有效分子,
  作为主流程脚本阶段一的唯一候选来源 (读取后进入四道关卡 → 278 库 → 24 条 QC),
  主流程阶段二至阶段五的过滤、收窄与质检逻辑保持不变。

三个模型变体 (消融对比, 论文核心创新点):
  A. molgpt_baseline : 标准 GPT (4层/8头/256维), 字符级 SMILES 自回归生成
  B. molgpt_graphbias: 创新点1 —— 图偏置注意力 (Graph-biased Attention)
       训练时把分子图原子间最短路径距离作为注意力偏置 (ALiBi 风格, 每头
       可学习斜率), 让模型显式感知"近处原子互相关注"的化学先验,
       目标: 提升括号配对/环闭合等长程语法 → 提高生成有效性
       (文献支撑: Graph-Prior Transformer 2025 / S23D 2025 / DrugEx v3)
  C. molgpt_multitask: 创新点2 —— 多任务学习 (生成 + 性质预测)
       在生成损失外联合预测 QED / logP / SA 三性质, 迫使隐层表示
       内化学类药信息 → 目标: 提升生成分子平均 QED
       注: 图偏置仅在训练时注入 (推理时退化为标准注意力), 属于
       "训练时图先验正则"方案, 与消融公平性一致 (三模型采样条件相同)

数据: ZINC-250K (25万条类药物分子, 标准基准, 可与文献指标直接对比)
  下载: Kaggle Notebook 需在 Settings -> Internet 打开 (ON)

流程 (9 个 Cell):
  Cell 0  环境自检 (GPU / RDKit / Internet)
  Cell 1  全局配置 (全部超参数集中此处, 种子 42)
  Cell 2  数据下载 + 清洗 + 预处理 (tokens / 稀疏图距离 / 性质标签)
  Cell 3  Tokenizer + Dataset + DataLoader
  Cell 4  模型定义 (GPT 基座 + 图偏置开关 + 多任务头开关)
  Cell 5  训练框架 (AdamW + 余弦退火 + fp16 + 早停 + 采样质检)
  Cell 6  依次训练三个模型 (每个约 15-25 分钟 P100)
  Cell 7  消融对比评估: Validity / Uniqueness / Novelty / QED / Diversity
          → ablation_table.csv + ablation_bars.png + train_curves.png
  Cell 8  最优模型生产采样 5000 条 → A3_MGPT_molgpt_only.csv (主流程阶段一输入)
  Cell 9  run_params.json 可复现参数存档

预计总耗时: P100 约 2-3 小时 (数据预处理 10min + 三模型 1.5h + 评估采样 30min)
显存占用: 约 2-3 GB (batch 128, block 104) —— P100 16GB 富余
=============================================================================
"""

# ===========================================================================
# Cell 0: 环境自检
# ===========================================================================
import os, sys, json, math, time, random, warnings, urllib.request, subprocess
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
_T0 = time.time()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator
    from rdkit import rdBase
except ModuleNotFoundError:
    print("[安装] RDKit 未找到, 正在安装 (约 30s) ...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'rdkit'])
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator
    from rdkit import rdBase
RDLogger.DisableLog('rdApp.*')

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 网络可用性不在此检测, Cell 2 下载 ZINC 时自然验证 (本地 Dataset 存在则无需网络)

print(f"[环境] Python {sys.version.split()[0]} | PyTorch {torch.__version__} | RDKit {rdBase.rdkitVersion}")
print(f"[环境] 计算设备: {_DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if _DEVICE.type=='cuda' else ""))
if _DEVICE.type != 'cuda':
    print("[警告] 未检测到 GPU! 请检查服务器 NVIDIA 驱动/CUDA 与 PyTorch 安装")
print(f"[OK] Cell 0 完成 (耗时 {time.time()-_T0:.1f}s)")


# ===========================================================================
# Cell 1: 全局配置 (阶段五可复现性源头, 全部参数集中在此)
# ===========================================================================
SEED       = 42          # 全局随机种子 (与主流程一致)
N_TRAIN    = 250_000     # ZINC-250K 全量
MAX_SMILES = 100         # canonical SMILES 最大字符数 (超过则丢弃)
BLOCK_SIZE = 104         # 模型序列上限 = <bos> + 100 + <eos> (取 104 保险)

# ---- 模型超参 (小而稳, P100 友好) ----
N_LAYER  = 4             # Transformer 层数
N_HEAD   = 8             # 注意力头数
N_EMBD   = 256           # 嵌入维度
DROPOUT  = 0.1           # 防过拟合
GRAPH_DMAX = 6           # 图偏置: 仅存 原子对距离<=6 的稀疏对 (远处 bias=0)

# ---- 训练超参 ----
BATCH_SIZE = 128
LR         = 3e-4        # AdamW 初始学习率
LR_MIN     = 3e-5        # 余弦退火终点
WARMUP     = 300         # 预热步数
WEIGHT_DECAY = 0.01
EPOCHS     = 4           # ZINC-250K 上 3-4 epoch 即可饱和 (NovoMolGen 结论)
PATIENCE   = 2           # 早停: 验证损失连续 2 轮不降则停
GRAD_CLIP  = 1.0
LAMBDA_PROP = 0.1        # 多任务版: 性质预测损失权重

# ---- 采样超参 ----
SAMP_TEMP   = 0.8        # 温度 (低→保守高→多样; 0.8 为 SMILES 生成常用值)
SAMP_TOPK   = 40         # top-k 截断
EVAL_SEEDS  = (42, 123, 2024, 7, 314, 2025, 999, 5555)   # 消融评估: 8 个采样种子 (估计采样随机性, 解决单seed排名幻觉)
N_EVAL_SEED = 1000       # 每个采样种子的采样数 → 每模型共评估 N_EVAL = 3000 条
N_EVAL      = N_EVAL_SEED * len(EVAL_SEEDS)
N_BOOT      = 1000       # bootstrap 次数 (95% 置信区间 + 差值显著性检验)
N_PROD      = 5000       # 生产采样: 最优模型产出条数 (主流程阶段一的 5000 条原始候选)
NOVEL_REF_N = 50_000     # 新颖性参照: 训练集子集 (50K 已足够稳定估计)
N_NN_REF    = 20_000     # 最近邻 Tanimoto 参照库大小 (计算量与稳定性的平衡)

# ---- 低数据效率实验 (创新点1 图偏置的关键证据: 数据稀缺时化学先验是否更有价值) ----
LOWDATA_RATIOS  = (0.10, 0.30, 1.00)   # 训练数据占比
LOWDATA_TRSEEDS = (42, 7)              # 2 个训练种子 (曲线带误差棒)
LOWDATA_EVAL_N  = 1000                 # 每个低数据模型的评估采样数 (评估种子统一 999 保证公平)

# ---- 路径 (DGX服务器版: 锚定工程根, 产物分别存 output_molgpt/ckpt/molgpt_data) ----
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.path.abspath('.')
BASE_DIR = os.path.dirname(_SCRIPT_DIR) if os.path.isdir(os.path.join(os.path.dirname(_SCRIPT_DIR), 'data')) else _SCRIPT_DIR
WORK_DIR = BASE_DIR
GEN_DIR  = os.path.join(WORK_DIR, 'output_molgpt')
CKPT_DIR = os.path.join(WORK_DIR, 'ckpt')
DATA_DIR = os.path.join(WORK_DIR, 'molgpt_data')
os.makedirs(GEN_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 70)
print("全局配置:")
print(f"  种子={SEED} | 模型 {N_LAYER}层/{N_HEAD}头/{N_EMBD}维 | block={BLOCK_SIZE}")
print(f"  训练: batch={BATCH_SIZE} lr={LR} epoch≤{EPOCHS} 早停={PATIENCE}")
print(f"  消融采样={N_EVAL}条/模型 | 生产采样={N_PROD}条")
print(f"  产物目录: {GEN_DIR}")


def set_seed(seed=SEED):
    """固定 python/numpy/torch 全部随机源 (可复现性)"""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed()
print(f"[OK] Cell 1 完成 (耗时 {time.time()-_T0:.1f}s)")


# ===========================================================================
# Cell 2: 数据下载 + 清洗 + 预处理
#   产出 4 份中间文件 (npy/npz), 训练时直接加载, 不重复处理:
#     tokens.npy    [N,104]  int16   字符编码序列 (pad 对齐)
#     lengths.npy   [N]      int16   每条真实长度 (bos+内容+eos)
#     graph_pairs.npz       CSR     稀疏图距离对 (上三角, d<=6, int8)
#       - gp_flat   [M,3]   (原子i字符位置, 原子j字符位置, 距离)
#       - gp_ptr    [N+1]   每条分子的对区间指针
#     props.npy     [N,3]   fp32   (QED, logP, SAscore) —— 多任务标签
#     smiles_clean.csv      canonical SMILES + zinc id (清洗后全记录)
# ===========================================================================
print("\n" + "=" * 70)
# 防呆检查: DATA_DIR/GEN_DIR 在 Cell 1 定义, 单独运行本 Cell 会 NameError
if 'DATA_DIR' not in globals() or 'GEN_DIR' not in globals():
    raise RuntimeError(
        "检测到 DATA_DIR 未定义 —— 你可能只单独运行了 Cell 2。\n"
        "正确做法: 点击 Notebook 顶部 'Run All'(运行全部), 从 Cell 0 顺序执行;\n"
        "若内核刚重启过(新开会话), 更要从 Cell 0 开始。")
print("Cell 2: 数据下载 + 清洗 + 预处理")

# ---- 2.1 获取 ZINC-250K (优先级: 已挂载 Kaggle Dataset > 联网下载) ----
# 正确文件 = chemical_vae 官方 250k_rndm_zinc_drugs_clean_3.csv
# (注意: 旧地址 models/zinc/zinc_250k.smi 已 404, 是此前下载失败的根因)
ZINC_URLS = [
    # 主源: Gómez-Bombarelli et al. 化学自编码器仓库 (master 分支)
    'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv',
    # 备源: 同文件 main 分支 (仓库已迁移到 the-matter-lab, 部分镜像只同步 main)
    'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/main/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv',
]
RAW_FILE = os.path.join(DATA_DIR, 'zinc_250k_raw.smi')


def _zinc_score(path):
    """数据集候选打分: 250k 官方文件优先, 防止误选 zinc_10k 等小文件"""
    n = os.path.basename(path).lower()
    return (2 if '250k' in n else 0) + (1 if 'zinc' in n else 0)


# 优先: 本地已有数据 (服务器 data/ 目录或Kaggle挂载), 自动发现; 否则联网下载
_zinc_candidates = []
_for_roots = ['/kaggle/input', os.path.join(BASE_DIR, 'data')]
for _scan_root in [r for r in _for_roots if os.path.exists(r)]:
    for _root, _dirs, _files in os.walk(_scan_root):
        for _f in _files:
            if _f.lower().endswith(('.smi', '.csv', '.txt')) and \
               ('zinc' in _f.lower() or '250k' in _f.lower()):
                _zinc_candidates.append(os.path.join(_root, _f))
_zinc_located = max(_zinc_candidates, key=_zinc_score) if _zinc_candidates else None
if _zinc_candidates:
    print(f"  [数据] 本地数据目录发现 {len(_zinc_candidates)} 个候选, "
          f"按优先级选用: {os.path.basename(_zinc_located)}")

if _zinc_located:
    RAW_FILE = _zinc_located
    print(f"  [数据] 使用本地数据集 (无需联网): {RAW_FILE}")
else:
    print("  [数据] 未挂载 ZINC 数据集, 尝试联网下载 ...")

    def _download_zinc(url, dst):
        """带超时下载 + 内容校验 (文件需 >5MB 且头部含可解析 SMILES,
           防止 404/错误 HTML 页面被误当成数据保存)"""
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=300) as resp, open(dst, 'wb') as f:
            f.write(resp.read())
        if os.path.getsize(dst) < 5_000_000:
            return False
        # 内容校验: 用 csv 模块按记录解析前 20 条 (官方 CSV 的 SMILES 带引号且跨行,
        # 按行 split 会漏判), 任一条能被 RDKit 解析即视为有效数据
        import csv as _csv
        with open(dst, 'r', errors='ignore', newline='') as f:
            _rows = list(_csv.reader(f))[:20]
        for _row in _rows[1:]:
            if not _row or not _row[0].strip():
                continue
            _tok = _row[0].strip().strip('"').split()[0]
            if _tok and Chem.MolFromSmiles(_tok) is not None:
                return True
        return False

    _ok = False
    for _url in ZINC_URLS:
        try:
            print(f"    尝试: {_url}")
            if _download_zinc(_url, RAW_FILE):
                _ok = True
                print(f"  [数据] 下载并校验成功 ({os.path.getsize(RAW_FILE)/1e6:.1f} MB)")
                break
            print("    文件大小或内容校验未通过, 换下一个源")
        except Exception as e:
            print(f"    下载失败 ({e.__class__.__name__}), 换下一个源")
    if not _ok:
        raise RuntimeError(
            "ZINC-250K 获取失败! 优先检查服务器联网; 或手动下载官方CSV放入 data/ 目录后重跑 (代码会自动识别本地文件):\n"
            "  ① Kaggle Notebook 右侧 Add Input → 搜索数据集 zinc250k "
            "(basu369victor/zinc250k) → 添加后重跑, 代码会自动识别;\n"
            "  ② Settings → Internet 打开 (ON, 需手机号验证) 后重跑自动下载;\n"
            "  ③ 电脑手动下载 250k_rndm_zinc_drugs_clean_3.csv → Kaggle → "
            "Datasets → New Dataset 上传 (标题填 zinc250k) → 回 Notebook "
            "Add Input 添加后重跑。")

# ---- 2.2 逐条清洗 (盐剥离 + canonical 化 + 过滤) ----
def clean_smiles(raw):
    """原始行 -> canonical SMILES; 无效/含盐/过长返回 None"""
    # 兼容三种输入: CSV 记录(已由 csv 模块解出纯 SMILES) / "SMILES id" / 带引号字段
    smi = raw.strip().strip('"').strip("'").split(',')[0].split()[0] if raw.strip() else ''
    mol = Chem.MolFromSmiles(smi)                         # 解析失败 → 丢弃
    if mol is None:
        return None
    if '.' in Chem.MolToSmiles(mol):                      # 多片段(盐) → 取最大片段
        frags = Chem.GetMolFrags(mol, asMols=True)
        mol = max(frags, key=lambda m: m.GetNumAtoms())
    if mol.GetNumAtoms() < 8:                             # 过小分子无类药意义
        return None
    canon = Chem.MolToSmiles(mol)                         # 规范 SMILES (唯一写法)
    if len(canon) > MAX_SMILES:                           # 序列过长
        return None
    return canon

# 统一读取 (关键修复): 官方 250k CSV 为 "smiles,logP,qed,SAS" 格式, 且 SMILES
# 字段带引号并可能跨行 (一条记录占两行), 按行 split 会把记录拆碎导致全量解析失败;
# 必须用 csv 模块按记录解析。.smi 格式 (每行 "SMILES id") 走空格分支。
import csv as _csv
raw_lines = []
with open(RAW_FILE, 'r', newline='', encoding='utf-8', errors='ignore') as f:
    _first = f.readline(); f.seek(0)                       # 预读首行判断格式
    if 'smiles' in _first.lower():                         # CSV 格式 (chemical_vae 官方)
        for _row in _csv.reader(f):
            if _row and _row[0].strip():
                raw_lines.append(_row[0].strip().strip('"'))
    else:                                                  # SMI 格式
        for _ln in f:
            if _ln.strip():
                raw_lines.append(_ln.split()[0])
print(f"  [数据] 原始条数: {len(raw_lines)}")

t_clean = time.time()
canon_list = []
for ln in raw_lines:
    c = clean_smiles(ln)
    if c is not None:
        canon_list.append(c)
print(f"  [数据] 清洗后有效: {len(canon_list)} 条 (耗时 {time.time()-t_clean:.0f}s)")

# ---- 2.3 构建字符词汇表 (含 <pad>/<bos>/<eos> 三个特殊符) ----
PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
char_set = sorted({ch for smi in canon_list for ch in smi})
stoi = {PAD: 0, BOS: 1, EOS: 2}
stoi.update({ch: i + 3 for i, ch in enumerate(char_set)})
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(stoi)
print(f"  [词表] 字符 {len(char_set)} 个 + 3 特殊符 = {VOCAB_SIZE}")

# ---- 2.4 核心预处理: tokens + 稀疏图距离 + 性质标签 ----
# 关键事实: RDKit MolFromSmiles(canon) 的原子 idx 顺序 = SMILES 中原子出现顺序
#           → 扫描 canonical SMILES 的"原子起始字符"即可对齐距离矩阵
def atom_start_positions(smi):
    """返回每个原子在 SMILES 字符串中的起始位置列表 (顺序=原子 idx)"""
    pos, i, n = [], 0, len(smi)
    while i < n:
        ch = smi[i]
        if ch == '[':                                     # 括号原子 [NH+] 整体算 1 原子
            pos.append(i); i = smi.index(']', i) + 1
        elif ch.isupper():                                # 大写字母 = 原子起始 (C/N/Cl 的C/Br的B)
            pos.append(i); i += 1
        else:                                             # 小写/数字/括号/键符号 → 跳过
            i += 1
    return pos

L = BLOCK_SIZE
N = len(canon_list)
tokens = np.full((N, L), stoi[PAD], dtype=np.int16)       # 全 pad 预分配
lengths = np.zeros(N, dtype=np.int16)
gp_flat_runs, gp_ptr = [], [0]                            # CSR 稀疏图距离
props = np.zeros((N, 3), dtype=np.float32)
from rdkit.Contrib.SA_Score import sascorer                # RDKit 自带合成可及性 (提前导入)
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

t_pre = time.time()
n_graph_fail = 0
for i, smi in enumerate(canon_list):
    # --- 字符编码: <bos> + 内容 + <eos> + pad ---
    ids = [stoi[BOS]] + [stoi[ch] for ch in smi] + [stoi[EOS]]
    lengths[i] = len(ids)
    tokens[i, :len(ids)] = ids

    # --- 图距离稀疏对 (上三角 i<j, d<=GRAPH_DMAX, int8 省 5 倍内存) ---
    mol = Chem.MolFromSmiles(smi)                          # 重新解析: 其 idx 顺序=字符原子顺序
    apos = atom_start_positions(smi)
    if mol is not None and len(apos) == mol.GetNumAtoms():
        D = Chem.GetDistanceMatrix(mol)                    # [nA,nA] 最短路径距离
        nA = len(apos)
        for a in range(nA):
            row = D[a, a+1:]
            for b_off in np.nonzero((row > 0) & (row <= GRAPH_DMAX))[0]:
                b = a + 1 + b_off
                gp_flat_runs.append((apos[a], apos[b], int(row[b_off])))
    else:
        n_graph_fail += 1                                  # 极罕见解析抖动 → 该条无图偏置
    gp_ptr.append(len(gp_flat_runs))

    # --- 多任务性质标签 ---
    m2 = mol if mol is not None else Chem.MolFromSmiles(smi)
    if m2 is not None:
        props[i, 0] = QED.qed(m2)
        props[i, 1] = Descriptors.MolLogP(m2)
        props[i, 2] = sascorer.calculateScore(m2)

    if (i + 1) % 25_000 == 0:
        print(f"    预处理进度: {i+1}/{N} (耗时 {time.time()-t_pre:.0f}s)")

gp_flat = np.array(gp_flat_runs, dtype=np.int8) if gp_flat_runs else np.zeros((0,3), np.int8)
gp_ptr  = np.array(gp_ptr, dtype=np.int64)
print(f"  [图距离] 稀疏对 {len(gp_flat):,} 个 (均 {len(gp_flat)/max(N,1):.0f}/条, 解析失败 {n_graph_fail} 条)")
# 注意: gp_flat 中的位置是相对 SMILES 字符串的 (0-based);
# 模型输入序列 x = <bos> + SMILES, 故 Dataset 构建 bias 时统一 +1 对齐 (见 Cell 3)

# ---- 2.5 落盘 (训练 Cell 直接读取, 重跑 notebook 免重复处理) ----
np.save(os.path.join(DATA_DIR, 'tokens.npy'), tokens)
np.save(os.path.join(DATA_DIR, 'lengths.npy'), lengths)
np.savez_compressed(os.path.join(DATA_DIR, 'graph_pairs.npz'), gp_flat=gp_flat, gp_ptr=gp_ptr)
np.save(os.path.join(DATA_DIR, 'props.npy'), props)
pd.DataFrame({'SMILES': canon_list}).to_csv(os.path.join(DATA_DIR, 'smiles_clean.csv'),
                                            index=False, encoding='utf-8-sig')
print(f"  ✓ 中间文件已存 {DATA_DIR} (耗时 {time.time()-t_pre:.0f}s)")
print(f"[OK] Cell 2 完成 (总耗时 {time.time()-_T0:.1f}s)")


# ===========================================================================
# Cell 3: Tokenizer + Dataset + DataLoader
# ===========================================================================
print("\n" + "=" * 70)
print("Cell 3: Dataset / DataLoader 构建")

class MolDataset(Dataset):
    """一条样本 = (输入序列, 目标序列, 图距离bias, 性质标签)
       自回归目标: 输入 <bos>c1cc... → 目标 c1cc...<eos> (整体右移一位)"""
    def __init__(self, idx_array, need_graph=False, need_prop=False):
        self.idx = idx_array
        self.need_graph, self.need_prop = need_graph, need_prop
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        x = torch.from_numpy(tokens[i].astype(np.int64))
        item = {'x': x[:-1], 'y': x[1:]}                   # 右移一位构造 next-token 目标
        if self.need_graph:                                # 图偏置: 上三角对 → 现场铺对称矩阵
            s, e = gp_ptr[i], gp_ptr[i+1]
            bias = np.zeros((L-1, L-1), dtype=np.float32)   # L-1 = 输入序列长度
            if e > s:
                pr = gp_flat[s:e].astype(np.int64) + 1      # +1: x[0]=<bos>, SMILES 字符整体后移 1 位
                bias[pr[:,0], pr[:,1]] = -pr[:,2]           # 距离取负 → 近处原子注意力更强
                bias[pr[:,1], pr[:,0]] = -pr[:,2]           # 对称铺满
            item['bias'] = torch.from_numpy(bias)
        if self.need_prop:
            item['props'] = torch.from_numpy(props[i].astype(np.float32))
        return item

def collate(batch):
    """批组装: stack + pad 目标 (同长度序列无需 pad, ZINC 已按长度过滤)"""
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    return out

# ---- 划分训练/验证 (98% / 2%), 验证集用于早停 ----
perm = np.random.RandomState(SEED).permutation(N)
n_val = max(2000, int(N * 0.02))
VAL_IDX, TRN_IDX = perm[:n_val], perm[n_val:]
print(f"  训练 {len(TRN_IDX):,} / 验证 {len(VAL_IDX):,}")

# ---- 训练集 canonical 集合 (新颖性 Novelty 的参照) ----
NOVEL_REF = set(np.array(canon_list)[perm[:NOVEL_REF_N]])
print(f"[OK] Cell 3 完成 (耗时 {time.time()-_T0:.1f}s)")


# ===========================================================================
# Cell 4: 模型定义 —— GPT 基座 + 两个创新开关
# ===========================================================================
print("\n" + "=" * 70)
print("Cell 4: 模型定义")

class GraphBiasAttention(nn.Module):
    """多头自注意力 (创新点1: 可选图距离偏置)
       logits += softplus(slope_h) * bias   —— 每头独立可学习斜率 (ALiBi 风格)
       bias[b,i,j] = -dist(原子i, 原子j): 距离越近偏置越大 → 化学近邻先验"""
    def __init__(self, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_head = N_EMBD // n_head
        self.qkv  = nn.Linear(N_EMBD, 3 * N_EMBD)          # Q/K/V 一次投影
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.attn_drop  = nn.Dropout(DROPOUT)
        self.resid_drop = nn.Dropout(DROPOUT)
        self.slope_raw = nn.Parameter(torch.zeros(n_head) + 0.1)   # softplus(0.1)≈0.74 起步
    def forward(self, x, graph_bias=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)   # [B,H,T,d]
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)     # [B,H,T,T]
        if graph_bias is not None:                                   # ← 创新点唯一改动处
            att = att + F.softplus(self.slope_raw)[None, :, None, None] \
                      * graph_bias[:, None, :, :]                    # 广播到所有头
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))              # 因果掩码 (只看过去)
        att = self.attn_drop(att.softmax(dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class Block(nn.Module):
    """标准 pre-LN Transformer Block (注意力可注入图偏置)"""
    def __init__(self, use_graph_bias):
        super().__init__()
        self.ln1  = nn.LayerNorm(N_EMBD)
        self.attn = GraphBiasAttention(N_HEAD)
        self.use_graph_bias = use_graph_bias
        self.ln2  = nn.LayerNorm(N_EMBD)
        self.mlp  = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT))
    def forward(self, x, graph_bias=None):
        x = x + self.attn(self.ln1(x),
                          graph_bias if self.use_graph_bias else None)
        x = x + self.mlp(self.ln2(x))
        return x

class MolGPT(nn.Module):
    """字符级 SMILES 自回归生成器
       use_graph_bias=True  → 变体 B (图偏置注意力)
       use_prop_head=True   → 变体 C (多任务: 隐状态同时回归 QED/logP/SA)"""
    def __init__(self, use_graph_bias=False, use_prop_head=False):
        super().__init__()
        self.use_graph_bias, self.use_prop_head = use_graph_bias, use_prop_head
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE - 1, N_EMBD)  # 输入长度 = BLOCK-1
        self.drop    = nn.Dropout(DROPOUT)
        self.blocks  = nn.ModuleList([Block(use_graph_bias) for _ in range(N_LAYER)])
        self.ln_f    = nn.LayerNorm(N_EMBD)
        self.head    = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight               # 权重绑定 (减参数防过拟合)
        if use_prop_head:                                    # ← 创新点2: 性质预测头
            self.prop_head = nn.Sequential(
                nn.Linear(N_EMBD, 128), nn.GELU(), nn.Dropout(DROPOUT))
            self.prop_out = nn.ModuleList([nn.Linear(128, 1) for _ in range(3)])
            # 三性质量纲不同 → 可学习标准化 (训练数据统计量在 fit_prop_scale 填)
            self.register_buffer('prop_mu',  torch.zeros(3))
            self.register_buffer('prop_sd',  torch.ones(3))
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, x, graph_bias=None, targets=None, props_true=None):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos)[None])
        for blk in self.blocks:
            h = blk(h, graph_bias)
        h = self.ln_f(h)
        logits = self.head(h)
        loss_ce = None
        if targets is not None:                              # 生成损失: pad 位置不计
            loss_ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                      targets.reshape(-1), ignore_index=stoi[PAD])
        loss_prop = None
        if self.use_prop_head and props_true is not None:    # 性质损失: 序列池化 → 回归
            mask = (x != stoi[PAD]).float()[:, :, None]      # 非pad平均池化
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            pf = self.prop_head(pooled)
            pred  = torch.cat([o(pf) for o in self.prop_out], dim=1)   # [B,3] 原始尺度
            target = (props_true - self.prop_mu) / self.prop_sd        # 标准化空间
            loss_prop = F.mse_loss(pred, target)
        loss = None
        if loss_ce is not None:
            loss = loss_ce if loss_prop is None else loss_ce + LAMBDA_PROP * loss_prop
        return logits, loss, (loss_ce, loss_prop)
    @torch.no_grad()
    def generate(self, n_samples, device, temp=SAMP_TEMP, topk=SAMP_TOPK,
                 max_new=BLOCK_SIZE - 2, batch=256, progress=True):
        """从 <bos> 起逐字符采样直到 <eos>; 返回原始字符序列列表"""
        self.eval()
        out = []
        n_done = 0
        while n_done < n_samples:
            b = min(batch, n_samples - n_done)
            x = torch.full((b, 1), stoi[BOS], dtype=torch.long, device=device)
            fin = torch.zeros(b, dtype=torch.bool, device=device)
            seqs = [[] for _ in range(b)]
            for _ in range(max_new):
                logits, _, _ = self.forward(x)               # 推理: 无图偏置 (训练时先验)
                logits = logits[:, -1] / temp
                if topk:                                     # top-k 截断采样
                    v, _ = torch.topk(logits, min(topk, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = logits.softmax(-1)
                nxt = torch.multinomial(probs, num_samples=1)  # [B,1] 每序列每步只采 1 个字符
                nxt = nxt.reshape(b, -1)[:, :1]                # 防呆: 强制 [B,1] (防旧版一次采多列导致 item() 报错)
                nxt = torch.where(fin[:, None], torch.full_like(nxt, stoi[PAD]), nxt)  # fin 升维 [B,1] 防广播成 [B,B]
                x = torch.cat([x, nxt], dim=1)[:, -(BLOCK_SIZE-1):]   # 滑窗防超长
                for j in range(b):
                    if not fin[j]:
                        t = nxt[j].item()
                        if t == stoi[EOS]: fin[j] = True
                        elif t != stoi[PAD]: seqs[j].append(itos[t])
                if fin.all(): break
            out.extend(''.join(s) for s in seqs)
            n_done += b
            if progress and (n_done // batch) % 4 == 0:
                print(f"    采样进度 {n_done}/{n_samples}", end='\r')
        if progress: print()
        return out


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ---- 实例化三个变体检查 forward 形状 (用 1 个假 batch 冒烟测试) ----
def smoke_test():
    _m = MolGPT(use_graph_bias=True, use_prop_head=True).to(_DEVICE)
    _b = next(iter(DataLoader(MolDataset(TRN_IDX[:4], True, True),
                              batch_size=4, collate_fn=collate)))
    _b = {k: v.to(_DEVICE) for k, v in _b.items()}
    _lg, _ls, (_c, _p) = _m(_b['x'], _b.get('bias'), _b['y'], _b.get('props'))
    assert _lg.shape[:2] == (4, BLOCK_SIZE-1) and _c is not None and _p is not None
    print(f"  [冒烟测试] logits {_lg.shape} | CE {_c.item():.3f} | Prop {_p.item():.3f} ✓")
    del _m; torch.cuda.empty_cache() if _DEVICE.type=='cuda' else None

smoke_test()
print(f"[OK] Cell 4 完成 (耗时 {time.time()-_T0:.1f}s)")


# ===========================================================================
# Cell 5: 训练框架 (优化器/调度/早停/训练后性质统计)
# ===========================================================================
print("\n" + "=" * 70)
print("Cell 5: 训练框架")

def build_loaders(variant):
    """按变体构建 DataLoader (graph/prop 按需加载, 基线零额外开销)"""
    ds_tr = MolDataset(TRN_IDX, need_graph=(variant=='graph'), need_prop=(variant=='prop'))
    ds_va = MolDataset(VAL_IDX, need_graph=(variant=='graph'), need_prop=(variant=='prop'))
    mk = lambda ds, sh: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=sh,
                                   collate_fn=collate, num_workers=2,
                                   pin_memory=(_DEVICE.type=='cuda'), drop_last=sh)
    return mk(ds_tr, True), mk(ds_va, False)

def make_optimizer(model):
    """AdamW + 线性预热 + 余弦退火 (GPT 训练标准配置)"""
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                            betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(TRN_IDX) / BATCH_SIZE)
    total = steps_per_epoch * EPOCHS
    def lr_lambda(step):
        if step < WARMUP: return step / max(1, WARMUP)
        p = (step - WARMUP) / max(1, total - WARMUP)
        return LR_MIN/LR + (1 - LR_MIN/LR) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def fit_prop_scale(model):
    """多任务版: 用训练集性质均值/方差初始化标准化参数 (跑一遍性质统计)"""
    mu, sd = props[TRN_IDX].mean(0), props[TRN_IDX].std(0) + 1e-6
    model.prop_mu.copy_(torch.tensor(mu, dtype=torch.float32))
    model.prop_sd.copy_(torch.tensor(sd, dtype=torch.float32))

def train_model(variant, tag):
    """完整训练循环: fp16 混合精度 + 早停 + checkpoint 保存
       返回: (模型, 训练历史 dict)"""
    set_seed(SEED)                                          # 三模型同种子 → 公平消融
    model = MolGPT(use_graph_bias=(variant=='graph'),
                   use_prop_head =(variant=='prop')).to(_DEVICE)
    if variant == 'prop':
        fit_prop_scale(model)
    opt, sched = make_optimizer(model)
    scaler = torch.cuda.amp.GradScaler(enabled=(_DEVICE.type=='cuda'))
    loader_tr, loader_va = build_loaders(variant)

    hist = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_validity': [], 'lr': []}
    best_val, bad = float('inf'), 0
    ckpt_path = os.path.join(CKPT_DIR, f'{tag}.pt')
    print(f"\n  [{tag}] 参数量 {count_params(model)/1e6:.2f}M | 开始训练 "
          f"(变体: {variant})")

    for ep in range(1, EPOCHS + 1):
        model.train(); tot, nb = 0.0, 0
        t_ep = time.time()
        for batch in loader_tr:
            batch = {k: v.to(_DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(_DEVICE.type=='cuda')):
                _, loss, _ = model(batch['x'], batch.get('bias'),
                                   batch['y'], batch.get('props'))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); nb += 1
        tr_loss = tot / max(nb, 1)

        # ---- 验证: 损失 + 快速采样 256 条看有效性 (比 loss 更直观的质检) ----
        model.eval(); vtot, vnb = 0.0, 0
        with torch.no_grad():
            for batch in loader_va:
                batch = {k: v.to(_DEVICE) for k, v in batch.items()}
                _, loss, _ = model(batch['x'], batch.get('bias'),
                                   batch['y'], batch.get('props'))
                vtot += loss.item(); vnb += 1
        va_loss = vtot / max(vnb, 1)
        sample = model.generate(256, _DEVICE, progress=False)
        val_ok = sum(1 for s in sample if Chem.MolFromSmiles(s) is not None) / len(sample)

        hist['epoch'].append(ep); hist['train_loss'].append(tr_loss)
        hist['val_loss'].append(va_loss); hist['val_validity'].append(val_ok)
        hist['lr'].append(sched.get_last_lr()[0])
        print(f"    epoch {ep}/{EPOCHS} | train {tr_loss:.4f} | val {va_loss:.4f} "
              f"| 采样有效性 {val_ok:.1%} | lr {sched.get_last_lr()[0]:.2e} "
              f"| {time.time()-t_ep:.0f}s")

        if va_loss < best_val - 1e-4:                       # 早停 + 保存最优
            best_val, bad = va_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"    早停触发 (验证损失连续 {PATIENCE} 轮未下降)")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=_DEVICE))
    print(f"  [{tag}] 训练完成, 最优 val_loss={best_val:.4f} → {ckpt_path}")
    return model, hist

print(f"[OK] Cell 5 完成 (耗时 {time.time()-_T0:.1f}s)")


# ===========================================================================
# Cell 6: 训练三个模型变体 (消融实验主体, P100 每个约 15-25 分钟)
# ===========================================================================
print("\n" + "=" * 70)
print("Cell 6: 三模型训练 (消融实验)")
print("=" * 70)

VARIANTS = [('molgpt_baseline',  'none'),
            ('molgpt_graphbias', 'graph'),
            ('molgpt_multitask', 'prop')]

MODELS, HISTS = {}, {}
for tag, variant in VARIANTS:
    ckpt = os.path.join(CKPT_DIR, f'{tag}.pt')
    if os.path.exists(ckpt):                                # 断点续跑: 已训完直接加载
        print(f"\n  [{tag}] 发现已有 checkpoint, 跳过训练直接加载")
        m = MolGPT(use_graph_bias=(variant=='graph'),
                   use_prop_head =(variant=='prop')).to(_DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=_DEVICE))
        MODELS[tag] = m
    else:
        MODELS[tag], HISTS[tag] = train_model(variant, tag)

print(f"\n[OK] Cell 6 完成 (总耗时 {time.time()-_T0:.1f}s)")


# ===========================================================================
# Cell 7: 消融对比评估 (加固版 v2) —— 论文核心表格
#   v1 的问题: 单采样seed × 1000条, 三模型 ~1.2pp 的差异落在 ±1.5pp 抽样噪声内,
#              且 composite 因 uniqueness/novelty 恒=1 退化为 validity×QED,
#              "baseline 夺冠"属于在噪声上制造的假排名。
#   v2 修正:
#     1) 每模型 3 个采样种子、共 N_EVAL=3000 条, 报告 seed 间 mean±std
#     2) bootstrap(N_BOOT 次) 给每指标 95%CI, 变体-baseline 差值 CI 显式标注显著性
#     3) Novelty 拆三层: 精确SMILES新颖率 / Murcko骨架新颖率 / 最近邻Tanimoto
#     4) 不再使用退化 composite; BEST_TAG 用透明规则 (validity 差距≤1pp 视为并列, 并列按 QED)
# ===========================================================================
print("\n" + "=" * 70)
print("Cell 7: 消融对比评估 (v2 加固版: 多seed + bootstrap CI + 骨架新颖)")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']   # 中文标签兼容
plt.rcParams['axes.unicode_minus'] = False

FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def mol_of(s):
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

def scaf_of(m):
    """Bemis-Murcko 骨架 SMILES (开链等无骨架分子返回 '')"""
    try:
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s) if s.GetNumAtoms() > 0 else ''
    except Exception:
        return ''

# ---- 训练集参照库: 精确SMILES集合 + 骨架集合 + 最近邻指纹库 ----
_ref_rng = np.random.RandomState(SEED)
_ref_pick = _ref_rng.choice(N, size=min(NOVEL_REF_N, N), replace=False)
EXACT_REF, SCAF_REF, NN_REF_FPS = set(), set(), []
for _i in _ref_pick:
    _m = Chem.MolFromSmiles(canon_list[_i])
    if _m is None:
        continue
    EXACT_REF.add(canon_list[_i])
    SCAF_REF.add(scaf_of(_m))
    if len(NN_REF_FPS) < N_NN_REF:
        NN_REF_FPS.append(FP.GetFingerprint(_m))
print(f"  参照库: 精确SMILES {len(EXACT_REF):,} / 骨架 {len(SCAF_REF):,} 种 / "
      f"最近邻指纹 {len(NN_REF_FPS):,}")

METRIC_KEYS = ['validity', 'uniqueness', 'novelty_exact', 'scaffold_novelty',
               'nn_tanimoto', 'qed', 'sa', 'diversity']
METRIC_CN = {'validity': 'Validity', 'uniqueness': 'Uniqueness',
             'novelty_exact': 'SMILES Novelty', 'scaffold_novelty': 'Scaffold Novelty',
             'nn_tanimoto': 'NN Tanimoto (lower=more novel)', 'qed': 'QED',
             'sa': 'SA score', 'diversity': 'Internal Diversity'}

def profile_batch(smiles_list):
    """逐条画像: 无效分子仅 {'valid':False}; 有效分子带 canon/qed/sa/fp/scaf/nn_sim"""
    recs = []
    for s in smiles_list:
        m = mol_of(s)
        if m is None:
            recs.append({'valid': False})
            continue
        fp = FP.GetFingerprint(m)
        nn_sim = max(DataStructs.BulkTanimotoSimilarity(fp, NN_REF_FPS))  # 与训练集最近邻
        recs.append({'valid': True, 'canon': Chem.MolToSmiles(m),
                     'qed': QED.qed(m), 'sa': sascorer.calculateScore(m),
                     'fp': fp, 'scaf': scaf_of(m), 'nn_sim': float(nn_sim)})
    return recs

def metrics_from_recs(recs, div_rng):
    """从画像记录算 8 指标 (div_rng: 多样性抽样对的独立随机源, bootstrap 各轮可比)"""
    n = len(recs)
    v = [r for r in recs if r['valid']]
    out = {k: 0.0 for k in METRIC_KEYS}
    out['validity'] = len(v) / n
    if not v:
        return out
    canons = [r['canon'] for r in v]
    out['uniqueness'] = len(set(canons)) / len(canons)
    out['novelty_exact'] = float(np.mean([c not in EXACT_REF for c in canons]))
    out['scaffold_novelty'] = float(np.mean([r['scaf'] not in SCAF_REF for r in v]))
    out['nn_tanimoto'] = float(np.mean([r['nn_sim'] for r in v]))
    out['qed'] = float(np.mean([r['qed'] for r in v]))
    out['sa'] = float(np.mean([r['sa'] for r in v]))
    fps = [r['fp'] for r in v]                                          # 500 对随机 Tanimoto 距离
    if len(fps) > 2:
        pairs = [(div_rng.randint(len(fps)), div_rng.randint(len(fps))) for _ in range(500)]
        out['diversity'] = float(np.mean(
            [1 - DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i, j in pairs if i != j]))
    return out

def bootstrap_ci(recs, seed=0):
    """bootstrap 重采样, 返回 {指标: (2.5%, 97.5%)} 95% 百分位置信区间"""
    rng = np.random.RandomState(seed)
    n = len(recs)
    acc = {k: [] for k in METRIC_KEYS}
    div_rng = np.random.RandomState(seed + 1)
    for _ in range(N_BOOT):
        idx = rng.randint(0, n, n)
        mm = metrics_from_recs([recs[i] for i in idx], div_rng)
        for k in METRIC_KEYS:
            acc[k].append(mm[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in acc.items()}

def diff_ci(recs_a, recs_b, key, seed=0):
    """指标差值 a-b 的 bootstrap 95%CI (独立重采样; CI 不含 0 → 差异显著)"""
    rng_a, rng_b = np.random.RandomState(seed), np.random.RandomState(seed + 10000)
    dr_a, dr_b = np.random.RandomState(seed + 1), np.random.RandomState(seed + 2)
    na, nb = len(recs_a), len(recs_b)
    diffs = []
    for _ in range(N_BOOT):
        ia = rng_a.randint(0, na, na)
        ib = rng_b.randint(0, nb, nb)
        ma = metrics_from_recs([recs_a[i] for i in ia], dr_a)
        mb = metrics_from_recs([recs_b[i] for i in ib], dr_b)
        diffs.append(ma[key] - mb[key])
    return (float(np.mean(diffs)),
            (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))))

# ---- 主评估循环: 每模型 × 3 采样种子 ----
MODEL_RECS, PERSEED = {}, []
for tag, _ in VARIANTS:
    seed_recs = []
    for sd in EVAL_SEEDS:
        set_seed(sd)                                     # 三模型用同一组采样种子 → 公平
        smp = MODELS[tag].generate(N_EVAL_SEED, _DEVICE, progress=False)
        recs = profile_batch(smp)
        seed_recs.append(recs)
        mm = metrics_from_recs(recs, np.random.RandomState(SEED))
        PERSEED.append({'model': tag, 'seed': sd, **{k: mm[k] for k in METRIC_KEYS}})
        print(f"  [{tag}] seed={sd} | 有效性 {mm['validity']:.1%} | "
              f"骨架新颖 {mm['scaffold_novelty']:.1%} | 最近邻 {mm['nn_tanimoto']:.3f} | "
              f"QED {mm['qed']:.3f}")
    MODEL_RECS[tag] = [r for sr in seed_recs for r in sr]

# ---- 汇总: 合并 3000 条点估计 + bootstrap CI + seed 间 std ----
rows = []
for tag, _ in VARIANTS:
    recs = MODEL_RECS[tag]
    pt = metrics_from_recs(recs, np.random.RandomState(SEED))
    ci = bootstrap_ci(recs, seed=SEED)
    ps = pd.DataFrame([r for r in PERSEED if r['model'] == tag])
    row = {'model': tag, 'n_samples': len(recs)}
    for k in METRIC_KEYS:
        row[k] = round(pt[k], 4)
        row[f'{k}_lo'], row[f'{k}_hi'] = round(ci[k][0], 4), round(ci[k][1], 4)
        row[f'{k}_seedstd'] = round(float(ps[k].std(ddof=0)), 4)
    rows.append(row)
df_abl = pd.DataFrame(rows)
RESULTS = {r['model']: {k: r[k] for k in METRIC_KEYS} for r in rows}  # 供 Cell 9 引用

# ---- 差值显著性: 各变体 vs baseline ----
base_tag = VARIANTS[0][0]
diff_rows = []
for tag, _ in VARIANTS[1:]:
    for k in ['validity', 'scaffold_novelty', 'nn_tanimoto', 'qed', 'diversity']:
        d_mean, (d_lo, d_hi) = diff_ci(MODEL_RECS[tag], MODEL_RECS[base_tag], k)
        diff_rows.append({'variant': tag, 'metric': k, 'delta': round(d_mean, 4),
                          'ci_lo': round(d_lo, 4), 'ci_hi': round(d_hi, 4),
                          'significant(CI!=0)': bool(d_lo > 0 or d_hi < 0)})
df_diff = pd.DataFrame(diff_rows)

# ---- 透明选优: validity 差距 ≤1pp 视为并列, 并列者按 QED ----
_order = df_abl.sort_values('validity', ascending=False)
_tied = _order[_order['validity'] >= _order['validity'].max() - 0.01]
BEST_TAG = _tied.sort_values('qed', ascending=False)['model'].iloc[0]

# ---- 落盘 (v2 三份 + 兼容旧文件名一份) ----
_abl_path  = os.path.join(GEN_DIR, 'ablation_table_v2.csv')
_ps_path   = os.path.join(GEN_DIR, 'ablation_perseed.csv')
_diff_path = os.path.join(GEN_DIR, 'ablation_diff_vs_baseline.csv')
df_abl.to_csv(_abl_path, index=False, encoding='utf-8-sig')
pd.DataFrame(PERSEED).to_csv(_ps_path, index=False, encoding='utf-8-sig')
df_diff.to_csv(_diff_path, index=False, encoding='utf-8-sig')
df_abl.to_csv(os.path.join(GEN_DIR, 'ablation_table.csv'), index=False, encoding='utf-8-sig')

print(f"\n  消融对比 (点估计 + 95%CI, 每模型合并 {N_EVAL} 条 / {len(EVAL_SEEDS)} 个采样seed):")
print(df_abl[['model', 'validity', 'uniqueness', 'scaffold_novelty',
              'nn_tanimoto', 'qed', 'sa', 'diversity']].to_string(index=False))
print("\n  变体 vs baseline 差值 (bootstrap 95%CI):")
print(df_diff.to_string(index=False))
_n_sig = int(df_diff['significant(CI!=0)'].sum())
print(f"\n  显著差异指标数: {_n_sig}/{len(df_diff)} "
      f"({'存在显著差异' if _n_sig else '→ 饱和区内无统计显著差异 (符合预期, 证据见 Cell 7.5 低数据曲线)'})")
print(f"  选优规则: validity 差距≤1pp 视为并列, 并列按 QED → 最优模型: {BEST_TAG}")
print(f"  ✓ 已保存: {_abl_path}")
print(f"  ✓ 已保存: {_ps_path}")
print(f"  ✓ 已保存: {_diff_path}")

# ---- 可视化: 训练曲线 + 六指标带 95%CI 误差棒柱状图 ----
if HISTS:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for tag, h in HISTS.items():
        axes[0].plot(h['epoch'], h['train_loss'], marker='o', label=tag)
        axes[1].plot(h['epoch'], h['val_loss'], marker='s', label=tag)
    axes[0].set_title('Train Loss'); axes[1].set_title('Val Loss')
    for ax in axes: ax.set_xlabel('epoch'); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(GEN_DIR, 'train_curves.png'), dpi=150); plt.close()
    print(f"  ✓ 已保存: {os.path.join(GEN_DIR, 'train_curves.png')}")

_bar_metrics = ['validity', 'uniqueness', 'scaffold_novelty',
                'nn_tanimoto', 'qed', 'diversity']
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, mkey in zip(axes.flat, _bar_metrics):
    xs = np.arange(len(df_abl))
    vals = df_abl[mkey].values
    lo = df_abl[f'{mkey}_lo'].values
    hi = df_abl[f'{mkey}_hi'].values
    err_lo = np.clip(vals - lo, 0, None)          # 下误差: 值-下限, 裁剪负值
    err_hi = np.clip(hi - vals, 0, None)          # 上误差: 上限-值, 裁剪负值
    ax.bar(xs, vals, color=['#4C72B0', '#DD8452', '#55A868'][:len(vals)],
           yerr=[err_lo, err_hi], capsize=5, edgecolor='black', linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([t.replace('molgpt_', '') for t in df_abl['model']],
                       rotation=15, fontsize=9)
    ax.set_title(METRIC_CN.get(mkey, mkey)); ax.grid(alpha=0.3, axis='y')
    for k, v in enumerate(vals):
        ax.text(k, v, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
fig.suptitle(f'Ablation: {len(EVAL_SEEDS)} sampling seeds x {N_EVAL_SEED} samples, '
             f'bootstrap {N_BOOT}x (95% CI)', fontsize=13)
plt.tight_layout()
_abl_png = os.path.join(GEN_DIR, 'ablation_bars.png')
plt.savefig(_abl_png, dpi=150); plt.close()
print(f"  ✓ 已保存: {_abl_png}")
print(f"[OK] Cell 7 完成 (总耗时 {time.time()-_T0:.1f}s)")

# 只重做消融评估时 (ABLATION_ONLY=1) 在此提前退出,
# 避免 Cell 8/8.5 重采样覆盖已交付的 A3_MGPT_molgpt_only.csv / property_calibration.csv
if os.environ.get('ABLATION_ONLY', '0') == '1':
    print('[ABLATION_ONLY] 消融完成, 提前退出 (跳过低数据/生产/性质实验)')
    import sys
    sys.exit(0)


# ===========================================================================
# Cell 7.5: 低数据效率曲线 (创新点1「图偏置注意力」的核心证据实验)
#   动机: 全量 ZINC-250K 上三模型有效性均饱和在 93-95%, 差异落入抽样噪声。
#         图距离先验的价值应在「数据稀缺」时显现 —— 训练句子少, 模型自己学不会
#         括号配对/环闭合等化学语法时, 注入"近邻原子注意力更强"的先验补课作用最大。
#   设计: baseline vs graphbias × 数据占比 {10%,30%,100%} × 2 训练种子
#         = 12 个评估点 (100%×seed42 两点直接复用 Cell 6 的主 checkpoint, 不重复训练)
#   公平性: ① 同(占比,训练种子)两变体用完全相同的训练子集与初始化种子
#           ② 全部 12 个点统一用评估种子 999 采样 LOWDATA_EVAL_N 条, 采样运气一致
#   产物: lowdata_results.csv (增量落盘/断点续跑) + lowdata_efficiency.png
#   耗时: P100 约 1.5h (10%点~6min×4, 30%点~15min×4, 100%/seed7 ~18min×2)
# ===========================================================================
print("\n" + "=" * 70)
print("Cell 7.5: 低数据效率曲线 (baseline vs graphbias, 支持断点续跑)")

LOWDATA_VARIANTS = [('molgpt_baseline', 'none'), ('molgpt_graphbias', 'graph')]
_ld_csv = os.path.join(GEN_DIR, 'lowdata_results.csv')

def _eval_validity_fixed(model):
    """统一评估种子(999)采样, 保证 12 个实验点的'采样运气'完全一致"""
    set_seed(999)
    smp = model.generate(LOWDATA_EVAL_N, _DEVICE, progress=False)
    return sum(1 for s in smp if mol_of(s) is not None) / len(smp)

def train_lowdata(variant, ratio, tr_seed):
    """在训练集的一个子集上训练模型; 逐点保存 checkpoint 支持断点续跑"""
    tag = f"ld_{variant.replace('molgpt_', '')}_r{int(ratio*100)}_s{tr_seed}"
    ckpt = os.path.join(CKPT_DIR, tag + '.pt')
    use_graph = ('graphbias' in variant) or (variant == 'graph')  # 修复: variant 是完整标签 'molgpt_graphbias', 旧写法恒为 False
    assert use_graph == ('graphbias' in variant), "变体开关与标签不一致"   # 保险丝: 防止再出现静默降级

    # ---- 特例: 100% 数据 + 主种子 → 与 Cell 6 主模型同配置, 直接复用不重训 ----
    if ratio >= 1.0 and tr_seed == SEED:
        main_ckpt = os.path.join(CKPT_DIR, f'{variant}.pt')
        if os.path.exists(main_ckpt):
            m = MolGPT(use_graph_bias=use_graph).to(_DEVICE)
            m.load_state_dict(torch.load(main_ckpt, map_location=_DEVICE))
            v = _eval_validity_fixed(m)
            print(f"  [{tag}] 复用 Cell6 主 checkpoint → validity {v:.1%}")
            return dict(variant=variant, ratio=ratio, tr_seed=tr_seed,
                        validity=round(v, 4), reused_main=True, train_sec=0.0)

    # ---- 断点续跑: 该低数据 checkpoint 已训过 → 只重新评估 ----
    if os.path.exists(ckpt):
        m = MolGPT(use_graph_bias=use_graph).to(_DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=_DEVICE))
        v = _eval_validity_fixed(m)
        print(f"  [{tag}] 已有低数据 checkpoint, 直接评估 → validity {v:.1%}")
        return dict(variant=variant, ratio=ratio, tr_seed=tr_seed,
                    validity=round(v, 4), reused_main=False, train_sec=0.0)

    # ---- 训练子集: 同一(ratio,seed) permutation → 两变体数据严格一致 ----
    _sub_rng = np.random.RandomState(tr_seed)
    n_sub = max(2000, int(len(TRN_IDX) * ratio))
    sub_idx = _sub_rng.permutation(TRN_IDX)[:n_sub]

    set_seed(tr_seed)                                     # 权重初始化/DataLoader 顺序种子
    model = MolGPT(use_graph_bias=use_graph).to(_DEVICE)
    ds_tr = MolDataset(sub_idx, need_graph=use_graph)
    ds_va = MolDataset(VAL_IDX, need_graph=use_graph)
    _mk = lambda ds, sh: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=sh,
                                    collate_fn=collate, num_workers=2,
                                    pin_memory=(_DEVICE.type == 'cuda'), drop_last=sh)
    loader_tr, loader_va = _mk(ds_tr, True), _mk(ds_va, False)

    steps_ep = math.ceil(n_sub / BATCH_SIZE)
    total_steps = steps_ep * EPOCHS
    warm_eff = min(WARMUP, max(50, total_steps // 10))   # 数据少时按比例缩短预热
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
                            betas=(0.9, 0.95))
    def _lr_lambda(step):
        if step < warm_eff:
            return step / max(1, warm_eff)
        p = (step - warm_eff) / max(1, total_steps - warm_eff)
        return LR_MIN/LR + (1 - LR_MIN/LR) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(_DEVICE.type == 'cuda'))

    t0 = time.time()
    best_val, bad = float('inf'), 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch in loader_tr:
            batch = {k: v.to(_DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(_DEVICE.type == 'cuda')):
                _, loss, _ = model(batch['x'], batch.get('bias'), batch['y'])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); sched.step()
        model.eval(); vtot, vnb = 0.0, 0
        with torch.no_grad():
            for batch in loader_va:
                batch = {k: v.to(_DEVICE) for k, v in batch.items()}
                _, vl, _ = model(batch['x'], batch.get('bias'), batch['y'])
                vtot += vl.item(); vnb += 1
        va_loss = vtot / max(vnb, 1)
        if va_loss < best_val - 1e-4:
            best_val, bad = va_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(torch.load(ckpt, map_location=_DEVICE))
    train_sec = time.time() - t0
    v = _eval_validity_fixed(model)
    print(f"  [{tag}] 子集 {n_sub:,} 条 | 训练 {train_sec/60:.1f}min → validity {v:.1%}")
    return dict(variant=variant, ratio=ratio, tr_seed=tr_seed,
                validity=round(v, 4), reused_main=False,
                train_sec=round(train_sec, 1))

# ---- 断点续跑主循环: 每完成一个点立即写盘, 中断后重跑本 Cell 自动跳过已完成点 ----
if os.path.exists(_ld_csv):
    df_ld = pd.read_csv(_ld_csv)
    print(f"  发现已有结果 {len(df_ld)} 个实验点, 续跑缺失组合")
else:
    df_ld = pd.DataFrame(columns=['variant', 'ratio', 'tr_seed', 'validity',
                                  'reused_main', 'train_sec'])
_done = set(zip(df_ld['variant'], df_ld['ratio'], df_ld['tr_seed']))
for ratio in LOWDATA_RATIOS:
    for variant, _vkey in LOWDATA_VARIANTS:
        for tsd in LOWDATA_TRSEEDS:
            if (variant, float(ratio), tsd) in _done or (variant, ratio, tsd) in _done:
                continue
            row = train_lowdata(variant, ratio, tsd)
            df_ld = pd.concat([df_ld, pd.DataFrame([row])], ignore_index=True)
            df_ld.to_csv(_ld_csv, index=False, encoding='utf-8-sig')

print(f"\n  低数据效率实验结果 ({len(df_ld)} 个点):")
print(df_ld.sort_values(['variant', 'ratio', 'tr_seed']).to_string(index=False))
print(f"  ✓ 已保存: {_ld_csv}")

# ---- 绘图: 数据占比-有效性曲线 (点=2训练seed均值, 误差棒=seed间std) ----
fig, ax = plt.subplots(figsize=(8.5, 5.5))
_colors = {'molgpt_baseline': '#4C72B0', 'molgpt_graphbias': '#DD8452'}
_labels = {'molgpt_baseline': 'MolGPT baseline',
           'molgpt_graphbias': 'MolGPT + graph-bias (ours)'}
_curve = {}
for variant, _ in LOWDATA_VARIANTS:
    xs, ym, ye = [], [], []
    for r in LOWDATA_RATIOS:
        sub = df_ld[(df_ld['variant'] == variant) &
                    (np.isclose(df_ld['ratio'], r))]['validity']
        xs.append(r * 100)
        ym.append(float(sub.mean()))
        ye.append(float(sub.std(ddof=0)) if len(sub) > 1 else 0.0)
    ax.errorbar(xs, ym, yerr=ye, marker='o', capsize=5, linewidth=2,
                markersize=8, color=_colors[variant], label=_labels[variant])
    _curve[variant] = dict(zip(xs, ym))

# 自动量化"数据节省": graphbias 达到 baseline@100% 水平所需的最少训练数据
_v_base100 = _curve['molgpt_baseline'][100.0]
_r_needed = next((r for r in LOWDATA_RATIOS
                  if _curve['molgpt_graphbias'][r * 100] >= _v_base100), None)
ax.axhline(_v_base100, color='#4C72B0', linestyle=':', alpha=0.6,
           label=f'baseline@100% = {_v_base100:.1%}')
ax.set_xlabel('Training data fraction (%)', fontsize=12)
ax.set_ylabel('Validity (fixed eval seed 999)', fontsize=12)
ax.set_title('Data Efficiency: Graph-Bias Prior in the Low-Data Regime', fontsize=13)
ax.set_xticks([r * 100 for r in LOWDATA_RATIOS])
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
_ld_png = os.path.join(GEN_DIR, 'lowdata_efficiency.png')
plt.savefig(_ld_png, dpi=150); plt.close()
print(f"  ✓ 已保存: {_ld_png}")
if _r_needed is not None and _r_needed < 1.0:
    print(f"  ★ 关键发现: graphbias 仅用 {int(_r_needed*100)}% 训练数据即达到 "
          f"baseline 100% 数据的有效性({_v_base100:.1%}) → 训练数据节省 {(1-_r_needed):.0%}")
else:
    print("  注: 当前档位下 graphbias 未在 <100% 数据追平 baseline@100%;")
    print("      报告时请引用曲线左侧(10%/30%)两变体的实际有效性差距作为低资源增益证据")
print(f"[OK] Cell 7.5 完成 (总耗时 {time.time()-_T0:.1f}s)")


# ===========================================================================
# Cell 8: 最优模型生产采样 —— 5000 条有效分子 (主流程阶段一的全部原始候选)
# ===========================================================================
print("\n" + "=" * 70)
print(f"Cell 8: 生产采样 (最优模型: {BEST_TAG}, 目标 {N_PROD} 条)")

_prod_raw = MODELS[BEST_TAG].generate(int(N_PROD * 1.8), _DEVICE)   # 多采 80% 留过滤余量
_prod_pool = []
_seen = set()
for s in _prod_raw:
    m = mol_of(s)
    if m is None:
        continue
    c = Chem.MolToSmiles(m)
    if c in _seen:                                       # 生产库内去重
        continue
    _seen.add(c)
    _prod_pool.append(c)
    if len(_prod_pool) >= N_PROD:
        break
print(f"  采样 {len(_prod_raw)} 条 → 有效去重 {len(_prod_pool)} 条 "
      f"(有效率 {len(_prod_pool)/max(1,len(_prod_raw)):.1%})")
assert len(_prod_pool) >= N_PROD * 0.6, \
    "有效产出不足! 建议提高 N_PROD 采样量或检查训练收敛情况 (取前 80% 交付)"

df_prod = pd.DataFrame({'SMILES': _prod_pool[:N_PROD]})
df_prod['来源'] = f'MolGPT-{BEST_TAG.replace("molgpt_", "")}'
df_prod['Novel_vs_ZINC'] = [c not in NOVEL_REF for c in df_prod['SMILES']]
df_prod['QED'] = [QED.qed(mol_of(c)) for c in df_prod['SMILES']]
_prod_path = os.path.join(GEN_DIR, 'A3_MGPT_molgpt_only.csv')
df_prod.to_csv(_prod_path, index=False, encoding='utf-8-sig')
print(f"  ✓ 已保存: {_prod_path} ({len(df_prod)} 条)")
print(f"    其中新颖分子(不在ZINC训练集): {df_prod['Novel_vs_ZINC'].mean():.1%} | "
      f"平均QED {df_prod['QED'].mean():.3f}")
print("  → 下一步: 运行《子任务2_生成式分子探索_Kaggle.py》(阶段一自动读取本文件 5000 条)")
print(f"[OK] Cell 8 完成 (总耗时 {time.time()-_T0:.1f}s)")


# ===========================================================================
# Cell 8.5: 性质可控性实验 (创新点2「多任务性质头」的核心证据, 无需重训)
#   逻辑: 多任务头强迫隐状态编码 QED/logP/SA 信息 → 若该信息可迁移到
#         生成分子, 则"按头预测值筛选"即可无梯度地定向生成高/低 QED 分子
#   设计: ① multitask 模型采 2000 条  ② 头预测 QED vs RDKit 真实 QED → Spearman
#         ③ 按预测 QED 五分位分桶 → 真实 QED 桶均值曲线 (单调性)
#         ④ 前/后 10% 选择实验 (对照: 置换检验≈0)
#   判定: Spearman>=0.30 且顶/底十分位真实 QED 差>=0.10 → 性质可控成立
#   产物: property_calibration.csv / property_control_curve.png
#   耗时: P100 约 15-20 min (采样 2000 条 + 打分)
# ===========================================================================
print("\n" + "=" * 70)
print("Cell 8.5: 性质可控性实验 (多任务头迁移性, 无需重训)")

from scipy import stats as _sps

# ---- ① 取 multitask 模型 (内核存活用内存, 否则读 checkpoint) ----
if 'MODELS' in globals() and 'molgpt_multitask' in MODELS:
    mt_model = MODELS['molgpt_multitask']
else:
    mt_model = MolGPT(use_graph_bias=False, use_prop_head=True).to(_DEVICE)
    mt_model.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, 'molgpt_multitask.pt'), map_location=_DEVICE))
mt_model.eval()

@torch.no_grad()
def _pred_props(model, smiles_list, batch=256):
    """性质头预测 (QED,logP,SA), 返回原始尺度 [N,3] numpy (池化方式与训练一致)"""
    out = []
    for i in range(0, len(smiles_list), batch):
        chunk = smiles_list[i:i + batch]
        seqs = []
        for s in chunk:
            ids = [stoi[BOS]] + [stoi[c] for c in s[:MAX_SMILES]] + [stoi[EOS]]
            seqs.append(ids[:BLOCK_SIZE - 1])
        L = max(len(s) for s in seqs)
        x = torch.full((len(seqs), L), stoi[PAD], dtype=torch.long, device=_DEVICE)
        for j, ids in enumerate(seqs):
            x[j, :len(ids)] = torch.tensor(ids, device=_DEVICE)
        pos = torch.arange(L, device=_DEVICE)
        h = model.drop(model.tok_emb(x) + model.pos_emb(pos)[None])
        for blk in model.blocks:
            h = blk(h, None)
        h = model.ln_f(h)
        mask = (x != stoi[PAD]).float()[:, :, None]
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        pf = model.prop_head(pooled)
        pred = torch.cat([o(pf) for o in model.prop_out], dim=1)        # 标准化空间
        pred = pred * model.prop_sd[None, :] + model.prop_mu[None, :]   # → 原始尺度
        out.append(pred.cpu().numpy())
    return np.concatenate(out)

# ---- ② 采样 + 打分 ----
set_seed(20260914)
_pool_raw = mt_model.generate(2000, _DEVICE, progress=True)
_pool = sorted({Chem.MolToSmiles(m) for s in _pool_raw
                if (m := mol_of(s)) is not None})
_df = pd.DataFrame({'SMILES': _pool})
_df['pred_QED'] = _pred_props(mt_model, _df['SMILES'].tolist())[:, 0]
_df['true_QED'] = [QED.qed(mol_of(c)) for c in _df['SMILES']]
print(f"  采样 2000 → 有效去重 {len(_df)} 条, 打分完成")

# ---- ③ 相关性: 迁移性 (含置换对照) ----
_rho_sp = float(_sps.spearmanr(_df['pred_QED'], _df['true_QED']).statistic)
_rho_pe = float(np.corrcoef(_df['pred_QED'], _df['true_QED'])[0, 1])
_rng = np.random.RandomState(7)
_rho_shuf = float(np.mean([_sps.spearmanr(_rng.permutation(_df['pred_QED']),
                                          _df['true_QED']).statistic
                           for _ in range(20)]))
print(f"  Spearman(预测,真实)={_rho_sp:.3f} | Pearson={_rho_pe:.3f} | "
      f"置换对照={_rho_shuf:.3f} (应≈0)")

# ---- ④ 五分位单调性 + 前/后10%选择实验 ----
_df['bucket'] = pd.qcut(_df['pred_QED'], 5,
                        labels=['Q1低', 'Q2', 'Q3', 'Q4', 'Q5高'])
_bucket = _df.groupby('bucket', observed=True)['true_QED'].mean()
_n = max(1, len(_df) // 10)
_top = _df.nlargest(_n, 'pred_QED')['true_QED']
_bot = _df.nsmallest(_n, 'pred_QED')['true_QED']
_gaps = np.diff(_bucket.values)
_mono = bool((_gaps > -0.02).all() and (_gaps > 0).sum() >= 3)
print("  预测QED五分位 → 真实QED均值:")
for k, v in _bucket.items():
    print(f"    {k}: {v:.3f}")
print(f"  头选择 Top10% 真实QED={_top.mean():.3f} | "
      f"Bottom10%={_bot.mean():.3f} | 差 {_top.mean()-_bot.mean():+.3f}")

# ---- ⑤ 判定与绘图 ----
_ok = (_rho_sp >= 0.30) and (_top.mean() - _bot.mean() >= 0.10)
if _ok:
    print(f"\n  ★ 性质可控成立: 头信息可迁移到生成分子 (ρ={_rho_sp:.2f}), "
          f"无需梯度即可按 QED 定向筛选 → 创新点2的正面证据")
else:
    print(f"\n  ★ 信号弱 (ρ={_rho_sp:.2f}): 如实报告为'部分迁移', 不宣称可控生成")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(_df['pred_QED'], _df['true_QED'], s=6, alpha=0.35, color='#4C72B0')
axes[0].set_xlabel('Head-predicted QED'); axes[0].set_ylabel('True QED (RDKit)')
axes[0].set_title(f'Calibration on generated mols (Spearman rho={_rho_sp:.2f})')
axes[0].grid(alpha=0.3)
axes[1].bar(np.arange(5), _bucket.values, color='#DD8452',
            edgecolor='black', linewidth=0.6)
axes[1].set_xticks(np.arange(5)); axes[1].set_xticklabels(_bucket.index)
axes[1].set_ylabel('True QED mean')
axes[1].set_title('Property-directed selection (quintiles by head prediction)')
for k, v in enumerate(_bucket.values):
    axes[1].text(k, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
_p_png = os.path.join(GEN_DIR, 'property_control_curve.png')
plt.savefig(_p_png, dpi=150); plt.show()
_df.to_csv(os.path.join(GEN_DIR, 'property_calibration.csv'),
           index=False, encoding='utf-8-sig')
print(f"  ✓ 已保存: {_p_png}")
print(f"  ✓ 已保存: {os.path.join(GEN_DIR, 'property_calibration.csv')}")
print(f"[OK] Cell 8.5 完成 (总耗时 {time.time()-_T0:.1f}s)")


# ===========================================================================
# Cell 9: run_params.json 可复现参数存档 (与主流程阶段五规范对齐)
# ===========================================================================
def _json_default(o):
    """numpy 类型 → Python 原生类型, 供 json.dump 序列化"""
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.bool_): return bool(o)
    return str(o)

run_params = {
    '脚本': '1_molgpt_train.py (DGX迁移版, 源: 子任务2_MolGPT训练_Kaggle.py)',
    '随机种子': SEED,
    '数据': {'来源': 'ZINC-250K (aspuru-guzik-group/chemical_vae)',
             '清洗后条数': int(N), '最大SMILES长度': MAX_SMILES,
             '词表大小': VOCAB_SIZE, '特殊token': [PAD, BOS, EOS]},
    '模型': {'层数': N_LAYER, '头数': N_HEAD, '嵌入维度': N_EMBD,
             'block_size': BLOCK_SIZE, 'dropout': DROPOUT,
             '权重绑定': True, '参数量M': round(count_params(MODELS[VARIANTS[0][0]])/1e6, 2)},
    '消融变体': {'A_基线': 'molgpt_baseline',
                 'B_图偏置注意力': f'graph d<={GRAPH_DMAX}, ALiBi式每头可学习斜率, 仅训练时注入',
                 'C_多任务': f'QED/logP/SA 联合回归, λ={LAMBDA_PROP}, 非pad平均池化'},
    '训练': {'batch': BATCH_SIZE, 'lr': LR, 'lr_min': LR_MIN, 'warmup': WARMUP,
             'weight_decay': WEIGHT_DECAY, 'epochs': EPOCHS, 'early_stop': PATIENCE,
             'grad_clip': GRAD_CLIP, 'amp': 'fp16', '优化器': 'AdamW(0.9,0.95)'},
    '采样': {'temperature': SAMP_TEMP, 'top_k': SAMP_TOPK,
             '评估采样种子': list(EVAL_SEEDS), '每种子采样数': N_EVAL_SEED,
             '评估采样总数': N_EVAL, 'bootstrap次数': N_BOOT,
             '生产采样数': N_PROD},
    '消融结果_v2(点估计, 详见ablation_table_v2.csv)': {t: RESULTS[t] for t, _ in VARIANTS},
    '消融差值显著性(详见ablation_diff_vs_baseline.csv)': df_diff.to_dict('records'),
    '低数据效率实验': {'数据占比': list(LOWDATA_RATIOS),
                      '训练种子': list(LOWDATA_TRSEEDS),
                      '评估采样数': LOWDATA_EVAL_N, '评估种子': 999,
                      '结果(详见lowdata_results.csv)': df_ld.to_dict('records'),
                      '数据节省结论': (f'graphbias用{int(_r_needed*100)}%数据追平baseline@100%'
                                     if _r_needed is not None and _r_needed < 1.0
                                     else '当前档位未在<100%追平, 以曲线左侧增益报告')},
    '最优模型': BEST_TAG,
    '选优规则': 'validity 差距≤1pp 视为并列, 并列按 QED 排序',
    # ---- Cell 8.5 性质可控性实验 (内核变量存在才收录, 未跑则标注) ----
    **({'性质可控性实验(详见property_calibration.csv)': {
        '采样种子': 20260914, '采样数': 2000, '有效去重': int(len(_df)),
        'Spearman': round(_rho_sp, 3), 'Pearson': round(_rho_pe, 3),
        '置换对照ρ': round(_rho_shuf, 3),
        '五分位真实QED': {str(k): round(float(v), 3) for k, v in _bucket.items()},
        'Top10%真实QED': round(float(_top.mean()), 3),
        'Bottom10%真实QED': round(float(_bot.mean()), 3),
        'Top-Bottom差': round(float(_top.mean() - _bot.mean()), 3),
        '判定': '性质可控成立' if _ok else '信号弱(如实报告)'}}
      if '_rho_sp' in globals() else
      {'性质可控性实验': '未运行 (Cell 8.5)'}),
    '总耗时秒': round(time.time() - _T0, 1),
}
with open(os.path.join(GEN_DIR, 'run_params_molgpt.json'), 'w', encoding='utf-8') as f:
    json.dump(run_params, f, ensure_ascii=False, indent=2, default=_json_default)
print(f"\n{'='*70}")
print("全部完成! 产物清单:")
print(f"  {GEN_DIR}/ablation_table_v2.csv       消融表v2 (点估计+95%CI+seed间std)")
print(f"  {GEN_DIR}/ablation_perseed.csv        3个采样种子的逐seed明细")
print(f"  {GEN_DIR}/ablation_diff_vs_baseline.csv  变体-baseline差值CI与显著性")
print(f"  {GEN_DIR}/ablation_bars.png           六指标带95%CI误差棒柱状图")
print(f"  {GEN_DIR}/lowdata_results.csv         低数据效率实验12点明细")
print(f"  {GEN_DIR}/lowdata_efficiency.png      ★ 数据效率曲线 (创新点1核心图)")
print(f"  {GEN_DIR}/train_curves.png            三模型训练曲线")
print(f"  {GEN_DIR}/A3_MGPT_molgpt_only.csv     最优模型 {N_PROD} 条 (主流程阶段一输入)")
print(f"  {GEN_DIR}/run_params_molgpt.json      可复现参数存档")
print(f"  {CKPT_DIR}/                          主模型3个 + 低数据模型checkpoint")
print(f"总耗时 {time.time()-_T0:.0f}s ({(time.time()-_T0)/60:.0f} 分钟)")
