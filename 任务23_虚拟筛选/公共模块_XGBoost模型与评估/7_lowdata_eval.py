#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================
PPT 第4页 A 图补充: 低数据效率曲线新增 5%/20%/50%/70% 四档实验点 (独立安全版)
=================================================================================
为什么需要本脚本:
  直接重跑 1_molgpt_train.py 会在 Cell 8 重新采样并覆盖 5000 生产库,
  且新增训练消耗随机数后采样流与原始库不一致。
  本脚本只复刻 Cell 7.5 的训练/评估逻辑, 复用 molgpt_data/ 预处理产物,
  不触碰主模型 checkpoint、不重采样生产库、不改任何已有结果;
  仅向 output_molgpt/lowdata_results.csv 增量追加新实验点。

公平性 (与原脚本完全一致):
  - 验证划分 RandomState(42) 同原口径
  - 同(占比, 训练种子)两个变体用同一训练子集
  - 所有点统一评估种子 999、采样 1000 条测 validity
  - 已存在于 CSV 的组合自动跳过 (断点续跑)

运行 (服务器, 预计 DGX/GPU 约 2~3 小时):
  cd ~/ASPA_planB
  python scripts/6_lowdata_extra.py

跑完后把 output_molgpt/lowdata_results.csv 下载回本地,
用 PPT图表_模型调优_v2/plot_a.py 一键出新 A 图 (含 7 点 + 100% 外推虚线)。
=================================================================================
"""
import os, sys, math, time, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# ---------------- 路径 (与 1_molgpt_train.py 同锚定规则) ----------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SCRIPT_DIR) if os.path.isdir(
    os.path.join(os.path.dirname(_SCRIPT_DIR), 'data')) else _SCRIPT_DIR
GEN_DIR  = os.path.join(BASE_DIR, 'output_molgpt')
CKPT_DIR = os.path.join(BASE_DIR, 'ckpt')
DATA_DIR = os.path.join(BASE_DIR, 'molgpt_data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[环境] BASE={BASE_DIR} | device={DEVICE}")

# ---------------- 全局超参 (逐字复制 Cell 1) ----------------
SEED = 42
BLOCK_SIZE = 104
N_LAYER, N_HEAD, N_EMBD, DROPOUT, GRAPH_DMAX = 4, 8, 256, 0.1, 6
BATCH_SIZE, LR, LR_MIN, WARMUP, WEIGHT_DECAY = 128, 3e-4, 3e-5, 300, 0.01
EPOCHS, PATIENCE, GRAD_CLIP = 4, 2, 1.0
LOWDATA_EVAL_N = 1000
NEW_RATIOS = (0.05, 0.20, 0.50, 0.70)          # 本次新增档位 (10/30/100 已有)
LOWDATA_VARIANTS = [('molgpt_baseline', False), ('molgpt_graphbias', True)]
LOWDATA_TRSEEDS = (42, 7)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------------- 加载预处理产物 + 重建词表 ----------------
tokens = np.load(os.path.join(DATA_DIR, 'tokens.npy'))
_gp = np.load(os.path.join(DATA_DIR, 'graph_pairs.npz'))
gp_flat, gp_ptr = _gp['gp_flat'], _gp['gp_ptr']
canon_df = pd.read_csv(os.path.join(DATA_DIR, 'smiles_clean.csv'))
canon_list = canon_df['SMILES'].astype(str).tolist()
N, L = tokens.shape
PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
char_set = sorted({ch for smi in canon_list for ch in smi})
stoi = {PAD: 0, BOS: 1, EOS: 2}
stoi.update({ch: i + 3 for i, ch in enumerate(char_set)})
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(stoi)
print(f"[数据] N={N} L={L} vocab={VOCAB_SIZE}")

perm = np.random.RandomState(SEED).permutation(N)
n_val = max(2000, int(N * 0.02))
VAL_IDX, TRN_IDX = perm[:n_val], perm[n_val:]

# ---------------- 模型 (逐字复制 Cell 3/4) ----------------
class MolDataset(Dataset):
    def __init__(self, idx_array, need_graph=False):
        self.idx = idx_array; self.need_graph = need_graph
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, k):
        i = self.idx[k]
        x = torch.from_numpy(tokens[i].astype(np.int64))
        item = {'x': x[:-1], 'y': x[1:]}
        if self.need_graph:
            s, e = gp_ptr[i], gp_ptr[i+1]
            bias = np.zeros((L-1, L-1), dtype=np.float32)
            if e > s:
                pr = gp_flat[s:e].astype(np.int64) + 1
                bias[pr[:,0], pr[:,1]] = -pr[:,2]
                bias[pr[:,1], pr[:,0]] = -pr[:,2]
            item['bias'] = torch.from_numpy(bias)
        return item

def collate(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

class GraphBiasAttention(nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.n_head = n_head; self.d_head = N_EMBD // n_head
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.attn_drop = nn.Dropout(DROPOUT); self.resid_drop = nn.Dropout(DROPOUT)
        self.slope_raw = nn.Parameter(torch.zeros(n_head) + 0.1)
    def forward(self, x, graph_bias=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if graph_bias is not None:
            att = att + F.softplus(self.slope_raw)[None, :, None, None] * graph_bias[:, None, :, :]
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = self.attn_drop(att.softmax(dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))

class Block(nn.Module):
    def __init__(self, use_graph_bias):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = GraphBiasAttention(N_HEAD)
        self.use_graph_bias = use_graph_bias
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT))
    def forward(self, x, graph_bias=None):
        x = x + self.attn(self.ln1(x), graph_bias if self.use_graph_bias else None)
        x = x + self.mlp(self.ln2(x))
        return x

class MolGPT(nn.Module):
    def __init__(self, use_graph_bias=False):
        super().__init__()
        self.use_graph_bias = use_graph_bias
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE - 1, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(use_graph_bias) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, x, graph_bias=None, targets=None):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos)[None])
        for blk in self.blocks:
            h = blk(h, graph_bias)
        h = self.ln_f(h)
        logits = self.head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                                   targets.reshape(-1), ignore_index=stoi[PAD])
        return logits, loss
    @torch.no_grad()
    def generate(self, n_samples, device, temp=0.8, topk=40,
                 max_new=BLOCK_SIZE - 2, batch=256, progress=False):
        self.eval(); out = []; n_done = 0
        while n_done < n_samples:
            b = min(batch, n_samples - n_done)
            x = torch.full((b, 1), stoi[BOS], dtype=torch.long, device=device)
            fin = torch.zeros(b, dtype=torch.bool, device=device)
            seqs = [[] for _ in range(b)]
            for _ in range(max_new):
                logits, _ = self.forward(x)
                logits = logits[:, -1] / temp
                if topk:
                    vv, _ = torch.topk(logits, min(topk, logits.size(-1)))
                    logits[logits < vv[:, [-1]]] = float('-inf')
                probs = logits.softmax(-1)
                nxt = torch.multinomial(probs, num_samples=1).reshape(b, -1)[:, :1]
                nxt = torch.where(fin[:, None], torch.full_like(nxt, stoi[PAD]), nxt)
                x = torch.cat([x, nxt], dim=1)[:, -(BLOCK_SIZE-1):]
                for j in range(b):
                    if not fin[j]:
                        t = nxt[j].item()
                        if t == stoi[EOS]: fin[j] = True
                        elif t != stoi[PAD]: seqs[j].append(itos[t])
                if fin.all(): break
            out.extend(''.join(s) for s in seqs); n_done += b
        return out

# ---------------- 低数据训练/评估 (复刻 train_lowdata) ----------------
def eval_validity(model):
    set_seed(999)
    smp = model.generate(LOWDATA_EVAL_N, DEVICE)
    return sum(1 for s in smp if Chem.MolFromSmiles(s) is not None) / len(smp)

def train_lowdata(use_graph, ratio, tr_seed):
    tag = f"ld_{'graphbias' if use_graph else 'baseline'}_r{int(ratio*100)}_s{tr_seed}"
    ckpt = os.path.join(CKPT_DIR, tag + '.pt')
    if os.path.exists(ckpt):
        m = MolGPT(use_graph_bias=use_graph).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        v = eval_validity(m)
        print(f"  [{tag}] checkpoint 已存在 → validity {v:.1%}")
        return v, 0.0

    sub_rng = np.random.RandomState(tr_seed)
    n_sub = max(2000, int(len(TRN_IDX) * ratio))
    sub_idx = sub_rng.permutation(TRN_IDX)[:n_sub]
    set_seed(tr_seed)
    model = MolGPT(use_graph_bias=use_graph).to(DEVICE)
    ds_tr = MolDataset(sub_idx, need_graph=use_graph)
    ds_va = MolDataset(VAL_IDX, need_graph=use_graph)
    mk = lambda ds, sh: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=sh,
                                   collate_fn=collate, num_workers=2,
                                   pin_memory=(DEVICE.type == 'cuda'), drop_last=sh)
    loader_tr, loader_va = mk(ds_tr, True), mk(ds_va, False)

    steps_ep = math.ceil(n_sub / BATCH_SIZE)
    total_steps = steps_ep * EPOCHS
    warm_eff = min(WARMUP, max(50, total_steps // 10))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    def lr_lambda(step):
        if step < warm_eff: return step / max(1, warm_eff)
        p = (step - warm_eff) / max(1, total_steps - warm_eff)
        return LR_MIN/LR + (1 - LR_MIN/LR) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

    t0 = time.time(); best_val, bad = float('inf'), 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        for batch in loader_tr:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                _, loss = model(batch['x'], batch.get('bias'), batch['y'])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update(); sched.step()
        model.eval(); vtot, vnb = 0.0, 0
        with torch.no_grad():
            for batch in loader_va:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                _, vl = model(batch['x'], batch.get('bias'), batch['y'])
                vtot += vl.item(); vnb += 1
        va_loss = vtot / max(vnb, 1)
        if va_loss < best_val - 1e-4:
            best_val, bad = va_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    train_sec = time.time() - t0
    v = eval_validity(model)
    print(f"  [{tag}] 子集 {n_sub:,} 条 | 训练 {train_sec/60:.1f}min → validity {v:.1%}")
    del model; torch.cuda.empty_cache()
    return v, round(train_sec, 1)

# ---------------- 主循环: 增量写 CSV ----------------
csv_path = os.path.join(GEN_DIR, 'lowdata_results.csv')
df = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame(
    columns=['variant', 'ratio', 'tr_seed', 'validity', 'reused_main', 'train_sec'])
done = set(zip(df['variant'], df['ratio'].astype(float), df['tr_seed']))

for ratio in NEW_RATIOS:
    for variant, use_graph in LOWDATA_VARIANTS:
        for tsd in LOWDATA_TRSEEDS:
            if (variant, float(ratio), tsd) in done:
                print(f"  跳过已完成: {variant} r{ratio} s{tsd}"); continue
            v, tsec = train_lowdata(use_graph, ratio, tsd)
            row = dict(variant=variant, ratio=ratio, tr_seed=tsd,
                       validity=round(v, 4), reused_main=False, train_sec=tsec)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')   # 每点立即落盘
            print(f"  ✓ 已追加并保存: {row}")

print("\n===== 全部实验点 =====")
print(df.sort_values(['variant', 'ratio', 'tr_seed']).to_string(index=False))
print(f"\n[完成] {csv_path} 请下载该文件回本地出新 A 图")
