#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================
PPT 第4页 C 图数据: 多任务模型「高 QED 导向」生成 5000 个分子 (全量, 不抽样)
=================================================================================
流程 (与 1_molgpt_train.py Cell 8.5 完全同一套方法):
  1. 加载多任务模型 checkpoint (molgpt_multitask.pt, 含性质预测头)
  2. 自由采样一个大候选池 (12000 条原始 SMILES, 固定种子保证可复现)
  3. RDKit 解析 + canonical 去重 (约 1.1 万个有效唯一分子)
  4. 性质头预测每个分子的 QED, 按预测值从高到低取 Top 5000
     —— 不缩减到几百, 橙色曲线就是完整的 5000 个导向分子
  5. RDKit 计算真实 QED, 导出 directed_qed_5000.csv

运行 (服务器, GPU 约 5~10 分钟):
  cd ~/ASPA_planB
  nohup ~/aspa_env/bin/python scripts/6_directed_5000.py > directed_5000.log 2>&1 &
  tail -n 30 directed_5000.log

产物:
  output_molgpt/directed_qed_5000.csv  (SMILES, pred_QED, true_QED; 5000 行)
  下载回本地 d:\\ASPA-肌肽\\PPT图表_模型调优_v2\\ 后运行 plot_bce.py 重绘 C 图
=================================================================================
"""
import os, math, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
RDLogger.DisableLog('rdApp.*')

# ---------------- 路径/超参 (与 1_molgpt_train.py 一致) ----------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_SCRIPT_DIR) if os.path.isdir(
    os.path.join(os.path.dirname(_SCRIPT_DIR), 'data')) else _SCRIPT_DIR
GEN_DIR  = os.path.join(BASE_DIR, 'output_molgpt')
CKPT_DIR = os.path.join(BASE_DIR, 'ckpt')
DATA_DIR = os.path.join(BASE_DIR, 'molgpt_data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 20260925
BLOCK_SIZE = 104
MAX_SMILES = 100
N_LAYER, N_HEAD, N_EMBD, DROPOUT = 4, 8, 256, 0.1
N_RAW = 12000          # 原始采样数 (保证有效去重后 >= 5000)
N_KEEP = 5000          # 按预测 QED 保留的头部分子数
SAMP_TEMP, SAMP_TOPK = 0.8, 40
LAMBDA_PROP = 0.1

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------------- 词表 ----------------
canon_df = pd.read_csv(os.path.join(DATA_DIR, 'smiles_clean.csv'))
canon_list = canon_df['SMILES'].astype(str).tolist()
PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
char_set = sorted({ch for smi in canon_list for ch in smi})
stoi = {PAD: 0, BOS: 1, EOS: 2}
stoi.update({ch: i + 3 for i, ch in enumerate(char_set)})
itos = {i: ch for ch, i in stoi.items()}
VOCAB_SIZE = len(stoi)
print(f"[环境] BASE={BASE_DIR} | device={DEVICE} | vocab={VOCAB_SIZE}")

# ---------------- 多任务模型 (含性质头, 与 Cell 4 一致) ----------------
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
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD); self.attn = GraphBiasAttention(N_HEAD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT))
    def forward(self, x, graph_bias=None):
        x = x + self.attn(self.ln1(x), None)
        x = x + self.mlp(self.ln2(x))
        return x

class MolGPTMulti(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE - 1, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok_emb.weight
        self.prop_head = nn.Sequential(
            nn.Linear(N_EMBD, 128), nn.GELU(), nn.Dropout(DROPOUT))
        self.prop_out = nn.ModuleList([nn.Linear(128, 1) for _ in range(3)])
        self.register_buffer('prop_mu', torch.zeros(3))
        self.register_buffer('prop_sd', torch.ones(3))

    def forward(self, x):
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos)[None])
        for blk in self.blocks:
            h = blk(h)
        return self.ln_f(h)

    @torch.no_grad()
    def generate(self, n_samples, temp=SAMP_TEMP, topk=SAMP_TOPK,
                 max_new=BLOCK_SIZE - 2, batch=256):
        self.eval(); out = []; n_done = 0
        while n_done < n_samples:
            b = min(batch, n_samples - n_done)
            x = torch.full((b, 1), stoi[BOS], dtype=torch.long, device=DEVICE)
            fin = torch.zeros(b, dtype=torch.bool, device=DEVICE)
            seqs = [[] for _ in range(b)]
            for _ in range(max_new):
                h = self.forward(x)
                logits = self.head(h[:, -1]) / temp
                if topk:
                    vv, _ = torch.topk(logits, min(topk, logits.size(-1)))
                    logits[logits < vv[:, [-1]]] = float('-inf')
                nxt = torch.multinomial(logits.softmax(-1), 1).reshape(b, -1)[:, :1]
                nxt = torch.where(fin[:, None], torch.full_like(nxt, stoi[PAD]), nxt)
                x = torch.cat([x, nxt], dim=1)[:, -(BLOCK_SIZE-1):]
                for j in range(b):
                    if not fin[j]:
                        t = nxt[j].item()
                        if t == stoi[EOS]: fin[j] = True
                        elif t != stoi[PAD]: seqs[j].append(itos[t])
                if fin.all(): break
            out.extend(''.join(s) for s in seqs); n_done += b
            print(f"    采样 {n_done}/{n_samples}", end='\r')
        print()
        return out

    @torch.no_grad()
    def pred_props(self, smiles_list, batch=256):
        """性质头预测 (QED,logP,SA) 原始尺度, 池化方式与训练一致"""
        out = []
        for i in range(0, len(smiles_list), batch):
            chunk = smiles_list[i:i + batch]
            seqs = [[stoi[BOS]] + [stoi[c] for c in s[:MAX_SMILES]] + [stoi[EOS]]
                    for s in chunk]
            L = max(len(s) for s in seqs)
            x = torch.full((len(seqs), L), stoi[PAD], dtype=torch.long, device=DEVICE)
            for j, ids in enumerate(seqs):
                x[j, :len(ids)] = torch.tensor(ids, device=DEVICE)
            pos = torch.arange(L, device=DEVICE)
            h = self.drop(self.tok_emb(x) + self.pos_emb(pos)[None])
            for blk in self.blocks:
                h = blk(h)
            h = self.ln_f(h)
            mask = (x != stoi[PAD]).float()[:, :, None]
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            pf = self.prop_head(pooled)
            pred = torch.cat([o(pf) for o in self.prop_out], dim=1)
            pred = pred * self.prop_sd[None, :] + self.prop_mu[None, :]
            out.append(pred.cpu().numpy())
        return np.concatenate(out)

# ---------------- 加载 checkpoint ----------------
ckpt = os.path.join(CKPT_DIR, 'molgpt_multitask.pt')
assert os.path.exists(ckpt), f"找不到多任务模型 checkpoint: {ckpt}"
set_seed(SEED)
model = MolGPTMulti().to(DEVICE)
model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
model.eval()
print(f"[模型] 已加载 {ckpt}")

# ---------------- 采样 → 去重 ----------------
print(f"[采样] 自由生成 {N_RAW} 条候选 ...")
raw = model.generate(N_RAW)
pool = sorted({Chem.MolToSmiles(m) for s in raw
               if (m := Chem.MolFromSmiles(s)) is not None})
print(f"[采样] 有效去重后 {len(pool):,} 个 (需要 ≥ {N_KEEP})")
assert len(pool) >= N_KEEP, f"有效分子不足 {N_KEEP}, 请把 N_RAW 调大后重跑"

# ---------------- 性质头预测 → Top 5000 → 真实 QED ----------------
print("[打分] 性质头预测 QED ...")
pred = model.pred_props(pool)[:, 0]
order = np.argsort(-pred)[:N_KEEP]
kept = [pool[i] for i in order]
true_qed = [QED.qed(Chem.MolFromSmiles(s)) for s in kept]
df = pd.DataFrame({'SMILES': kept,
                   'pred_QED': np.round(pred[order], 6),
                   'true_QED': np.round(true_qed, 6)})
out_csv = os.path.join(GEN_DIR, 'directed_qed_5000.csv')
df.to_csv(out_csv, index=False, encoding='utf-8-sig')

print(f"\n[完成] {out_csv} ({len(df)} 行)")
print(f"  导向组真实 QED: 均值 {df['true_QED'].mean():.4f} | "
      f"中位 {df['true_QED'].median():.4f} | "
      f"≥0.8 占比 {(df['true_QED']>=0.8).mean():.1%}")
print("  下一步: 下载该 CSV 到本地 PPT图表_模型调优_v2\\, 运行 plot_bce.py 重绘 C 图")
