"""
=============================================================
TinyGPT - Mode 3: Word-Level Tokenization
=============================================================
Topik Corpus: E-Commerce Dynamic Pricing & NLP
Nama       : Nazal Syamaidzar Mahendra
NIM        : 23.11.5547
Mata Kuliah: Proyek Data Mining
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
import time
from transformer_blocks import Block

# ─────────────────────────────────────────────
# Konfigurasi
# ─────────────────────────────────────────────
TOKENIZER_MODE = "word"      # ← Mode tokenisasi kata
CORPUS_FILE    = "corpus.txt"

block_size    = 16           # lebih pendek karena word-level
embedding_dim = 64
n_heads       = 4
n_layers      = 3
lr            = 1e-3
epochs        = 2000
batch_size    = 32

print("=" * 60)
print(f"  TinyGPT | Mode Tokenisasi: {TOKENIZER_MODE.upper()}")
print("=" * 60)

# ─────────────────────────────────────────────
# 1. Baca Corpus
# ─────────────────────────────────────────────
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    text = f.read()

word_count = len(text.split())
char_count = len(text)
print(f"\n[Corpus] Jumlah kata      : {word_count:,}")
print(f"[Corpus] Jumlah karakter  : {char_count:,}")

# ─────────────────────────────────────────────
# 2. Word-Level Tokenizer
# ─────────────────────────────────────────────
# Tokenisasi sederhana: pisahkan per spasi & tanda baca
def word_tokenize(text):
    # Split berdasarkan whitespace dan tanda baca, pertahankan kata
    tokens = re.findall(r"[A-Za-z0-9]+|[^\w\s]", text.lower())
    return tokens

tokens_list = word_tokenize(text)
vocab_set   = sorted(set(tokens_list))
vocab_size  = len(vocab_set)

# Tambah <UNK> untuk kata tidak dikenal
vocab_set   = ['<UNK>'] + vocab_set
stoi = {w: i for i, w in enumerate(vocab_set)}
itos = {i: w for i, w in enumerate(vocab_set)}

def encode(tokens):
    return [stoi.get(t, 0) for t in tokens]  # 0 = <UNK>

def decode(ids):
    return ' '.join([itos.get(i, '<UNK>') for i in ids])

# Encode seluruh corpus
ids   = encode(tokens_list)
data  = torch.tensor(ids, dtype=torch.long)
vocab_size = len(stoi)

print(f"\n[Tokenizer] Mode          : Word-Level")
print(f"[Tokenizer] Vocabulary    : {vocab_size} kata unik")
print(f"[Tokenizer] Total token   : {len(ids):,}")
print(f"[Tokenizer] Rasio         : {word_count / len(ids):.3f} kata/token")

# Contoh tokenisasi
sample       = "dynamic pricing e-commerce"
sample_toks  = word_tokenize(sample)
sample_ids   = encode(sample_toks)
print(f"\n[Contoh Word] '{sample}'")
print(f"  → Tokens: {sample_toks}")
print(f"  → IDs   : {sample_ids}")

# ─────────────────────────────────────────────
# 3. Model TinyGPT
# ─────────────────────────────────────────────
class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[
            Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# ─────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i+block_size]   for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

model     = TinyGPT(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
n_params  = sum(p.numel() for p in model.parameters())
print(f"\n[Model] Parameter total: {n_params:,}")
print(f"\n[Training] Mulai pelatihan {epochs} epoch...")

loss_history = []
t0 = time.time()

for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if step % 400 == 0 or step == epochs - 1:
        elapsed = time.time() - t0
        perplexity = math.exp(min(loss.item(), 20))
        print(f"  Step {step:4d}/{epochs} | Loss: {loss.item():.4f} | "
              f"Perplexity: {perplexity:.2f} | Waktu: {elapsed:.1f}s")

total_time = time.time() - t0
final_loss = loss_history[-1]
final_ppl  = math.exp(min(final_loss, 20))

print(f"\n[Hasil] Loss akhir   : {final_loss:.4f}")
print(f"[Hasil] Perplexity  : {final_ppl:.2f}")
print(f"[Hasil] Waktu total : {total_time:.1f} detik")

# ─────────────────────────────────────────────
# 5. Generasi Teks
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  HASIL GENERASI TEKS - Mode Word")
print("=" * 60)

prompts = ["dynamic pricing", "deep learning", "e-commerce"]
for prompt in prompts:
    enc_prompt = encode(word_tokenize(prompt))
    if not enc_prompt:
        continue
    context = torch.tensor([enc_prompt], dtype=torch.long)
    out = model.generate(context, max_new_tokens=25, temperature=0.8)
    generated = decode(out[0].tolist())
    print(f"\nPrompt: '{prompt}'")
    print(f"Output: {generated}")

# ─────────────────────────────────────────────
# 6. Ringkasan Analisis
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ANALISIS PERFORMA - Mode Word")
print("=" * 60)
print(f"  Mode tokenisasi  : Word-Level")
print(f"  Vocabulary size  : {vocab_size} kata unik")
print(f"  Total token      : {len(ids):,}")
print(f"  Loss awal        : {loss_history[0]:.4f}")
print(f"  Loss akhir       : {final_loss:.4f}")
print(f"  Perplexity akhir : {final_ppl:.2f}")
print(f"  Parameter model  : {n_params:,}")
print(f"  Waktu pelatihan  : {total_time:.1f}s")
print("=" * 60)
