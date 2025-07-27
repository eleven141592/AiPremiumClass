import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

def get_batch(split):
    # 选择训练或验证数据集
    data = train_data if split == 'train' else val_data

    # 动态从数据集中选择位置索引
    ix = torch.randint(len(data) - block_size, (batch_size,)) # [0,103846]随机生成位置索引，向后截取block_size字符训练
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device),y.to(device)



class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)



class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, n_embd, head_size, dropout):
        super().__init__()
        # 确保 n_embd 可以被 num_heads 整除
        assert n_embd % num_heads == 0
        
        # 使用一个大的线性层一次性计算所有头的Q, K, V
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # 输出的投影层
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.num_heads = num_heads
        self.n_embd = n_embd
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B: batch_size, T: sequence_length, C: n_embd
        B, T, C = x.size() 

        # 1. 一次性计算 Q, K, V
        # x -> (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) # (B, T, C) -> 3 * (B, T, C)
        
        # 2. 重塑并转置，以满足 F.scaled_dot_product_attention 的 4D 输入要求
        # (B, T, C) -> (B, T, num_heads, head_size) -> (B, num_heads, T, head_size)
        head_size = C // self.num_heads
        k = k.view(B, T, self.num_heads, head_size).transpose(1, 2)
        q = q.view(B, T, self.num_heads, head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, head_size).transpose(1, 2)

        # 3. 只调用一次 scaled_dot_product_attention
        # FlashAttention2 只支持 bfloat16 或 float16 类型的张量
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        # PyTorch 2.0+ 的高效实现，可以自动选择 Flash Attention
        # 输出 attn_output 的形状也是 (B, num_heads, T, head_size)
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        
        # 4. 将输出重塑回原始的 (B, T, C) 形状
        # (B, num_heads, T, head_size) -> (B, T, num_heads, head_size) -> (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        
        # 5. 通过最后的投影层
        out = self.c_proj(attn_output.to(torch.float32)) # 将数据类型转换回 float32
        
        return out

# ===================== 修改后的 Block 类 =====================
# 你的 Block 类初始化需要稍微调整一下，因为 head_size 现在在 MultiHeadAttention 内部计算
class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head # 这行其实可以不要了，但保留也无妨
        # 注意这里的 MultiHeadAttention 初始化参数变了
        self.sa = MultiHeadAttention(n_head, n_embd, head_size, dropout)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 注意力计算的输出需要变回 float32 和残差连接的张量类型匹配
        x_sa = self.sa(self.ln1(x)).to(x.dtype)
        x = x + x_sa  # 残差连接
        
        x_ffwd = self.ffwd(self.ln2(x)).to(x.dtype)
        x = x + x_ffwd # 残差连接
        return x

class BingramLanguageModel(nn.Module):
    
    def __init__(self, block_size, vocab_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        # 每个token直接输出的logits值作为下一个token的映射
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx和target都是维度为 (B,T) 的整型tensor
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device), ) # (T,C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx指当前语料集(B,T)中的索引
        for _ in range(max_new_tokens):
            # 限定索引列的取值范围
            idx_cond = idx[:, -block_size:]
            # 推理
            logits, loss = self(idx_cond)
            # 只提取最后一个时间步的结果
            logits = logits[:, -1, :]  # (B,C)
            # 通过softmax转换为概率值
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # 随机采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # 把采样的索引追加在当前解码序列末尾
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

if __name__ == '__main__':

    # 模型训练数据集
    block_size = 8
    batch_size = 32
    max_iter = 5000
    learn_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd = 32
    eval_interval = 500
    eval_iters = 200
    head_size = 8
    num_layers = 4
    dropout = 0.1

    
    with open(r'/mnt/data_1/zfy/self/八斗精品班/第十七周_模型部署优化/资料/homework/homework1/剑来(1-500章).txt', encoding='utf-8') as f:
        text = f.read()

    # 字典、编码器(函数)、解码器(函数)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}  #str_to_index
    itos = {i:ch for i,ch in enumerate(chars)}  #index_to_str

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 文本转换token index
    data = torch.tensor(encode(text), dtype=torch.long)

    # 拆分数据集
    n = int(len(data) * .9)
    train_data = data[:n]
    val_data = data[n:]

    # 模型训练
    model = BingramLanguageModel(block_size, vocab_size, n_embd, head_size, num_layers, dropout)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    for iter in range(max_iter):

        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # 批次样本
        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 模型生成
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(model.generate(idx, max_new_tokens=500)[0].tolist())) 