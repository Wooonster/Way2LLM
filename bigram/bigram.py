import torch
import torch.nn as nn
from torch.nn import functional as F

# ----- hyperparameters -----
BATCH_SIZE = 64   # number of independent sequences process in parallel
BLOCK_SIZE = 256  # maximum context length for prediction
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
LEARNING_RATE = 3e-4
N_EMBED = 384
N_HEAD = 6  # every head is 384/6=64 dimensional
N_LAYERS = 6
DROPOUT = 0.2
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# ---------------------------

torch.manual_seed(5525)

# open file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create encode, decode
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# data split
data = torch.tensor(encode(text), dtype=torch.long)
N = int(0.9 * len(data))
train_data = data[:N]
val_data = data[N:]

# load data
def get_batch(split):
    data = train_data if split == "train" else val_data

    idx = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, ))

    x = torch.stack([data[i:i+BLOCK_SIZE] for i in idx])
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in idx])

    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

# estimate loss per batch
@torch.no_grad()  # everything in the following func, no backprop function, save memory
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# attention mechanism
# one head of self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        """
        Add a buffer to the module. To be a part of this module's state_dict.
        This is typically used to register a buffer that should not to be considered a model parameter
        """
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)

        weight = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T]==0, float('-inf'))
        weight = self.dropout(F.softmax(weight, dim=-1))  # (B, T, T)

        v = self.value(x)  # (B, T, C)
        out = weight @ v   # (B, T, C)
        return out

# multi-head attention in parallel
class MultiHeadattention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(N_EMBED, N_EMBED)  # residual connection projection
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.projection(out))
    
# FeedForward 
class FFN(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  #   # residual connection projection    
        )

    def forward(self, x):
        return self.net(x)
    
# Transformer block
class Block(nn.Module):
    '''
    Transformer block: communication followed by computation
    '''
    def __init__(self, n_embed, n_head):
        super().__init__()
        # n_embed: embedding dimension, n_head: the number of heads we want
        head_size = n_embed // n_head
        self.sa = MultiHeadattention(n_head, head_size)  # communication
        self.ffn = FFN(n_embed)  # computation
        # layer normalisation
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # residual connection, folk off then come back
        x = x + self.ffn(self.ln2(x))
        return x

# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_lookup_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)  # add a positional embedding
        
        # add blocks of attention
        # self.blocks = nn.Sequential(
        #     Block(N_EMBED, n_head=4),
        #     Block(N_EMBED, n_head=4),
        #     Block(N_EMBED, n_head=4),
        #     nn.LayerNorm(N_EMBED),
        # )
        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYERS)])

        # self.lm_linear = nn.Linear(N_EMBED, vocab_size)  # add a linear layer
        
        # # single head
        # self.sa_head = Head(N_EMBED)  # self-attention head
        
        # # multi-head
        # self.sa_heads = MultiHeadattention(4, N_EMBED//4)
        # # add ffn layer
        # self.ffn = FFN(N_EMBED)

        self.ln_f = nn.LayerNorm(N_EMBED)  # final layer normlisation
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # (B, T)

        token_emb = self.embedding_lookup_table(idx)  # (B, T, n_embed)
        position_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))  # (T, C) 0~(T-1)
        x = token_emb + position_emb  # (B, T, C)
        # x = self.sa_head(x)  # apply one head of self-attention
        # x = self.sa_heads(x)   # apply multihead attention
        # x = self.ffn(x)

        x = self.blocks(x)  # add blocks of heads
        x = self.ln_f(x)  # add normlisation

        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            # idx cant be larger than BLOCK_SIZE
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            prods = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prods, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
model.to(DEVICE)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# train loop
for iter in range(MAX_ITERS):
    if iter % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # sample a batch of data
    x, y = get_batch('train')

    # evaluate the loss
    logits, loss = model(x, y)
    # set grad=0
    optimizer.zero_grad(set_to_none=True)
    # bp loss
    loss.backward()
    # step optim
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(model.generate(context, 1000)[0].tolist()))