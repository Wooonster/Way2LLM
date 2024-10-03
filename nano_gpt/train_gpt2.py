from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2LMHeadModel
import tiktoken

# ------------------------------------------------------------------------------------------------------------------------
@dataclass
class GPTConfigs:
    block_size: int = 1024    # max sequence length
    vocab_size: int = 50257   # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> special tokens
    n_layer: int = 12         # number of layers
    n_head: int = 12          # number of heads
    n_embed: int = 768        # embedding dimensions

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu   = nn.GELU(approximate='tanh')  # nonlinearity, activation func
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # K, Q, V projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # output projection
        " projecting the combined information from all the heads back into the original embedding space. "
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # for scaling the weights of residual layers at initialization by a factor of 1/sqrt(N)
        # regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        # not actually a 'bias', more like a mask, just following gpt2's naming
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x):
        B, T, C = x.size()  # B: batch_size, T: sequence length, C: embedding dimensionality (n_embed)
        '''
        calculate the query, keys, values for all heads in batch and move head forward to be the batch dim
        nh: number of heads, hs: head size, C = nh * hs: number of channels
        e.g. for gpt2(124M), n_head=12, head_size=64, C=n_head*head_size=768 channels in Transformer
        '''
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # attention (materializes the large (T, T) matrix for all the queries and keys)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # autoaggressive mask to make sure that the tokens only attend to tokens before
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        ' y is still of shape (B, T, config.n_embed) after re-assembling all heads '
        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    '''
    the layer norms are after the application of attention or feed forward
    and they are inside the residual stream

    Layer normalization was moved to the input of each sub-block, 
    similar to a pre-activation residual network 
    '''
    
    def forward(self, x):
        " the attentions function is where the tokens communicate, it's a aggragation function "
        x = x + self.attn(self.ln_1(x))  
        
        ' x += ... is change in-place, which disrupts the computation of gradients'

        " in the mlp, which happends every single token individually, no info being collected or exchanging "
        x = x + self.mlp(self.ln_2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config  # set up configs

        self.transformer = nn.ModuleDict(dict(
            # word token embedding 
            wte = nn.Embedding(config.vocab_size, config.n_embed),  # weights of token embed
            # word positional embedding 
            wpe = nn.Embedding(config.block_size, config.n_embed),  # weights of positional embed
            # hidden layers 
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # hidden layer
            # an additional layer normalization was added after the final self-attention block 
            ln_f = nn.LayerNorm(config.n_embed),  # an additional final layer norm according to the paper
        ))

        # the final classifier, the language model head, projects the n_embed_dims to vocab_size
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    " how gpt2 initialise the model "
    " iterates the NanoGPT module"
    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            " scaling the weights of residual layers at initialization by a factor of 1/sqrt(N) "
            # 2 -> every transformer layer has 2 blocks add to residual: attention and mlp
            std *= (2 * self.config.n_layer) ** -0.5

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: token index, is of shape (B, T), B: batch_size, T: sequence_length
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence length of length {T} which is larger than the max_seq_len, block size of {B}"
        # forward the token and positional embedding
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embed)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # flatten to 2D -> (B*T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    " load params from huggingface transformers "
    @classmethod
    def from_pretrained(cls, model_type):
        " Load pretrained GPT-2 model weights from huggingface "
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f"Loading weights from pretrained gpt of type: {model_type}")

        # n_layer, n_head, n_embed are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # create a form-scratch initialized nanoGPT model
        config = GPTConfigs(**config_args)
        model = NanoGPT(config=config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the params are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore the 2 buffers
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# ------------------------------------------------------------------------------------------------------------------------

" A simple data loader "
class DataLoaderLite:
    def __init__(self, B, T) -> None:
        self.B = B
        self.T = T

        with open('../input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B * T)} batches')

        # state
        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : B * T + self.current_position + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        # advance the position in the tensor
        self.current_position += B * T
        # resets, if loading the next batch would be out of bounds
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
    
# ------------------------------------------------------------------------------------------------------------------------

if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f'Device sets to: {device}')

# set seeds
torch.manual_seed(5525)
if torch.cuda.is_available():
    torch.cuda.manual_seed(5525)

# prefix tokens
# tokens = enc.encode('Mayday is a rock band from China,')
# tokens = torch.tensor(tokens, dtype=torch.long)  # (token_num, )
# tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1)  # (5, token_num)
# x = tokens.to(device)

B, T = 4, 32
train_loader = DataLoaderLite(B, T)

# model = NanoGPT.from_pretrained('gpt2')  # init with huggingface pretrained weights
model = NanoGPT(GPTConfigs())  # random model initialization
print('Model init succeed.')

# model.eval()
model.to(device)

' optimisation '
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    ' get the loss '
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f'step: {i}, loss: {loss.item()}')

print(f'loss = {loss}')
import sys; sys.exit(0)

" Generate "
# num_return_sequence = 5
# max_length = 40

while x.size(1) < max_length:
    while torch.no_grad():
        logits = model(x)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # get the probabilities
        probs = F.softmax(logits, dim=-1)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        # select a token from the top-k probabilities
        # note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1)
        # gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # append to the sequence
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequence):
    tokens = x[i, :max_length].tolist()
    decodeed = enc.decode(tokens)
    print(">>>", decodeed)