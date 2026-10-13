import torch
from model import GPTConfig, TinyGPT
cfg=GPTConfig(vocab_size=11,block_size=16,n_layer=2,n_head=2,n_embd=32); m=TinyGPT(cfg); x=torch.randint(0,11,(3,8)); y=torch.randint(0,11,(3,8)); logits,loss=m(x,y); assert logits.shape==(3,8,11) and torch.isfinite(loss); assert torch.all(m.blocks[0].attn.mask[0,0].triu(1)==0); loss.backward(); print('PASS: shapes, finite loss, causal mask, backward')
