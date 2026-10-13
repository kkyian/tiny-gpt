import argparse, json, math, time
from pathlib import Path
import torch
from model import GPTConfig, TinyGPT
from tokenizer import CharTokenizer
from device import get_device

ROOT = Path(__file__).parent

def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in starts]); y = torch.stack([data[i+1:i+block_size+1] for i in starts])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate(model, data, batch_size, block_size, device, steps=10):
    model.eval(); vals = [model(*get_batch(data, batch_size, block_size, device))[1].item() for _ in range(steps)]; model.train(); return sum(vals)/len(vals)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=2000); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--block-size',type=int,default=128); p.add_argument('--eval-interval',type=int,default=200); p.add_argument('--checkpoint',default='checkpoints/model.pt'); p.add_argument('--overfit-check',action='store_true'); args=p.parse_args()
    torch.manual_seed(1337); device=get_device(); tok=CharTokenizer.load(ROOT/'data/tokenizer.json')
    train=torch.tensor(tok.encode((ROOT/'data/train.txt').read_text()),dtype=torch.long); val=torch.tensor(tok.encode((ROOT/'data/val.txt').read_text()),dtype=torch.long)
    cfg=GPTConfig(tok.vocab_size,args.block_size,n_layer=4,n_head=4,n_embd=128,dropout=0.1); model=TinyGPT(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=0.1)
    if args.overfit_check: args.steps=300; args.batch_size=8
    for step in range(args.steps+1):
        if step % args.eval_interval == 0: print(f'step {step:5d} train {estimate(model,train,args.batch_size,args.block_size,device):.4f} val {estimate(model,val,args.batch_size,args.block_size,device):.4f}')
        if step == args.steps: break
        x,y=get_batch(train,args.batch_size,args.block_size,device); _,loss=model(x,y); assert torch.isfinite(loss); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    Path(args.checkpoint).parent.mkdir(exist_ok=True); torch.save({'model':model.state_dict(),'config':cfg.__dict__,'vocab':tok.characters},args.checkpoint); print(f'Saved {args.checkpoint} on {device}')

if __name__=='__main__': main()
