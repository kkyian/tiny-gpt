import argparse
from pathlib import Path
import torch
from model import GPTConfig, TinyGPT
from tokenizer import CharTokenizer
from device import get_device

def main():
    p=argparse.ArgumentParser(); p.add_argument('prompt'); p.add_argument('--checkpoint',default='checkpoints/model.pt'); p.add_argument('--tokens',type=int,default=300); p.add_argument('--temperature',type=float,default=0.8); p.add_argument('--top-k',type=int,default=20); a=p.parse_args(); device=get_device(); ck=torch.load(a.checkpoint,map_location=device); tok=CharTokenizer(ck['vocab']); model=TinyGPT(GPTConfig(**ck['config'])).to(device); model.load_state_dict(ck['model']); model.eval(); ids=torch.tensor([tok.encode(a.prompt)],device=device)
    with torch.no_grad():
        for _ in range(a.tokens):
            logits,_=model(ids[:,-model.config.block_size:]); logits=logits[:,-1,:]/max(a.temperature,1e-6); values,indices=torch.topk(logits,min(a.top_k,logits.size(-1))); logits[logits<values[:,-1,None]]=-float('inf'); ids=torch.cat((ids,torch.multinomial(torch.softmax(logits,dim=-1),1)),dim=1)
    print(tok.decode(ids[0].tolist()))
if __name__=='__main__': main()
