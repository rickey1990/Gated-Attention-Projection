"""GAP20 family and standard Transformer, configurable without changing the block equation.

Original 100M run: standard(width=64), GAP(width=72,qk_head=22), 4 blocks,
4 heads, 2048 vocab, context 96. Parameter count: 337280 vs 270128.
The compact GAP(width=64,qk_head=16) uses 221568 parameters but was NOT
trained in the published 100M experiment.
"""
# Required Notice: Copyright (c) 2026 GAP20 original author, rickeyernest1990@gmail.com
import torch
from torch import nn
from torch.nn import functional as F

class MHAttention(nn.Module):
    def __init__(self,width,heads=4,qk_head=None):
        super().__init__()
        if width%heads: raise ValueError('width must be divisible by heads')
        self.width=width
        self.heads=heads
        self.v_dim=width//heads
        self.k_dim=self.v_dim if qk_head is None else qk_head
        if self.k_dim<1:raise ValueError('qk_head must be positive')
        self.qkv=nn.Linear(width,heads*(2*self.k_dim+self.v_dim))
    def forward(self,x):
        B,T,C=x.shape
        a=self.qkv(x).view(B,T,self.heads,2*self.k_dim+self.v_dim).permute(0,2,1,3)
        q,k,v=torch.split(a,[self.k_dim,self.k_dim,self.v_dim],dim=-1)
        out=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
        return out.transpose(1,2).reshape(B,T,C)

class StandardBlock(nn.Module):
    def __init__(self,width=64,heads=4,mlp_ratio=4):
        super().__init__()
        self.ln1=nn.LayerNorm(width)
        self.attn=MHAttention(width,heads)
        self.attn_proj=nn.Linear(width,width)
        self.ln2=nn.LayerNorm(width)
        hidden=int(mlp_ratio*width)
        self.ff=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,width))
    def forward(self,x):
        z=x+self.attn_proj(self.attn(self.ln1(x)))
        return z+self.ff(self.ln2(z))

class GAPBlock(nn.Module):
    """z=LN2(x+MHA(LN1(x))); y=x+W_O(z * SiLU(W_G(z)))."""
    def __init__(self,width=72,heads=4,qk_head=22):
        super().__init__()
        self.ln1=nn.LayerNorm(width)
        self.attn=MHAttention(width,heads,qk_head=qk_head)
        self.ln2=nn.LayerNorm(width)
        self.gate=nn.Linear(width,width)
        self.output=nn.Linear(width,width)
    def forward(self,x):
        z=self.ln2(x+self.attn(self.ln1(x)))
        return x+self.output(z*F.silu(self.gate(z)))

class LanguageModel(nn.Module):
    def __init__(self,kind,context=96,vocab=2048,depth=4,heads=4,
                 standard_width=64,gap_width=72,gap_qk_head=22,mlp_ratio=4):
        super().__init__()
        if kind not in ('standard','gap20'):raise ValueError(kind)
        self.kind=kind;self.vocab=vocab;self.context=context
        self.width=standard_width if kind=='standard' else gap_width
        width=self.width
        if width%heads:raise ValueError('width must be divisible by heads')
        self.tok=nn.Embedding(vocab,width)
        self.pos=nn.Embedding(context,width)
        self.blocks=nn.ModuleList([
            StandardBlock(width,heads,mlp_ratio) if kind=='standard' else
            GAPBlock(width,heads,gap_qk_head)
            for _ in range(depth)])
        self.final_norm=nn.LayerNorm(width)
        self.lm_head=nn.Linear(width,vocab,bias=False)
        self.lm_head.weight=self.tok.weight
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0.,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None:nn.init.zeros_(m.bias)
    def forward(self,tokens):
        B,T=tokens.shape
        if T>self.context:raise ValueError(f'length {T} exceeds context {self.context}')
        x=self.tok(tokens)+self.pos(torch.arange(T,device=tokens.device))
        for block in self.blocks:x=block(x)
        return self.lm_head(self.final_norm(x))

def count(model):return sum(p.numel() for p in model.parameters())

PRESETS={
    'paper100m': {'gap_width':72,'gap_qk_head':22},
    'compact34': {'gap_width':64,'gap_qk_head':16},
}

def make_model(kind='gap20',variant='paper100m',**kwargs):
    if variant not in PRESETS and variant!='custom':raise ValueError(variant)
    if variant!='custom':
        defaults=dict(PRESETS[variant]);defaults.update(kwargs);kwargs=defaults
    return LanguageModel(kind,**kwargs)

if __name__=='__main__':
    baseline=count(make_model('standard'))
    for variant in PRESETS:
        m=make_model('gap20',variant=variant)
        print(variant,'parameters',count(m),'smaller than standard %',round(100*(1-count(m)/baseline),4))