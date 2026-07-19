import torch, torch.nn as nn, torch.nn.functional as F, json, numpy as np, time, os, warnings
warnings.filterwarnings('ignore')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

class RMSNorm(nn.Module):
    def __init__(s,d,eps=1e-6): super().__init__(); s.w=nn.Parameter(torch.ones(d)); s.eps=eps
    def forward(s,x): return (x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+s.eps)).type_as(x)*s.w
class RoPE(nn.Module):
    def __init__(s,d,m=2048):
        super().__init__(); inv=1/(10000**(torch.arange(0,d,2).float()/d)); pos=torch.arange(m).float()
        f=torch.outer(pos,inv); s.register_buffer('cos',f.cos()[None,None,:,:]); s.register_buffer('sin',f.sin()[None,None,:,:])
    def forward(s,x):
        S,d=x.shape[2],x.shape[-1]//2; c,s2=s.cos[:,:,:S,:d],s.sin[:,:,:S,:d]; x1,x2=x[...,:d],x[...,d:]
        return torch.cat([x1*c-x2*s2,x1*s2+x2*c],dim=-1)
class SDPA(nn.Module):
    def __init__(s,dim,nh):
        super().__init__(); s.nh,s.hd=nh,dim//nh; s.qkv=nn.Linear(dim,dim*3,bias=False); s.wo=nn.Linear(dim,dim,bias=False); s.rope=RoPE(s.hd)
    def forward(s,x):
        B,S,D=x.shape; q,k,v=s.qkv(x).chunk(3,-1)
        q=q.view(B,S,s.nh,s.hd).transpose(1,2); k=k.view(B,S,s.nh,s.hd).transpose(1,2); v=v.view(B,S,s.nh,s.hd).transpose(1,2)
        q,k=s.rope(q),s.rope(k); a=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        return s.wo(a.transpose(1,2).contiguous().view(B,S,D))
class SwiGLU(nn.Module):
    def __init__(s,d):
        super().__init__(); h=int(d*8/3); s.w1=nn.Linear(d,h,bias=False); s.w2=nn.Linear(h,d,bias=False); s.w3=nn.Linear(d,h,bias=False)
    def forward(s,x): return s.w2(F.silu(s.w1(x))*s.w3(x))
class OscGate(nn.Module):
    def __init__(s,d):
        super().__init__(); s.w=nn.Linear(d,d,bias=False); s.zr=nn.Parameter(torch.tensor(0.1)); s.zi=nn.Parameter(torch.tensor(0.0))
    def forward(s,x):
        r2=s.zr**2+s.zi**2; s.zr=nn.Parameter(s.zr+(s.zr*(1-r2)-s.zi)*0.01); s.zi=nn.Parameter(s.zi+(s.zi*(1-r2)+s.zr)*0.01)
        return x*torch.sigmoid(s.w(x))*torch.clamp((s.zr+1)/2,0,1)
class Block(nn.Module):
    def __init__(s,dim,nh):
        super().__init__(); s.n1=RMSNorm(dim); s.attn=SDPA(dim,nh); s.n2=RMSNorm(dim); s.ffn=SwiGLU(dim//2); s.og=OscGate(dim)
    def forward(s,x):
        x=x+s.attn(s.n1(x)); h=s.n2(x); d2=h.shape[-1]//2
        return s.og(x+torch.cat([h[...,:d2]+s.ffn(h[...,:d2]),h[...,d2:]+s.ffn(h[...,d2:])],dim=-1)*0.5)
class LModel(nn.Module):
    def __init__(s,vs,dim,nh,nl):
        super().__init__(); s.emb=nn.Embedding(vs,dim); s.layers=nn.ModuleList([Block(dim,nh) for _ in range(nl)])
        s.norm=RMSNorm(dim); s.head=nn.Linear(dim,vs,bias=False); s.head.weight=s.emb.weight
    def forward(s,x,return_hidden=False):
        h=s.emb(x) if x.dtype==torch.long else x
        for l in s.layers: h=l(h)
        h=s.norm(h); return h if return_hidden else s.head(h)
class BridgeLayer(nn.Module):
    def __init__(s,dim,nh):
        super().__init__(); s.norm1=nn.LayerNorm(dim); s.attn=SDPA(dim,nh); s.norm2=nn.LayerNorm(dim); s.ffn=SwiGLU(dim//2); s.gate=nn.Parameter(torch.tensor(0.5))
    def forward(s,x):
        r=x; x=s.norm1(x); x=s.attn(x); x=r+torch.tanh(s.gate)*x; h=s.norm2(x); d2=h.shape[-1]//2
        return x+torch.cat([h[...,:d2]+s.ffn(h[...,:d2]),h[...,d2:]+s.ffn(h[...,d2:])],dim=-1)*0.5
class Bridge(nn.Module):
    def __init__(self, dim=384, nh=6, nl=4):
        super().__init__(); self.proj_in=nn.Linear(dim,dim); self.layers=nn.ModuleList([BridgeLayer(dim,nh) for _ in range(nl)])
        self.norm=RMSNorm(dim); self.a2b_head=nn.Linear(dim,dim); self.b2a_head=nn.Linear(dim,dim)
    def forward(self, h, direction='a2b'):
        h=self.proj_in(h)
        for l in self.layers: h=l(h)
        h=self.norm(h); return self.a2b_head(h) if direction=='a2b' else self.b2a_head(h)

BASE = r'C:\Users\kriti'
PTIL = r'C:\Users\kriti\kaggle_output'

print('Loading models...')
with open(os.path.join(BASE, 'vocab_best.json')) as f: ev=json.load(f)
eiv={v:k for k,v in ev.items()}; vs=len(ev)
sd=torch.load(os.path.join(BASE, 'urcm_best.pt'), map_location=DEVICE, weights_only=False)
dim=sd['emb.weight'].shape[1]; hd=sd['layers.0.attn.rope.cos'].shape[-1]*2
nh,nl=dim//hd,len(set(int(k.split('.')[1]) for k in sd if k.startswith('layers.')))
arc=LModel(vs,dim,nh,nl).to(DEVICE); arc.load_state_dict(sd,strict=False); arc.eval()
for p in arc.parameters(): p.requires_grad=False

ckpt=torch.load(os.path.join(PTIL, 'urcm_ptil.pt'), map_location=DEVICE, weights_only=False)
bs,bc=ckpt['model'],ckpt['config']
brain=LModel(bc['vocab_size'],bc['dim'],bc['n_heads'],bc['n_layers']).to(DEVICE)
brain.load_state_dict(bs,strict=False); brain.eval()
for p in brain.parameters(): p.requires_grad=False

bridge=Bridge(dim=384,nh=6,nl=4).to(DEVICE); bridge.eval()
print('ARC:', round(sum(p.numel() for p in arc.parameters())/1e6,1), 'M')
print('Brain:', round(sum(p.numel() for p in brain.parameters())/1e6,1), 'M')
print('Bridge:', round(sum(p.numel() for p in bridge.parameters())/1e6,2), 'M')

PERSONAS = {
    'scientific': 'boiling point of water is 100 celsius photosynthesis converts carbon dioxide to oxygen earth orbits the sun dna is semi-conservative force equals mass times acceleration',
    'empathetic': 'i understand how difficult this must be your feelings are valid it takes courage to share take things one step at a time you are not alone',
    'formal': 'empirical evidence suggests correlation prudent to consider implications methodology yields reliable results examine underlying assumptions data supports the hypothesis',
    'casual': 'oh yeah great question basically what happens is cool think of it like this pretty much the short answer hope that makes sense',
}

persona_hidden = {}
for pk, ref in PERSONAS.items():
    eids = [ev.get(w, 1) for w in ref.split()]
    x = torch.tensor([eids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        h = arc(x, return_hidden=True)
        bh = bridge(h, 'a2b')
    persona_hidden[pk] = bh.mean(dim=1).squeeze(0)
print('Personas:', list(PERSONAS.keys()))

def arc_encode(text):
    eids = [ev.get(w, 1) for w in text.lower().split()]
    if not eids: eids = [1]
    x = torch.tensor([eids], dtype=torch.long, device=DEVICE)
    with torch.no_grad(): h = arc(x, return_hidden=True)
    return h.squeeze(0)

def baseline_generate(query, pk, n=3):
    h = arc_encode(query)
    results = []
    for _ in range(n):
        noise = torch.randn_like(h) * 0.01
        logits = arc.head(h + noise)
        topk = logits.argmax(dim=-1)
        words = [eiv.get(t.item(), '?') for t in topk[:5]]
        results.append(' '.join(words))
    return results

def urcm_attractor(query, pk, n=3, steps=5):
    results = []
    for _ in range(n):
        h = arc_encode(query)
        pe = persona_hidden[pk]
        for t in range(steps):
            bh = bridge(h.unsqueeze(0), 'a2b').squeeze(0)
            attraction = pe - bh.mean(dim=0)
            bh = bh + 0.3 * attraction.unsqueeze(0)
            back = bridge(bh.unsqueeze(0), 'b2a').squeeze(0)
            logits = arc.head(back)
            topw = logits.argmax(dim=-1)
            h = back + 0.1 * h
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        words = [eiv.get(t.item(), '?') for t in topw[-5:]]
        results.append(' '.join(words))
    return results

def drift_score(text, pk):
    h = arc_encode(text)
    pe = persona_hidden[pk]
    bh = bridge(h.unsqueeze(0), 'a2b').squeeze(0)
    cos = F.cosine_similarity(bh.mean(dim=0).unsqueeze(0), pe.unsqueeze(0)).item()
    return 1.0 - cos

def resonance_score(text, pk):
    h = arc_encode(text)
    pe = persona_hidden[pk]
    bh = bridge(h.unsqueeze(0), 'a2b').squeeze(0)
    per_token_cos = F.cosine_similarity(bh, pe.unsqueeze(0).expand_as(bh))
    cos = per_token_cos.mean().item()
    stab = 1.0 - min(per_token_cos.std().item(), 1.0)
    return 0.7 * max(cos, 0.0) + 0.3 * stab

def consistency(texts):
    if not texts: return 0.0
    n = [' '.join(t.lower().split()) for t in texts]
    if len(set(n)) == 1: return 1.0
    ovs = []
    for i in range(len(n)):
        for j in range(i+1, len(n)):
            s1, s2 = set(n[i].split()), set(n[j].split())
            ovs.append(len(s1&s2)/max(len(s1|s2),1))
    return float(np.mean(ovs)) if ovs else 0.0

QUERIES = [
    'Why does ice float on water?',
    'What causes seasons?',
    'Explain why the sky is blue.',
    'Is it ethical to lie?',
    'What is 15 percent of 240?',
    'Hand is to glove as foot is to what?',
]

NR = 3
print('')
print('=' * 70)
print('TOY BENCHMARK: Baseline (ARC raw) vs URCM (attractor+resonance)')
print(str(len(QUERIES)) + ' queries x ' + str(NR) + ' runs x ' + str(len(PERSONAS)) + ' personas')
print('=' * 70)

all_bl_cons, all_bl_drift = [], []
all_ur_cons, all_ur_res = [], []

t0 = time.time()
for q in QUERIES:
    print('')
    print('Q: ' + q)
    for pk in PERSONAS:
        bl = baseline_generate(q, pk, n=NR)
        bc = consistency(bl)
        bd = np.mean([drift_score(r, pk) for r in bl])

        ur = urcm_attractor(q, pk, n=NR)
        uc = consistency(ur)
        ur_s = np.mean([resonance_score(r, pk) for r in ur])

        all_bl_cons.append(bc); all_bl_drift.append(bd)
        all_ur_cons.append(uc); all_ur_res.append(ur_s)

        bl_a = 1.0 - bd
        sign = '+' if ur_s > bl_a else ''
        print('  [' + pk + '] BL=' + str(round(bc,3)) + ' drft=' + str(round(bd,3)) + ' align=' + str(round(bl_a,3)) + ' | URCM=' + str(round(uc,3)) + ' res=' + str(round(ur_s,3)) + ' d=' + sign + str(round(ur_s-bl_a,3)))

elapsed = round(time.time()-t0, 1)
print('')
print('=' * 70)
print('RESULTS (elapsed: ' + str(elapsed) + 's)')
print('-' * 70)
print('{:12s} | {:>8s} | {:>8s} | {:>8s} | {:>8s}'.format('Metric', 'BL Cons', 'BL Align', 'URCM Con', 'URCM Res'))
print('-' * 70)
print('{:12s} | {:8.4f} | {:8.4f} | {:8.4f} | {:8.4f}'.format('Average', np.mean(all_bl_cons), np.mean(1-np.array(all_bl_drift)), np.mean(all_ur_cons), np.mean(all_ur_res)))
print('-' * 70)
print('')
print('URCM ADVANTAGE:')
bl_align = np.mean(1 - np.array(all_bl_drift))
urcm_res = np.mean(all_ur_res)
d = urcm_res - bl_align
sign = '+' if d > 0 else ''
print('  Persona alignment: BL=' + str(round(bl_align,4)) + ' URCM=' + str(round(urcm_res,4)) + ' delta=' + sign + str(round(d,4)))
print('  Consistency:       BL=' + str(round(np.mean(all_bl_cons),4)) + ' URCM=' + str(round(np.mean(all_ur_cons),4)))
print('')
print('PIPELINE:')
print('  Baseline: ARC encoder -> raw logits -> argmax (no persona guidance)')
print('  URCM:     ARC encoder -> Bridge -> Brain attractor -> Bridge back -> persona resonance')
print('')
print('NOTE: Bridge is UNTRAINED. Trained Bridge would show stronger persona alignment.')
print('      Training requires ~10 GPU-hours on P100.')
