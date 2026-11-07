"""Extra checks behind SUMMARY.md: is the fine-tuned FR model really worse
(bootstrap over the 1000 test signals), does the counting gain sit in the FC
model or the FR model (swap test), and which frequencies the FFT misses.

  python experiments/6_final_comparison/extra_checks.py
"""
import sys; sys.path.insert(0, "experiments")
import numpy as np,torch
from common import *
from data import fr as F
from e2_counting import run_net
o=load_deepfreq(PRETRAINED);t=load_deepfreq(FINETUNED)
rng=np.random.default_rng(0)
def per_signal(est,tg):
    c=(tg>=-.5).sum(1);m=[]
    for e,g,k in zip(est,tg,c):
        g=g[g>=-.5];e=e[e>=-.5]
        d=np.abs(((g[:,None]-e[None])+.5)%1-.5).min(1)
        m.append((d>1/100).sum()/k)
    return np.array(m)
print('snr fnr_o fnr_t diff ci | medO medT | rawbias O T rawstd O T | peakO peakT')
for s in [0,10,20,25,30,40,50]:
    sig,tg,c=load_test_set(s)
    ro,ao=run_net(*o,sig);rt,at=run_net(*t,sig)
    eo=F.find_freq(ro,c,XGRID,MAX_FREQ);et=F.find_freq(rt,c,XGRID,MAX_FREQ)
    mo,mt=per_signal(eo,tg),per_signal(et,tg)
    d=mt-mo;bs=[d[rng.integers(0,len(d),len(d))].mean()*100 for _ in range(2000)]
    print(s,round(mo.mean()*100,3),round(mt.mean()*100,3),round(d.mean()*100,3),np.round(np.percentile(bs,[2.5,97.5]),3),'|',round(matched_median(eo,tg),4),round(matched_median(et,tg),4),'|',round((ao-c).mean(),3),round((at-c).mean(),3),round((ao-c).std(),3),round((at-c).std(),3),'| acc',round(np.mean(np.rint(ao)==c)*100,1),round(np.mean(np.rint(at)==c)*100,1))
# FFT misses vs separation at 50 dB
sig,tg,c=load_test_set(50);x=to_complex(sig)
e=periodogram_peaks(x,c)
rows=[]
clean=to_complex(np.load('test_dataset/infdB.npy'))
for i in range(len(x)):
    g=tg[i][tg[i]>=-.5];est=e[i][e[i]>=-.5]
    if len(g)<2:
        continue
    A=np.exp(2j*np.pi*np.outer(np.arange(50),g));amp=np.abs(np.linalg.lstsq(A,clean[i],rcond=None)[0])
    dd=np.abs(((g[:,None]-g[None])+.5)%1-.5);np.fill_diagonal(dd,9);near=dd.min(1)*50
    miss=np.abs(((g[:,None]-est[None])+.5)%1-.5).min(1)>1/100
    rows+=list(zip(near,miss,amp/amp.max()))
r=np.array(rows,float)
for a,b in [(1,1.25),(1.25,1.5),(1.5,2),(2,3),(3,99)]:
    k=(r[:,0]>=a)&(r[:,0]<b);print('sep',a,b,'n',k.sum(),'miss%',round(r[k,1].mean()*100,1))
for a,b in [(0,.1),(.1,.25),(.25,.5),(.5,1.01)]:
    k=(r[:,2]>=a)&(r[:,2]<b);print('relamp',a,b,'n',k.sum(),'miss%',round(r[k,1].mean()*100,1))
print('share of freq with sep<2',round((r[:,0]<2).mean()*100,1),'total freqs',int((tg>=-.5).sum()),'mean count',c.mean())

# ---- part 2: median-error bootstrap and model swap
rng=np.random.default_rng(1)
def errs(est,tg):
    out=[]
    for e,g in zip(est,tg):
        g=g[g>=-.5];e=e[e>=-.5]
        out.append(np.abs(((g[:,None]-e[None])+.5)%1-.5).min(1)*50)
    return out
for s in [20,30,50]:
    sig,tg,c=load_test_set(s)
    x=torch.from_numpy(sig).float()
    with torch.no_grad(): ro,rt=o[0](x),t[0](x)
    eo=F.find_freq(ro.numpy(),c,XGRID,MAX_FREQ);et=F.find_freq(rt.numpy(),c,XGRID,MAX_FREQ)
    a,b=errs(eo,tg),errs(et,tg);ds=[]
    for _ in range(500):
        i=rng.integers(0,len(a),len(a))
        ds.append(np.median(np.concatenate([b[j] for j in i]))-np.median(np.concatenate([a[j] for j in i])))
    print('snr',s,'median diff CI',np.round(np.percentile(ds,[2.5,97.5]),5))
    with torch.no_grad():
        for n,fr_m,fc_m in [('FRo+FCo',o[0],o[1]),('FRt+FCo',t[0],o[1]),('FRo+FCt',o[0],t[1]),('FRt+FCt',t[0],t[1])]:
            raw=fc_m(fr_m(x)).view(-1).numpy();k=np.clip(np.rint(raw),1,10)
            print(' ',n,'count acc',round(np.mean(k==c)*100,1))
