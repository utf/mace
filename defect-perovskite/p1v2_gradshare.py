import numpy as np, torch, sys
from ase.io import read
from mace import data as mace_data, tools
from mace.tools import torch_geometric as tg
from mace.modules.loss import UniversalFieldLoss
dev=torch.device('cuda'); torch.set_default_dtype(torch.float32)
tag=sys.argv[1]
m=torch.load(f'/home/alex/runs/dscc/p1v2/ckpt_probe/{tag}/base_v6_{tag}.model',map_location='cpu',weights_only=False).to(dev).float().train()
zt=tools.AtomicNumberTable([int(z) for z in m.atomic_numbers]); ks=mace_data.KeySpecification.from_defaults()
ks.info_keys.update(energy="REF_energy", stress="REF_stress", polarization="REF_polarization", polarizability="REF_polarizability"); ks.arrays_keys.update(forces="REF_forces", becs="REF_becs")
W=dict(energy_weight=10, forces_weight=10, stress_weight=1, polarization_weight=1000, becs_weight=100, polarizability_weight=100)
losses={k: UniversalFieldLoss(**{kk:(v if kk==k else 0.0) for kk,v in W.items()}).to(dev) for k in W}
def batch(frames, head):
    ads=[mace_data.AtomicData.from_config(mace_data.config_from_atoms(a,key_specification=ks,head_name=head),z_table=zt,cutoff=float(m.r_max),heads=list(m.heads)) for a in frames]
    return next(iter(tg.dataloader.DataLoader(ads,batch_size=len(ads)))).to(dev)
backbone=[p for n,p in m.named_parameters() if 'readout' not in n]
def gshare(frames, head, label):
    out={}
    for k,lf in losses.items():
        tot=0.0
        for j in range(0,len(frames),2):
            b=batch(frames[j:j+2],head); m.zero_grad(set_to_none=True)
            o=m(b.to_dict(),training=True,compute_force=True,compute_stress=True,compute_polarization=True,compute_becs=True,compute_polarizability=True)
            l=lf(b,o); l.backward()
            g=torch.sqrt(sum((p.grad.float()**2).sum() for p in backbone if p.grad is not None)); tot+=float(g)*2/len(frames)
        out[k]=tot
    print(f'{label}: backbone grad norm per term (2-frame batches, mean):', {k: f'{v:.2e}' for k,v in out.items()}, 'sum', f'{sum(out.values()):.2e}', flush=True)
    return out
va=read('/home/alex/runs/dscc/p1v2/valid_v2.xyz','::19')[:8]
rp_all=read('/home/alex/runs/dscc/p1v2/replay_v2.xyz',':')
rp=rp_all[::58][:160][:16]
d=gshare(va,'Default','Default(valid, 8 frames)'); r=gshare(rp,'pt_head','pt_head(replay, 16 strided frames)')
print('epoch budget (grad-norm x frames): Default', f'{sum(d.values())*1619:.2e}', 'replay', f'{sum(r.values())*8439:.2e}', flush=True)
# alpha labels on the worst replay cells
m.eval()
for idx in (3074,2784,3016,5452,3944,1856,4466,7424,7946,5394):
    a=rp_all[idx]; b=batch([a],'pt_head')
    o=m(b.to_dict(),training=True,compute_force=False,compute_polarization=True,compute_becs=False,compute_polarizability=True)
    ref=np.asarray(a.info['REF_polarizability']).reshape(3,3); pred=o['polarizability'].detach().cpu().numpy().reshape(3,3)
    print(f'{a.get_chemical_formula():10s} V/atom {a.get_volume()/len(a):5.1f}  REF alpha diag {np.diag(ref).round(2)} tr/3 {np.trace(ref)/3:7.2f}  pred tr/3 {np.trace(pred)/3:7.2f}  |dalpha| {np.abs(pred-ref).max():.2f}', flush=True)
tr=np.array([np.trace(np.asarray(a.info['REF_polarizability']).reshape(3,3))/3 for a in rp_all])
print('REF alpha tr/3 over all replay: median', np.median(tr).round(2), 'p90', np.percentile(tr,90).round(2), 'p99', np.percentile(tr,99).round(1), 'max', tr.max().round(1), 'n>10:', int((tr>10).sum()), 'n>50:', int((tr>50).sum()))
