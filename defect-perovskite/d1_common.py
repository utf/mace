"""Load a MACE training checkpoint (a cuEquivariance state dict when the run used cueq) into a
same-architecture saved .model: the e3nn template is converted to cueq FIRST so every key --
including `products.N.symmetric_contractions.weight/projection`, which an e3nn model does not
have -- loads strictly. (Loading with strict=False into an e3nn template silently keeps the
template's product weights: the in-flight L0 reads of 2026-09-15 morning had that flaw.)"""
import copy, os, torch


def load_checkpoint(ckpt_path, template_path, device="cuda"):
    obj = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    if not (isinstance(obj, dict) and "model" in obj):
        return obj.to(device).eval(), "saved model"
    sd = obj["model"]
    tpl = copy.deepcopy(torch.load(os.path.expanduser(template_path), map_location="cpu", weights_only=False))
    dt = next(v.dtype for v in sd.values() if v.is_floating_point())
    tpl = tpl.to(dt)
    cueq_keys = any("symmetric_contractions.weight" in k for k in sd)
    if cueq_keys:
        from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
        tpl = run_e3nn_to_cueq(tpl, device=device)
    tsd = tpl.state_dict()
    sd = {k: (v.reshape(tsd[k].shape) if k in tsd and v.numel() == tsd[k].numel() and v.shape != tsd[k].shape else v) for k, v in sd.items()}
    res = tpl.load_state_dict(sd, strict=False)
    bad_missing = [k for k in res.missing_keys if not (k.endswith(".bias") or k.endswith("output_mask") or "graph" in k)]
    bad_unexp = [k for k in res.unexpected_keys if "graph" not in k]
    if bad_missing or bad_unexp:
        raise RuntimeError(f"checkpoint/template mismatch: missing {bad_missing[:5]} unexpected {bad_unexp[:5]}")
    return tpl.to(device).eval(), f"checkpoint (raw weights, {'cueq' if cueq_keys else 'e3nn'} layout)"
