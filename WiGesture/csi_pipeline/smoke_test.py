"""Verification smoke tests (plan steps 1-8). Run: python -m csi_pipeline.smoke_test"""

from __future__ import annotations

import numpy as np
import torch

from .config import CONFIG, resolve_device
from .data import preprocess as pp
from .data import windowing as wd
from .data.loader import (
    iq_to_amp_phase,
    load_person_recordings,
    parse_csi_csv,
    resample_uniform,
)
from .models import build_encoder
from .training import contrastive as cl


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}{(' -- ' + detail) if detail else ''}")
    assert cond, f"{name} failed: {detail}"


def main():
    cfg = CONFIG
    path = f"{cfg['data_root']}/ID1/applause.csv"

    print("Step 1: loader")
    iq, label, ts = parse_csi_csv(path)
    check("iq shape [N,104]", iq.ndim == 2 and iq.shape[1] == 104, str(iq.shape))
    check("label", label == "applause", label)
    amp, phase = iq_to_amp_phase(iq)
    check("amp/phase [N,52]", amp.shape[1] == 52 and phase.shape[1] == 52, str(amp.shape))
    check("amp finite & >=0", np.all(np.isfinite(amp)) and np.all(amp >= 0))

    print("Step 2: resample")
    amp_u, phase_u = resample_uniform(amp, phase, ts, cfg["resample"]["target_fs"])
    check("resampled rows grew (~100Hz)", amp_u.shape[0] >= amp.shape[0], f"{amp.shape[0]}->{amp_u.shape[0]}")
    check("resample finite", np.all(np.isfinite(amp_u)) and np.all(np.isfinite(phase_u)))

    print("Step 3: preprocess")
    a = pp.amplitude_pipeline(amp_u, cfg["resample"]["target_fs"])
    p = pp.phase_pipeline(phase_u)
    check("no NaN/Inf after clean", np.all(np.isfinite(a)) and np.all(np.isfinite(p)))
    mean, std = pp.fit_zscore(a)
    z = pp.apply_zscore(a, mean, std)
    check("zscore mean~0 std~1", abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 0.05, f"mean={z.mean():.2e} std={z.std():.3f}")

    print("Step 4: windowing (ID1)")
    recs = load_person_recordings("ID1", cfg["data_root"])
    for r in recs:
        r.amp = pp.amplitude_pipeline(r.amp, r.fs)
        r.phase = pp.phase_pipeline(r.phase)
    train, val, test = wd.split_person(recs, cfg)
    stats = wd.fit_zscore_stats(train)
    Xtr, ytr = wd.windows_to_arrays(train, stats, cfg["classes"])
    shape = tuple(cfg["tensor"]["shape"])
    check("tensor shape [2,52,250]", Xtr.shape[1:] == shape, str(Xtr.shape))
    check("all 6 classes in train", len(set(ytr.tolist())) == 6, str(sorted(set(ytr.tolist()))))
    total = len(train) + len(val) + len(test)
    print(f"    windows: train {len(train)} val {len(val)} test {len(test)} (total {total})")

    print("Step 5: leakage assertion (no raw-time overlap per recording)")
    def intervals(ws):
        d = {}
        for w in ws:
            d.setdefault(w.src_recording_id, []).append((w.start, w.start + cfg["window"]["length"]))
        return d
    itr, ite = intervals(train), intervals(test)
    overlap = False
    for rid in set(itr) & set(ite):
        for s1, e1 in itr[rid]:
            for s2, e2 in ite[rid]:
                if s1 < e2 and s2 < e1:
                    overlap = True
    check("zero train/test raw-time overlap", not overlap)
    check("all 6 classes in test", len(set(w.label for w in test)) == 6)

    print("Step 6: NT-Xent numeric")
    B, D = 8, 16
    base = torch.randn(B, D)
    z_same = torch.cat([base, base], dim=0)
    z_rand = torch.randn(2 * B, D)
    l_same = cl.nt_xent_loss(z_same, 0.2).item()
    l_rand = cl.nt_xent_loss(z_rand, 0.2).item()
    check("loss finite", np.isfinite(l_same) and np.isfinite(l_rand))
    check("identical-views loss < random", l_same < l_rand, f"same={l_same:.3f} rand={l_rand:.3f}")

    print("Step 7: encoder shape contract")
    x = torch.randn(4, *shape)
    for name in cfg["encoders"]:
        enc = build_encoder(name, cfg)
        out = enc(x)
        check(f"{name} -> [4,{cfg['embed_dim']}]", tuple(out.shape) == (4, cfg["embed_dim"]), str(tuple(out.shape)))

    print("Step 8: augmentation shape preserved")
    v1, v2 = cl.two_views(x, cfg["simclr"]["aug"])
    check("augment preserves shape", v1.shape == x.shape and v2.shape == x.shape)

    print(f"\nDevice resolved: {resolve_device(cfg)}")
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
