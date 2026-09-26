"""Sweep extra slowdown factor k on existing retargeted pkl; ID feasibility.
k multiplies the current time_scale: acceleration demand ~ 1/k^2."""
import sys
from pathlib import Path
import numpy as np
import pickle

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402
from scipy.ndimage import gaussian_filter1d  # noqa: E402

pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
d, m = s2s.d, s2s.m


def resample(F, k):
    n = len(F)
    n_out = int(round(n * k))
    idx = np.clip(np.arange(n_out) / k, 0, n - 1 - 1e-6)
    i0 = np.floor(idx).astype(int)
    a = (idx - i0)[..., None]
    return F[i0] * (1 - a) + F[i0 + 1] * a


def qpos_of(f):
    q = np.zeros(m.nq)
    q[:3] = f[0:3]
    em = f[3:6]
    ang = np.linalg.norm(em)
    if ang < 1e-8:
        q[3:7] = [1, 0, 0, 0]
    else:
        ax = em / ang
        q[3:7] = [np.cos(ang / 2), *(ax * np.sin(ang / 2))]
    q[s2s.qadr] = f[6:35]
    return q


def id_demand(F, dt):
    n = len(F)
    qs = gaussian_filter1d(np.stack([qpos_of(f) for f in F]), sigma=2,
                           axis=0, mode="nearest")

    def qvel_at(i):
        v = np.zeros(m.nv)
        i0, i1 = max(i - 1, 0), min(i + 1, n - 1)
        s2s.mj.mj_differentiatePos(m, v, dt * (i1 - i0), qs[i0], qs[i1])
        return v

    qv = np.array([qvel_at(i) for i in range(n)])
    dv = np.gradient(qv, dt, axis=0)
    tau = np.zeros((n, 29))
    for i in range(n):
        d.qpos[:] = qs[i]
        d.qvel[:] = qv[i]
        d.qacc[:] = dv[i]
        s2s.mj.mj_inverse(m, d)
        tau[i] = d.qfrc_inverse[s2s.vadr]
    return tau


for pkl in ["x1_sprint1_subject4_seg3.pkl", "x1_sprint1_subject2_seg0.pkl",
            "x1_sprint1_subject4_seg0.pkl"]:
    mo = pickle.load(open(REPO / "data/motions/x1" / pkl, "rb"))
    F = np.array(mo["frames"], dtype=np.float64)
    print(f"\n=== {pkl} (baked warp {mo.get('time_scale',1):.2f}, {len(F)} frames) ===")
    for k in (1.0, 1.25, 1.5, 2.0):
        Fr = resample(F, k)  # more frames = more slowdown at fixed fps
        dt = 1.0 / mo["fps"]
        tau = id_demand(Fr, dt)
        ratio = np.abs(tau) / s2s.eff[None, :]
        frac = (ratio > 1.0).any(axis=1).mean()
        dist = np.linalg.norm(np.diff(Fr[:, 0:3], axis=0), axis=1).sum()
        v = dist / (len(Fr) * dt)
        print(f"  extra-k={k:.2f}: over-limit frames "
              f"{frac*100:5.1f}%  worst-joint {ratio.max():4.2f}x  "
              f"p99 {np.quantile(ratio.max(axis=1),0.99):4.2f}x  path-speed {v:.2f} m/s")
