"""Physical feasibility of the retargeted demo via inverse dynamics:
1) kinematic replay: foot penetration, joint-limit violations
2) mj_inverse torque demand vs effort limits (the X1 envelope test)"""
import sys
from pathlib import Path
import numpy as np
import pickle

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402

pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
d, m = s2s.d, s2s.m

mo = pickle.load(open(REPO / "data/motions/x1/x1_sprint1_subject4_seg3.pkl", "rb"))
F = np.array(mo["frames"], dtype=np.float64)
fps, ts = mo["fps"], mo.get("time_scale", 1.0)
dt = 1.0 / fps / ts


def set_state(i, qvel_from_diff=False):
    f = F[i]
    d.qpos[:3] = f[0:3]
    em = f[3:6]
    ang = np.linalg.norm(em)
    if ang < 1e-8:
        d.qpos[3:7] = [1, 0, 0, 0]
    else:
        ax = em / ang
        d.qpos[3:7] = [np.cos(ang / 2), *(ax * np.sin(ang / 2))]
    d.qpos[s2s.qadr] = f[6:35]
    d.qvel[:] = 0
    s2s.mj.mj_forward(m, d)


# ---------- 1) kinematic replay checks ----------
pen, jv = [], []
floor_gids = [g for g in range(m.ngeom)
              if (m.geom(g).name or "") == "floor"]
foot_gids = [g for g in range(m.ngeom)
             if (m.geom(g).name or "").endswith("_sole")]
for i in range(len(F)):
    set_state(i)
    for g in foot_gids:
        pen.append(d.geom_xpos[g][2] - m.geom_size[g][0] * 0)  # sole center z
    dof = F[i, 6:35]
    lo = -s2s.a_bound / 1.4  # actual limits are a_bound/1.4
    hi = s2s.a_bound / 1.4
    jv.append(np.maximum(lo - dof, dof - hi).max())
pen = np.array(pen)
print(f"[kinematic] sole-center z: min {pen.min():.4f} max {pen.max():.4f} m")
print(f"[kinematic] joint-limit violation: max {np.max(jv):.4f} rad")

# ---------- 2) inverse dynamics torque demand ----------
n = len(F)
qpos_all = np.zeros((n, m.nq))
for i in range(n):
    f = F[i]
    qpos_all[i, :3] = f[0:3]
    em = f[3:6]
    ang = np.linalg.norm(em)
    if ang < 1e-8:
        qpos_all[i, 3:7] = [1, 0, 0, 0]
    else:
        ax = em / ang
        qpos_all[i, 3:7] = [np.cos(ang / 2), *(ax * np.sin(ang / 2))]
    qpos_all[i, s2s.qadr] = f[6:35]

# smooth qpos mildly (30fps noise -> ID spikes)
from scipy.ndimage import gaussian_filter1d  # noqa: E402
qs = gaussian_filter1d(qpos_all, sigma=2, axis=0, mode="nearest")

def qvel_at(i):
    v = np.zeros(m.nv)
    i0, i1 = max(i - 1, 0), min(i + 1, n - 1)
    s2s.mj.mj_differentiatePos(m, v, dt * (i1 - i0), qs[i0], qs[i1])
    return v

qv = np.array([qvel_at(i) for i in range(n)])
dv = np.gradient(qv, dt, axis=0)
qa = np.zeros_like(qv)
for i in range(n):
    qa[i, 6:] = dv[i, 6:]
    qa[i, :6] = dv[i, :6]

tau_dem = np.zeros((n, 29))
for i in range(n):
    d.qpos[:] = qs[i]
    d.qvel[:] = qv[i]
    d.qacc[:] = qa[i]
    s2s.mj.mj_inverse(m, d)
    tau_dem[i] = d.qfrc_inverse[s2s.vadr]

over = np.abs(tau_dem) > s2s.eff[None, :]
print(f"[ID] frames over effort limit: {over.any(axis=1).mean()*100:.1f}%")
print(f"[ID] peak demand / limit (worst joints):")
ratio = np.abs(tau_dem).max(axis=0) / s2s.eff
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402
order = np.argsort(-ratio)
for j in order[:8]:
    print(f"    {X1_DOF_ORDER[j]:>32}: {ratio[j]:5.2f}x  "
          f"(max {np.abs(tau_dem[:, j]).max():6.1f} vs {s2s.eff[j]:5.1f} Nm)")
print(f"[ID] median joint demand/limit ratio: {np.median(ratio):.2f}")
