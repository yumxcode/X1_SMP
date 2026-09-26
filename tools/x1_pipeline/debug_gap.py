"""Locate the Isaac->MuJoCo transfer gap:
A) mid-run start (demo frame + finite-diff velocity): does the policy run?
B) standing start with friction sweep."""
import sys
from pathlib import Path
import numpy as np
import pickle

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402


def rollout(s2s, seconds=10.0):
    d = s2s.d
    fell = None
    xs = []
    for it in range(int(seconds * 30)):
        o = s2s.obs()
        a = s2s.policy.forward(o)
        q_tar = np.clip(a, -s2s.a_bound, s2s.a_bound)
        for _ in range(4):
            tau = np.clip(s2s.kp * (q_tar - d.qpos[s2s.qadr]),
                          -s2s.eff, s2s.eff)
            d.ctrl[:] = tau
            s2s.mj.mj_step(s2s.m, d)
        xs.append(d.qpos[0])
        if d.qpos[2] < 0.30 and fell is None:
            fell = it / 30
            break
    speed = 0.0
    if len(xs) > 30:
        speed = np.abs(np.diff(xs[10:])).mean() * 30
    return fell, speed


mo = pickle.load(open(REPO / "data/motions/x1/x1_sprint1_subject4_seg3.pkl", "rb"))
F = np.array(mo["frames"], dtype=np.float64)

pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
s2s.m.dof_damping[s2s.vadr] = s2s.kd
d, m = s2s.d, s2s.m

# ---- A) mid-run starts from several demo frames ----
print("== mid-run starts (frame, fall, speed) ==")
for fidx in (60, 100, 150, 200):
    i0, i1 = fidx - 1, fidx + 1
    dt = 1.0 / 30.0
    d.qpos[:3] = F[fidx, 0:3]
    em = F[fidx, 3:6]
    ang = np.linalg.norm(em)
    ax = em / max(ang, 1e-8)
    d.qpos[3:7] = [np.cos(ang / 2), *(ax * np.sin(ang / 2))]
    d.qpos[s2s.qadr] = F[fidx, 6:35]
    # velocity from finite differences (world root vel; hinge vels)
    d.qvel[:3] = (F[i1, 0:3] - F[i0, 0:3]) / (2 * dt)
    # approx ang vel from expmap diff (small-angle)
    dw = (F[i1, 3:6] - F[i0, 3:6]) / (2 * dt)
    R = d.xmat[s2s.root_bid].reshape(3, 3) if False else None
    d.qvel[3:6] = dw  # local frame approx
    d.qvel[s2s.vadr] = (F[i1, 6:35] - F[i0, 6:35]) / (2 * dt)
    s2s.mj.mj_forward(m, d)
    fell, spd = rollout(s2s)
    print(f"  frame {fidx}: {'FELL %.2fs' % fell if fell else 'UP 10s'} "
          f"speed {spd:.2f} m/s")

# ---- B) standing start, friction sweep ----
print("== standing start friction sweep ==")
for fr in (0.8, 1.0, 1.2, 1.6):
    s2s.m.geom_friction[:, 0] = fr
    res = []
    for seed in (5, 6, 7):
        s2s.reset(seed)
        fell, spd = rollout(s2s)
        res.append((fell, spd))
    txt = " ".join(f"{'%.1fs' % f if f else 'UP'}/{s:.1f}" for f, s in res)
    print(f"  friction {fr}: {txt}")
