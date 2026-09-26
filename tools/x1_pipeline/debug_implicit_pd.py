"""Test implicit-damping PD (option C) for sim2sim: kd via m.dof_damping
(MuJoCo Euler integrates joint damping implicitly), explicit clip(kp*err)."""
import sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

import sim2sim_validate as sv  # noqa: E402
from sim2sim_validate import Policy  # noqa: E402


def make_implicit_pd(s2s):
    m = s2s.m
    m.dof_damping[s2s.vadr] = s2s.kd
    s2s.implicit_kd = True
    return s2s


def run_hold(s2s, target, seconds=3.0, label=""):
    d = s2s.d
    s2s.reset(seed=3)
    fell = None
    for it in range(int(seconds * 30)):
        q = d.qpos[s2s.qadr]
        tau = np.clip(s2s.kp * (target - q), -s2s.eff, s2s.eff)
        d.ctrl[:] = tau
        for _ in range(4):
            s2s.mj.mj_step(s2s.m, d)
        if d.qpos[2] < 0.30:
            fell = it / 30
            break
    R = d.xmat[s2s.torso_bid].reshape(3, 3)
    pitch = np.degrees(np.arctan2(R[2, 0], R[0, 0]))
    print(f"[{label}] rootz={d.qpos[2]:.3f} pitch={pitch:.1f} "
          f"{'FELL %.2fs' % fell if fell else 'UP'}")
    return fell


pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = make_implicit_pd(sv.Sim2Sim(pol))
run_hold(s2s, s2s.home, label="implicit-kd hold-home")
run_hold(s2s, np.zeros(29), label="implicit-kd zero")
