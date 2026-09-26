"""Compare checkpoints by fall time & progress; friction sensitivity on final.
Uses the SAME implicit-kd fixed physics as sim2sim_validate."""
import sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402


def rollout(s2s, seed, seconds=10.0):
    s2s.reset(seed)
    d = s2s.d
    fell = None
    max_x = 0.0
    for it in range(int(seconds * 30)):
        o = s2s.obs()
        a = s2s.policy.forward(o)
        q_tar = np.clip(a, -s2s.a_bound, s2s.a_bound)
        for _ in range(4):
            q = d.qpos[s2s.qadr]
            tau = np.clip(s2s.kp * (q_tar - q), -s2s.eff, s2s.eff)
            d.ctrl[:] = tau
            s2s.mj.mj_step(s2s.m, d)
        max_x = max(max_x, d.qpos[0])
        if d.qpos[2] < 0.30 and fell is None:
            fell = it / 30
            break
    return fell, max_x, d.qpos[0]


ckpts = ["amp_it1000.pt", "amp_it2000.pt", "amp_it3000.pt", "amp_final.pt"]
for ck in ckpts:
    pol = Policy(str(REPO / "output/remote_ckpt" / ck))
    s2s = Sim2Sim(pol)
    s2s.m.dof_damping[s2s.vadr] = s2s.kd
    res = [rollout(s2s, s) for s in (11, 22)]
    falls = " ".join(f"{f if f else 99:.1f}s/{x:.1f}m" for f, x, _ in res)
    print(f"{ck:>16}: {falls}")

# friction sensitivity on final ckpt
pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
s2s.m.dof_damping[s2s.vadr] = s2s.kd
for fr in (1.5, 2.0):
    s2s.m.geom_friction[:, 0] = fr  # all geoms sliding friction
    res = [rollout(s2s, s) for s in (11, 22)]
    falls = " ".join(f"{f if f else 99:.1f}s/{x:.1f}m" for f, x, _ in res)
    print(f"{'final fr=%.1f' % fr:>16}: {falls}")
