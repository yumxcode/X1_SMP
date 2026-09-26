"""One-episode rollout dump: distinguish obs-mismatch vs PD-mismatch vs skill-gap."""
import sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402

pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
s2s.m.dof_damping[s2s.vadr] = s2s.kd  # implicit kd
s2s.reset(seed=7)

n = 90  # 3 s
print(f"{'t':>5} {'rootz':>6} {'pitch':>7} {'vx':>6} {'|a|med':>7} "
      f"{'sat%':>5} {'|tau|med':>8} {'clip%':>6}  q_lhip q_rhip q_lknee")
for it in range(n):
    o = s2s.obs()
    a = pol.forward(o)
    q_tar = np.clip(a, -s2s.a_bound, s2s.a_bound)
    sat = 100 * np.mean(np.abs(a) >= s2s.a_bound * 0.999)
    taus = []
    for _ in range(4):  # 30 Hz control, 120 Hz physics
        q = s2s.d.qpos[s2s.qadr]
        qd = s2s.d.qvel[s2s.vadr]
        tau = np.clip(s2s.kp * (q_tar - q), -s2s.eff, s2s.eff)
        taus.append(np.abs(tau))
        s2s.d.ctrl[:] = tau
        s2s.mj.mj_step(s2s.m, s2s.d)
    taus = np.mean(taus, axis=0)
    clip = 100 * np.mean(taus >= s2s.eff * 0.999)
    d = s2s.d
    R = d.xmat[s2s.torso_bid].reshape(3, 3)
    pitch = np.degrees(np.arctan2(R[2, 0], R[0, 0]))
    q = d.qpos[s2s.qadr]
    if it % 3 == 0:
        print(f"{it/30:5.2f} {d.qpos[2]:6.3f} {pitch:7.1f} {d.qvel[0]:6.2f} "
              f"{np.median(np.abs(a)):7.3f} {sat:5.1f} {np.median(taus):8.1f} "
              f"{clip:6.1f}  {q[17]:+.2f} {q[23]:+.2f} {q[20]:+.2f}")
    if d.qpos[2] < 0.2 and it > 10:
        print(f"FALL at t={it/30:.2f}s rootz={d.qpos[2]:.3f}")
        break
