"""Which joints diverge in the isaacPD replay?"""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj.pt",
                  map_location="cpu", weights_only=False)
pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
d, m = s2s.d, s2s.m
m.dof_damping[s2s.vadr] = 0
m.opt.timestep = 1.0 / 1200.0
fine = 10

d.qpos[:3] = traj["root_pos"][0].numpy()
q_ = traj["root_quat"][0].numpy()
d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
d.qpos[s2s.qadr] = traj["dof_pos"][0].numpy()
d.qvel[:3] = traj["root_vel"][0].numpy()
d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
d.qvel[s2s.vadr] = traj["dof_vel"][0].numpy()
s2s.mj.mj_forward(m, d)

errs = []
for t in range(60):
    a = traj["action"][t].numpy()
    q_tar = np.clip(a, -s2s.a_bound, s2s.a_bound)
    for _ in range(4 * fine):
        q = d.qpos[s2s.qadr]
        qd = d.qvel[s2s.vadr]
        tau = np.clip(s2s.kp * (q_tar - q) - s2s.kd * qd,
                      -s2s.eff, s2s.eff)
        d.ctrl[:] = tau
        s2s.mj.mj_step(m, d)
    errs.append(np.abs(d.qpos[s2s.qadr] - traj["dof_pos"][min(t + 1, 119)].numpy()))

E = np.stack(errs)
order = np.argsort(-E.max(axis=0))
print("worst joints (max err over 2s):")
for j in order[:6]:
    print(f"  {X1_DOF_ORDER[j]:>32}: max {E[:, j].max():.3f} "
          f"@t{E[:, j].argmax()/30:.2f}s  eff={s2s.eff[j]:.0f} "
          f"kp={s2s.kp[j]:.0f}")
print("\nerr>0.3 joints per step (first 15 steps):")
for t in range(0, 15, 3):
    bad = [X1_DOF_ORDER[j] for j in np.where(E[t] > 0.3)[0]]
    print(f"  t={t/30:.2f}: {bad}")
