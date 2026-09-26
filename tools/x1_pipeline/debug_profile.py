"""Divergence profile for the isaacPD fine-integration replay."""
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path("/Users/yumx/code/MimicKit")
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
sys.path.insert(0, str(REPO))

from sim2sim_validate import Policy, Sim2Sim  # noqa: E402

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj.pt",
                  map_location="cpu", weights_only=False)
pol = Policy(str(REPO / "output/remote_ckpt/amp_final.pt"))
s2s = Sim2Sim(pol)
d, m = s2s.d, s2s.m
m.dof_damping[s2s.vadr] = 0  # explicit kd (isaacPD)
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

print(f"{'t':>5} {'dof_med':>8} {'dof_max':>8} {'rootz':>6} {'z_err':>6}")
for t in range(90):
    a = traj["action"][t].numpy()
    q_tar = np.clip(a, -s2s.a_bound, s2s.a_bound)
    for _ in range(4 * fine):
        q = d.qpos[s2s.qadr]
        qd = d.qvel[s2s.vadr]
        tau = np.clip(s2s.kp * (q_tar - q) - s2s.kd * qd,
                      -s2s.eff, s2s.eff)
        d.ctrl[:] = tau
        s2s.mj.mj_step(m, d)
    k = min(t + 1, 119)
    err = np.abs(d.qpos[s2s.qadr] - traj["dof_pos"][k].numpy())
    zerr = d.qpos[2] - traj["root_pos"][k, 2].item()
    if t % 3 == 0:
        print(f"{t/30:5.2f} {np.median(err):8.3f} {err.max():8.3f} "
              f"{d.qpos[2]:6.3f} {zerr:6.3f}")
    if d.qpos[2] < 0.30:
        print(f"FELL {t/30:.2f}s")
        break
