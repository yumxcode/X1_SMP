"""Right ankle roll deep-dive: my value vs Isaac's, contact state."""
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
m.dof_damping[s2s.vadr] = 0
m.opt.timestep = 1.0 / 1200.0

d.qpos[:3] = traj["root_pos"][0].numpy()
q_ = traj["root_quat"][0].numpy()
d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
d.qpos[s2s.qadr] = traj["dof_pos"][0].numpy()
d.qvel[:3] = traj["root_vel"][0].numpy()
d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
d.qvel[s2s.vadr] = traj["dof_vel"][0].numpy()
s2s.mj.mj_forward(m, d)

rar_mine = s2s.qadr[28]
lar_mine = s2s.qadr[27]
rsole = [g for g in range(m.ngeom)
         if (m.geom(g).name or "") == "right_ankle_roll_link_sole"][0]
print(f"{'t':>5} {'isaac_r':>8} {'mine_r':>8} {'isaac_l':>8} {'mine_l':>8} "
      f"{'r_tau':>6} {'r_qd':>6} {'ncon':>4}")
for t in range(45):
    a = traj["action"][t].numpy()
    q_tar = np.clip(a, -s2s.a_bound, s2s.a_bound)
    for _ in range(40):
        q = d.qpos[s2s.qadr]
        qd = d.qvel[s2s.vadr]
        tau = np.clip(s2s.kp * (q_tar - q) - s2s.kd * qd,
                      -s2s.eff, s2s.eff)
        d.ctrl[:] = tau
        s2s.mj.mj_step(m, d)
    if t % 2 == 0:
        print(f"{t/30:5.2f} {traj['dof_pos'][min(t+1,119),28]:8.3f} "
              f"{d.qpos[rar_mine]:8.3f} "
              f"{traj['dof_pos'][min(t+1,119),27]:8.3f} "
              f"{d.qpos[lar_mine]:8.3f} "
              f"{d.ctrl[28]:6.1f} {d.qvel[s2s.vadr][28]:6.2f} {d.ncon:4d}")
