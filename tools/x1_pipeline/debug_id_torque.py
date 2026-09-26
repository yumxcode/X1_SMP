"""Reverse-engineer Isaac's actual torque from the dumped trajectory:
tau_required = mj_inverse(state_t, qacc_t) vs PD formula variants."""
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

dt = 1.0 / 30.0
N = 100
tau_req = np.zeros((N, 29))
for t in range(N):
    q_ = traj["root_quat"][t].numpy()
    d.qpos[:3] = traj["root_pos"][t].numpy()
    d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
    d.qpos[s2s.qadr] = traj["dof_pos"][t].numpy()
    d.qvel[:3] = traj["root_vel"][t].numpy()
    d.qvel[3:6] = traj["root_ang_vel"][t].numpy()
    d.qvel[s2s.vadr] = traj["dof_vel"][t].numpy()
    qa = (traj["dof_vel"][t + 1].numpy() - traj["dof_vel"][t].numpy()) / dt
    d.qacc[:] = 0
    d.qacc[s2s.vadr] = qa
    d.qacc[:3] = ((traj["root_vel"][t + 1].numpy() -
                   traj["root_vel"][t].numpy()) / dt)
    s2s.mj.mj_inverse(m, d)
    tau_req[t] = d.qfrc_inverse[s2s.vadr]

# candidate formulas
kp, kd, eff = s2s.kp, s2s.kd, s2s.eff
tau_pd = np.zeros_like(tau_req)
for t in range(N):
    a = np.clip(traj["action"][t].numpy(), -s2s.a_bound, s2s.a_bound)
    q = traj["dof_pos"][t].numpy()
    qd = traj["dof_vel"][t].numpy()
    tau_pd[t] = kp * (a - q) - kd * qd

print(f"{'joint':>32} {'|req|max':>8} {'|pd|max':>8} {'corr':>6} "
      f"{'req/eff':>8} {'pd/eff':>8}")
for j in range(29):
    r = tau_req[:, j]
    p = tau_pd[:, j]
    c = np.corrcoef(r, p)[0, 1] if r.std() > 1e-6 and p.std() > 1e-6 else np.nan
    print(f"{X1_DOF_ORDER[j]:>32} {np.abs(r).max():8.1f} "
          f"{np.abs(p).max():8.1f} {c:6.2f} "
          f"{np.abs(r).max()/eff[j]:8.2f} {np.abs(p).max()/eff[j]:8.2f}")
