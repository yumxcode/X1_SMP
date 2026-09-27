"""Replay with the dumped q_tar directly (eliminates the last action-clip
semantics difference)."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import mujoco

import sim2sim_validate as SV

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj_v2.pt",
                  map_location="cpu", weights_only=False)
pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v2_policy_final.pt"))
sim = SV.Sim2Sim(pol)
d, m = sim.d, sim.m
sim.mj.mj_resetData(m, d)
d.qpos[:3] = traj["root_pos"][0].numpy()
q_ = traj["root_quat"][0].numpy()
d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
d.qpos[sim.qadr] = traj["dof_pos"][0].numpy()
d.qvel[:3] = traj["root_vel"][0].numpy()
d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
d.qvel[sim.vadr] = traj["dof_vel"][0].numpy()
sim.mj.mj_forward(m, d)
steps = int(round((1.0 / 30.0) / m.opt.timestep))
dv1 = None
fell = None
for t in range(120):
    if t == 1:
        dv1 = np.abs(d.qvel[sim.vadr] - traj["dof_vel"][1].numpy()).max()
    q_tar = traj["q_tar"][t].numpy()
    for _ in range(steps):
        q = d.qpos[sim.qadr]
        tau = np.clip(sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                      -sim.eff, sim.eff)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    if d.qpos[2] < 0.30 and fell is None:
        fell = t / 30.0
        break
print(f"[q_tar replay] step-1 dof_vel diff: {dv1:.3f} | "
      f"replay: {'fell %.2fs' % fell if fell else '4s stable'}")
