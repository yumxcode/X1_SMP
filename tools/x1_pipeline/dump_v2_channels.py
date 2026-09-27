"""Per-channel first-steps diff against the v2 Isaac dump."""
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
segs = [(0, 1, "root_h"), (1, 7, "rot"), (7, 10, "vel"), (10, 13, "angvel"),
        (13, 71, "joint_rot"), (71, 100, "dof_vel"), (100, 115, "key")]
for t in range(4):
    o = sim.obs()
    diff = np.abs(o - traj["obs"][t].numpy())
    parts = " ".join(f"{nm}:{diff[a:b].max():.3f}" for a, b, nm in segs)
    print(f"t={t}: {parts}")
    if t < 3:
        a = traj["action"][t].numpy()
        q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
        for _ in range(steps):
            q = d.qpos[sim.qadr]
            tau = np.clip(sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                          -sim.eff, sim.eff)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        nxt = t + 1
        dv = np.abs(d.qvel[sim.vadr] - traj["dof_vel"][nxt].numpy()).max()
        print(f"   after step: dof_vel max diff {dv:.3f}")
        print(f"   mujoco root_angvel {np.round(d.qvel[3:6],2)} "
              f"isaac {np.round(traj['root_ang_vel'][nxt].numpy(),2)}")
        my_tau = np.clip(sim.kp * (q_tar - d.qpos[sim.qadr])
                         - sim.kd * d.qvel[sim.vadr], -sim.eff, sim.eff)
        print(f"   tau diff vs dump: "
              f"{np.abs(my_tau - traj['torque'][nxt].numpy()).max():.2f}")
