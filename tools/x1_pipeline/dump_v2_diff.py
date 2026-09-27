"""One-to-one replay diff against the v2-sourced Isaac dump (review item 4).

Uses output/remote_ckpt/isaac_traj_v2.pt (TASK_20260927_016, 2026-09-27
09:41) — the SAME policy (smp_v2_policy_final) and SAME env as training.
Steps MuJoCo with the dumped action sequence and compares, per control
step: obs channels, q_tar, and the state trajectory.
"""
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

kp = traj["kp"].numpy()
kd = traj["kd"].numpy()
print(f"[diff] engine gains kp[:6]={np.round(kp[:6],1)} "
      f"kd[:6]={np.round(kd[:6],1)}")
print(f"[diff] sim.kp matches: {np.allclose(kp, sim.kp)}, "
      f"kd matches: {np.allclose(kd, sim.kd)}")

sim.mj.mj_resetData(m, d)
d.qpos[:3] = traj["root_pos"][0].numpy()
q_ = traj["root_quat"][0].numpy()
d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
d.qpos[sim.qadr] = traj["dof_pos"][0].numpy()
d.qvel[:3] = traj["root_vel"][0].numpy()
d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
d.qvel[sim.vadr] = traj["dof_vel"][0].numpy()
sim.mj.mj_forward(m, d)

n = len(traj["obs"])
ctrl = 1.0 / 30.0
steps = int(round(ctrl / m.opt.timestep))
segs = [(0, 1, "root_h"), (1, 7, "rot"), (7, 10, "vel"), (10, 13, "angvel"),
        (13, 71, "joint_rot"), (71, 100, "dof_vel"), (100, 115, "key")]
print(f"[diff] t=0 obs maxdiff: "
      f"{np.abs(sim.obs() - traj['obs'][0].numpy()).max():.6f}")

rows = []
for t in range(n):
    o_mine = sim.obs()
    diff = np.abs(o_mine - traj["obs"][t].numpy())
    rows.append((t, diff.max(),
                 float(d.qpos[2] - traj["root_pos"][t][2]),
                 float(traj["root_pos"][t][2])))
    a = traj["action"][t].numpy()
    q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
    for _ in range(steps):
        q = d.qpos[sim.qadr]
        tau = np.clip(sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                      -sim.eff, sim.eff)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

print(f"{'t':>3} {'obsmax':>8} {'dz(m)':>8} {'isaac_z':>7}")
for t, om, dz, iz in rows:
    if t % 10 == 0 or t < 5 or om > 1.0:
        print(f"{t:3d} {om:8.3f} {dz:+8.3f} {iz:7.3f}")
dzs = np.array([r[2] for r in rows])
oms = np.array([r[1] for r in rows])
print(f"[diff] root_z divergence: |dz| med {np.median(np.abs(dzs)):.3f} "
      f"max {np.abs(dzs).max():.3f} @ {np.argmax(np.abs(dzs))}")
print(f"[diff] obs maxdiff: med {np.median(oms):.3f}")
# how far does the REPLAY itself stay up?
sim2 = SV.Sim2Sim(pol)
d2, m2 = sim2.d, sim2.m
sim2.mj.mj_resetData(m2, d2)
d2.qpos[:3] = traj["root_pos"][0].numpy()
q_ = traj["root_quat"][0].numpy()
d2.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
d2.qpos[sim2.qadr] = traj["dof_pos"][0].numpy()
d2.qvel[:3] = traj["root_vel"][0].numpy()
d2.qvel[3:6] = traj["root_ang_vel"][0].numpy()
d2.qvel[sim2.vadr] = traj["dof_vel"][0].numpy()
sim2.mj.mj_forward(m2, d2)
fell = None
for t in range(n):
    a = traj["action"][t].numpy()
    q_tar = np.clip(a, -sim2.a_bound, sim2.a_bound)
    for _ in range(steps):
        q = d2.qpos[sim2.qadr]
        tau = np.clip(sim2.kp * (q_tar - q) - sim2.kd * d2.qvel[sim2.vadr],
                      -sim2.eff, sim2.eff)
        d2.ctrl[:] = tau
        mujoco.mj_step(m2, d2)
    if d2.qpos[2] < 0.30:
        fell = t * ctrl
        break
print(f"[diff] open-loop replay (dump actions): "
      f"{'4s stable' if fell is None else f'fell at {fell:.2f}s'}")
