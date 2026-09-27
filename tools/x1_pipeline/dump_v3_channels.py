"""Per-channel first-step divergence attribution for the v3 dump.

At t=1 (after one 33ms control step), decompose the obs diff by segment
and directly compare dof_vel between MuJoCo replay and Isaac dump.
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

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj_v3.pt",
                  map_location="cpu", weights_only=False)
pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v3_it2200.pt"))
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

ctrl = 1.0 / 30.0
steps = int(round(ctrl / m.opt.timestep))

segs = [(0, 1, "root_h"), (1, 7, "rot"), (7, 10, "vel"), (10, 13, "angvel"),
        (13, 71, "joint_rot"), (71, 100, "dof_vel"), (100, 115, "key")]

a = traj["action"][0].numpy()
q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
for _ in range(steps):
    q = d.qpos[sim.qadr]
    tau = np.clip(sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                  -sim.eff, sim.eff)
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)

print("== t=1 per-segment obs diff ==")
obs_diff = np.abs(sim.obs() - traj["obs"][1].numpy())
for lo, hi, name in segs:
    print(f"  {name:10s} max {obs_diff[lo:hi].max():10.5f}")

print("\n== t=1 dof_vel direct diff (MuJoCo replay vs Isaac dump) ==")
dv_mine = d.qvel[sim.vadr]
dv_isaac = traj["dof_vel"][1].numpy()
dd = np.abs(dv_mine - dv_isaac)

# joint names from the actuators
names = [m.actuator(i).name.replace("motor_", "").replace("_joint", "")
         for i in range(m.nu)]
order = np.argsort(-dd)
for k in order[:8]:
    print(f"  {names[k]:28s} mujoco {dv_mine[k]:+8.3f} isaac {dv_isaac[k]:+8.3f}"
          f" diff {dd[k]:7.3f} rad/s")
print(f"  overall dof_vel diff: max {dd.max():.3f} med {np.median(dd):.4f}")

print("\n== contact state at t=1 (MuJoCo) ==")
print(f"  ncon {d.ncon}")
for c in range(min(d.ncon, 6)):
    con = d.contact[c]
    g1 = m.geom(con.geom1).name or con.geom1
    g2 = m.geom(con.geom2).name or con.geom2
    print(f"  {g1} <-> {g2} dist {con.dist * 1000:.1f}mm")

print("\n== reset-frame sole penetration (this dump's seed) ==")
zmin = np.inf
for side in ("left", "right"):
    g = m.geom(f"{side}_ankle_roll_link_sole").id
    R = d.geom_xmat[g].reshape(3, 3)
    h = m.geom_size[g]
    c0 = d.geom_xpos[g]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                zmin = min(zmin, (c0 + R @ np.array(
                    [sx*h[0], sy*h[1], sz*h[2]]))[2])
# (state has stepped once; recompute at t=0 would be cleaner but the dump
# start penetration was audited separately: v3 data min sole z = +0.5mm)
