"""Same-generation first-step diff: FIXED-asset Isaac dump (v3b it1800)
replayed in the FIXED-asset MuJoCo harness.

Readouts: t=0 obs diff (sanity), t=1 per-segment obs diff, t=1 dof_vel
diff (the number that decides hypothesis 3: still ~3 rad/s => engine
semantics divergence on clean resets; < 1 rad/s => look elsewhere).
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

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj_v3_fixed.pt",
                  map_location="cpu", weights_only=False)
pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v3b_it1800.pt"))
sim = SV.Sim2Sim(pol)
d, m = sim.d, sim.m

kp = traj["kp"].numpy()
kd = traj["kd"].numpy()
print(f"[diff3b] gains match: kp {np.allclose(kp, sim.kp)}, "
      f"kd {np.allclose(kd, sim.kd)}")

sim.mj.mj_resetData(m, d)
d.qpos[:3] = traj["root_pos"][0].numpy()
q_ = traj["root_quat"][0].numpy()
d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
d.qpos[sim.qadr] = traj["dof_pos"][0].numpy()
d.qvel[:3] = traj["root_vel"][0].numpy()
d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
d.qvel[sim.vadr] = traj["dof_vel"][0].numpy()
sim.mj.mj_forward(m, d)

# reset-frame sole penetration in the FIXED asset
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
                    [sx * h[0], sy * h[1], sz * h[2]]))[2])
print(f"[diff3b] reset-frame min sole z: {zmin * 1000:+.1f} mm "
      f"| ncon {d.ncon}")
print(f"[diff3b] t=0 obs maxdiff: "
      f"{np.abs(sim.obs() - traj['obs'][0].numpy()).max():.6f}")

ctrl = 1.0 / 30.0
steps = int(round(ctrl / m.opt.timestep))
a = traj["action"][0].numpy()
q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
for _ in range(steps):
    q = d.qpos[sim.qadr]
    tau = np.clip(sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                  -sim.eff, sim.eff)
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)

segs = [(0, 1, "root_h"), (1, 7, "rot"), (7, 10, "vel"), (10, 13, "angvel"),
        (13, 71, "joint_rot"), (71, 100, "dof_vel"), (100, 115, "key")]
obs_diff = np.abs(sim.obs() - traj["obs"][1].numpy())
print("[diff3b] t=1 per-segment obs diff:")
for lo, hi, name in segs:
    print(f"  {name:10s} max {obs_diff[lo:hi].max():10.5f}")

dv_mine = d.qvel[sim.vadr]
dv_isaac = traj["dof_vel"][1].numpy()
dd = np.abs(dv_mine - dv_isaac)
names = [m.actuator(i).name.replace("motor_", "") for i in range(m.nu)]
print(f"[diff3b] t=1 dof_vel diff: max {dd.max():.3f} med {np.median(dd):.4f}")
for k in np.argsort(-dd)[:6]:
    print(f"  {names[k]:32s} mujoco {dv_mine[k]:+8.3f} isaac {dv_isaac[k]:+8.3f}"
          f" diff {dd[k]:6.3f}")
print(f"[diff3b] t=1 ncon (MuJoCo) = {d.ncon}")
