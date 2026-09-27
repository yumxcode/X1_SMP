"""DECISIVE alignment test: pd_explicit Isaac dump vs MuJoCo harness replay.

pd_explicit semantics (tau=clip(kp*(tar-q)-kd*qd,+-tlim) per substep,
DOF_MODE_EFFORT) should be IDENTICAL to the MuJoCo sim2sim harness. The
first-control-step dof_vel diff must collapse from the pos-mode 3.3 rad/s
to near-zero — independent of policy quality (this dump's policy is weak:
Isaac itself staggers, but the DYNAMICS should match step by step).
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

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj_pdx2.pt",
                  map_location="cpu", weights_only=False)
pol = SV.Policy(str(REPO / "output/remote_ckpt/smppdx_it400.pt"))
sim = SV.Sim2Sim(pol)
d, m = sim.d, sim.m

kp = traj["kp"].numpy(); kd = traj["kd"].numpy()
print(f"[diff-pdx2] gains match: kp {np.allclose(kp, sim.kp)}, "
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
print(f"[diff-pdx2] reset min sole z {zmin*1000:+.1f} mm | ncon {d.ncon} | "
      f"t=0 obs maxdiff "
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

dv_mine = d.qvel[sim.vadr]
dv_isaac = traj["dof_vel"][1].numpy()
dd = np.abs(dv_mine - dv_isaac)
names = [m.actuator(i).name.replace("motor_", "") for i in range(m.nu)]
print(f"[diff-pdx2] t=1 dof_vel diff: max {dd.max():.4f} med {np.median(dd):.5f}")
for k in np.argsort(-dd)[:5]:
    print(f"  {names[k]:32s} mujoco {dv_mine[k]:+8.4f} isaac {dv_isaac[k]:+8.4f}"
          f" diff {dd[k]:7.4f}")

# longer-horizon replay: per-step max |dof_vel diff| for 30 steps
sim.mj.mj_resetData(m, d)
d.qpos[:3] = traj["root_pos"][0].numpy()
d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
d.qpos[sim.qadr] = traj["dof_pos"][0].numpy()
d.qvel[:3] = traj["root_vel"][0].numpy()
d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
d.qvel[sim.vadr] = traj["dof_vel"][0].numpy()
sim.mj.mj_forward(m, d)
print("\n[diff-pdx2] 30-step replay per-step max|dof_vel diff|:")
for t in range(30):
    a_t = traj["action"][t].numpy()
    q_tar_t = np.clip(a_t, -sim.a_bound, sim.a_bound)
    for _ in range(steps):
        q = d.qpos[sim.qadr]
        tau = np.clip(sim.kp * (q_tar_t - q) - sim.kd * d.qvel[sim.vadr],
                      -sim.eff, sim.eff)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    dd_t = np.abs(d.qvel[sim.vadr] - traj["dof_vel"][t + 1].numpy())
    dz = d.qpos[2] - traj["root_pos"][t + 1][2]
    if t < 8 or t % 10 == 0:
        print(f"  t={t+1:3d}: max|dv diff| {dd_t.max():8.4f} med {np.median(dd_t):8.5f} "
              f"dz {dz:+.4f} m")
