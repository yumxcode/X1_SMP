"""Replay-by-torque discrimination using the v3b force dump.

1. Is Isaac's dof_force sensor = PD model torque? (sign fit on arms)
2. MuJoCo open-loop replay applying the sensor forces as ctrl — if the
   trajectory matches Isaac, the drive semantics was the whole story.
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

t = torch.load(REPO / "output/remote_ckpt/isaac_traj_v3b.pt",
               map_location="cpu", weights_only=False)
kp = t["kp"].numpy(); kd = t["kd"].numpy(); tlim = t["tlim"].numpy()
q = t["dof_pos"].numpy(); qd = t["dof_vel"].numpy()
qt = t["q_tar"].numpy(); f = t["dof_force"].numpy()

pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v3_it2200.pt"))
sim = SV.Sim2Sim(pol)
m = sim.m
names = [m.actuator(i).name.replace("motor_", "") for i in range(m.nu)]

pd_model = np.clip(kp[None] * (qt - q) - kd[None] * qd, -tlim[None], tlim[None])
# note: q_tar[t] recorded AFTER step t; align: pd at step t uses q[t], qd[t],
# qt[t-1]. Use shifted q_tar.
qt_shift = np.vstack([qt[0][None], qt[:-1]])
pd_shift = np.clip(kp[None] * (qt_shift - q) - kd[None] * qd,
                   -tlim[None], tlim[None])

for sign in (+1.0, -1.0):
    err = np.abs(sign * f - pd_shift)
    print(f"sign {sign:+.0f}: |sensor*sign - PDmodel| med {np.median(err):.2f} "
          f"p95 {np.quantile(err, 0.95):.2f}")

# per-joint fit (sign -1 assumed = reaction), first 20 steps, no contact
err = np.abs(-f[:20] - pd_shift[:20])
print("\nper-joint |sensor+PD| first 20 steps (worst 8):")
for k in np.argsort(-err.mean(axis=0))[:8]:
    print(f"  {names[k]:32s} mean {err[:, k].mean():7.2f}  "
          f"sensor[1] {f[1, k]:+8.2f}  PD[1] {pd_shift[1, k]:+8.2f}")

# replay-by-torque in MuJoCo
sim2 = SV.Sim2Sim(pol)
m2, d2 = sim2.m, sim2.d
sim2.mj.mj_resetData(m2, d2)
d2.qpos[:3] = t["root_pos"][0].numpy()
q_ = t["root_quat"][0].numpy()
d2.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
d2.qpos[sim2.qadr] = q[0]
d2.qvel[:3] = t["root_vel"][0].numpy()
d2.qvel[3:6] = t["root_ang_vel"][0].numpy()
d2.qvel[sim2.vadr] = qd[0]
sim2.mj.mj_forward(m2, d2)
ctrl = 1.0 / 30.0
steps = int(round(ctrl / m2.opt.timestep))

for sign, tag in ((-1.0, "ctrl=-sensor"), (+1.0, "ctrl=+sensor")):
    sim2.mj.mj_resetData(m2, d2)
    d2.qpos[:3] = t["root_pos"][0].numpy()
    d2.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
    d2.qpos[sim2.qadr] = q[0]
    d2.qvel[:3] = t["root_vel"][0].numpy()
    d2.qvel[3:6] = t["root_ang_vel"][0].numpy()
    d2.qvel[sim2.vadr] = qd[0]
    sim2.mj.mj_forward(m2, d2)
    fs = None
    fell = None
    dzs = []
    for tt in range(120):
        d2.ctrl[:] = np.clip(sign * f[tt], -sim2.eff, sim2.eff)
        for _ in range(steps):
            mujoco.mj_step(m2, d2)
        if tt == 0:
            fs = float(np.abs(d2.qvel[sim2.vadr] - qd[1]).max())
        dzs.append(float(d2.qpos[2] - t["root_pos"][tt][2]))
        if fell is None and d2.qpos[2] < 0.30:
            fell = tt * ctrl
            break
    dzs = np.array(dzs)
    print(f"\n[{tag}] 1st-step dv diff {fs:.3f} | root dz med "
          f"{np.median(np.abs(dzs[:30])):.3f} max {np.abs(dzs[:30]).max():.3f} "
          f"| replay {'stable 4s' if fell is None else f'fell {fell:.2f}s'}")
