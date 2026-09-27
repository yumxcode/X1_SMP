"""Forward-match drive scale on the contact-free first control step.

Grid over (kp_scale, kd_scale): replay ONE control step from the dump's
exact (q0, qd0, q_tar0) in MuJoCo, compare resulting qd1 vs Isaac's qd1.
The best scale pair tells us the effective drive the policy trained under.
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

t = torch.load(REPO / "output/remote_ckpt/isaac_traj_v3_fixed.pt",
               map_location="cpu", weights_only=False)
pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v3b_it1800.pt"))
sim = SV.Sim2Sim(pol)
m, d = sim.m, sim.d
dt = 1.0 / 30.0
steps = int(round(dt / m.opt.timestep))

q0 = t["dof_pos"][0].numpy()
qd0 = t["dof_vel"][0].numpy()
q_tar = np.clip(t["action"][0].numpy(), -sim.a_bound, sim.a_bound)
qd1_isaac = t["dof_vel"][1].numpy()

def first_step(kp_s, kd_s):
    sim.mj.mj_resetData(m, d)
    d.qpos[:3] = t["root_pos"][0].numpy()
    q_ = t["root_quat"][0].numpy()
    d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
    d.qpos[sim.qadr] = q0
    d.qvel[:3] = t["root_vel"][0].numpy()
    d.qvel[3:6] = t["root_ang_vel"][0].numpy()
    d.qvel[sim.vadr] = qd0
    kp = sim.kp * kp_s
    kd = sim.kd * kd_s
    m.dof_damping[sim.vadr] = kd      # implicit damping follows the scale
    sim.mj.mj_forward(m, d)
    for _ in range(steps):
        q = d.qpos[sim.qadr]
        tau = np.clip(kp * (q_tar - q) - kd * d.qvel[sim.vadr],
                      -sim.eff, sim.eff)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    m.dof_damping[sim.vadr] = sim.kd  # restore
    return d.qvel[sim.vadr]

print(f"{'kp_s':>6s} {'kd_s':>6s} {'max|qd1 diff|':>13s} {'med':>7s} {'rms':>7s}")
best = None
for kp_s in (1.0, 0.7, 0.5, 0.35, 0.26, 0.18, 0.12, 0.08, 0.05):
    for kd_s in (1.0, 0.7, 0.5, 0.3, 0.1):
        qd1 = first_step(kp_s, kd_s)
        err = np.abs(qd1 - qd1_isaac)
        rms = float(np.sqrt((err ** 2).mean()))
        print(f"{kp_s:6.2f} {kd_s:6.2f} {err.max():13.3f} "
              f"{np.median(err):7.3f} {rms:7.3f}", flush=True)
        if best is None or rms < best[0]:
            best = (rms, kp_s, kd_s)
print(f"\nBEST: kp_scale {best[1]:.2f} kd_scale {best[2]:.2f} rms {best[0]:.3f}")
