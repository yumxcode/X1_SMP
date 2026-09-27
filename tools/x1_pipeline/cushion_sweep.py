"""Sweep MuJoCo contact parameters to emulate PhysX's 2cm contact-offset
cushion, scored by open-loop replay of the v3 Isaac dump.

Metrics per config: open-loop replay fall time (baseline 0.63s, target
4s+) and first-step dof_vel diff vs Isaac (baseline 2.2 rad/s).
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


def replay(margin, solimp, solref, condim=4):
    sim = SV.Sim2Sim(pol)
    m, d = sim.m, sim.d
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            m.geom_margin[g] = margin
            m.geom_solimp[g] = solimp
            m.geom_solref[g] = solref
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
    fell = None
    fs_diff = None
    n = len(traj["obs"])
    for t in range(n):
        a = traj["action"][t].numpy()
        q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
        for _ in range(steps):
            q = d.qpos[sim.qadr]
            tau = np.clip(sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                          -sim.eff, sim.eff)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        if t == 0:
            fs_diff = float(np.abs(d.qvel[sim.vadr]
                                   - traj["dof_vel"][1].numpy()).max())
        if fell is None and d.qpos[2] < 0.30:
            fell = t * ctrl
            break
    return (4.0 if fell is None else fell), fs_diff


CONFIGS = [
    ("default solimp", 0.02, (0.9, 0.95, 0.001, 0.5, 2.0), (0.02, 1.0)),
    ("stiff  d0.99 w5mm", 0.02, (0.99, 0.9999, 0.005, 0.5, 2.0), (0.02, 1.0)),
    ("stiff  d0.99 w2mm", 0.02, (0.99, 0.9999, 0.002, 0.5, 2.0), (0.02, 1.0)),
    ("stiff  d0.95 w2mm", 0.02, (0.95, 0.99, 0.002, 0.5, 2.0), (0.02, 1.0)),
    ("stiff  d0.95 w1mm", 0.02, (0.95, 0.99, 0.001, 0.5, 2.0), (0.02, 1.0)),
    ("stiff+fast ref", 0.02, (0.99, 0.9999, 0.002, 0.5, 2.0), (0.005, 1.0)),
    ("stiff+med ref", 0.02, (0.99, 0.9999, 0.002, 0.5, 2.0), (0.01, 1.0)),
    ("soft   d0.8 w10mm", 0.02, (0.8, 0.9, 0.01, 0.5, 2.0), (0.02, 1.0)),
    ("very soft d0.5", 0.02, (0.5, 0.8, 0.02, 0.5, 2.0), (0.04, 1.0)),
    ("no margin", 0.0, (0.9, 0.95, 0.001, 0.5, 2.0), (0.02, 1.0)),
]

print(f"{'config':22s} {'replay_fall':>12s} {'1st-step dv diff':>16s}")
for name, margin, solimp, solref in CONFIGS:
    ft, fs = replay(margin, solimp, solref)
    print(f"{name:22s} {ft:10.2f}s {fs:14.3f} rad/s", flush=True)
