"""Emulate PhysX contact_offset=0.02 in MuJoCo and re-run the v3 replay diff.

Isaac engine sets physx.contact_offset = 0.02 (engines/isaac_gym_engine.py
line 704): contact forces act up to 2 cm BEFORE touching. The v3 retarget
data deliberately keeps soles at +0.5 mm (zero penetration), i.e. the whole
stance phase lives inside PhysX's phantom-contact band. MuJoCo's default
margin=0 produces no force until penetration -> the foot drops the last
2 cm and the trajectory diverges from the very first control step.

This test sets margin=0.02 (gap=0) on the robot geoms + floor and re-runs
the one-to-one replay of the v3 Isaac dump: first-step dof_vel diff and
open-loop 4 s survival are the readouts.
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

MARGIN = 0.02

traj = torch.load(REPO / "output/remote_ckpt/isaac_traj_v3.pt",
                  map_location="cpu", weights_only=False)
pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v3_it2200.pt"))
sim = SV.Sim2Sim(pol)
m, d = sim.m, sim.d

# apply PhysX-like contact margin to every collidable geom (robot + floor)
n_geom = 0
for g in range(m.ngeom):
    if m.geom_contype[g] or m.geom_conaffinity[g]:
        m.geom_margin[g] = MARGIN
        n_geom += 1
print(f"[margin-test] margin={MARGIN} applied to {n_geom} geoms")

sim.mj.mj_resetData(m, d)
d.qpos[:3] = traj["root_pos"][0].numpy()
q_ = traj["root_quat"][0].numpy()
d.qpos[3:7] = [q_[3], q_[0], q_[1], q_[2]]
d.qpos[sim.qadr] = traj["dof_pos"][0].numpy()
d.qvel[:3] = traj["root_vel"][0].numpy()
d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
d.qvel[sim.vadr] = traj["dof_vel"][0].numpy()
sim.mj.mj_forward(m, d)
print(f"[margin-test] t=0 ncon = {d.ncon}")

ctrl = 1.0 / 30.0
steps = int(round(ctrl / m.opt.timestep))
print(f"[margin-test] t=0 obs maxdiff: "
      f"{np.abs(sim.obs() - traj['obs'][0].numpy()).max():.6f}")

rows = []
n = len(traj["obs"])
fell = None
for t in range(n):
    o_mine = sim.obs()
    rows.append((t, np.abs(o_mine - traj["obs"][t].numpy()).max(),
                 float(d.qpos[2] - traj["root_pos"][t][2])))
    a = traj["action"][t].numpy()
    q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
    for _ in range(steps):
        q = d.qpos[sim.qadr]
        tau = np.clip(sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                      -sim.eff, sim.eff)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    if t == 0:
        dv = d.qvel[sim.vadr]
        dvi = traj["dof_vel"][1].numpy()
        dd = np.abs(dv - dvi)
        names = [m.actuator(i).name.replace("motor_", "")
                 for i in range(m.nu)]
        print("[margin-test] t=1 dof_vel diff: "
              f"max {dd.max():.3f} med {np.median(dd):.4f}")
        for k in np.argsort(-dd)[:5]:
            print(f"    {names[k]:32s} mujoco {dv[k]:+7.3f} "
                  f"isaac {dvi[k]:+7.3f} diff {dd[k]:6.3f}")
        print(f"[margin-test] t=1 ncon = {d.ncon}")
    if fell is None and d.qpos[2] < 0.30:
        fell = t * ctrl

dzs = np.array([r[2] for r in rows])
oms = np.array([r[1] for r in rows])
print(f"[margin-test] root_z divergence: |dz| med {np.median(np.abs(dzs)):.3f} "
      f"max {np.abs(dzs).max():.3f} @ {np.argmax(np.abs(dzs))}")
print(f"[margin-test] obs maxdiff: med {np.median(oms):.3f}")
print(f"[margin-test] open-loop replay: "
      f"{'4s stable' if fell is None else f'fell at {fell:.2f}s'}")
