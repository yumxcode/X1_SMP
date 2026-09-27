"""Replay diff with MuJoCo-only passive terms zeroed (armature,
frictionloss) — Isaac's PhysX MJCF importer is suspected to ignore them.
If the first-step dof_vel divergence vanishes, the engine gap is THESE
terms, not contact impulses."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import mujoco

import sim2sim_validate as SV


def replay(zero_armature, zero_friction, n_steps=6):
    traj = torch.load(REPO / "output/remote_ckpt/isaac_traj_v2.pt",
                      map_location="cpu", weights_only=False)
    pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v2_policy_final.pt"))
    sim = SV.Sim2Sim(pol)
    d, m = sim.d, sim.m
    if zero_armature:
        m.dof_armature[sim.vadr] = 0.0
    if zero_friction:
        m.dof_frictionloss[sim.vadr] = 0.0
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
    dv_max, t_fall = 0.0, None
    n_total = 120
    for t in range(n_total):
        if t < n_steps:
            dv = np.abs(d.qvel[sim.vadr]
                        - traj["dof_vel"][t].numpy()).max()
            dv_max = max(dv_max, dv)
        a = traj["action"][t].numpy()
        q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
        for _ in range(steps):
            q = d.qpos[sim.qadr]
            tau = np.clip(sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                          -sim.eff, sim.eff)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        if d.qpos[2] < 0.30 and t_fall is None:
            t_fall = t / 30.0
            break
    return dv_max, t_fall


if __name__ == "__main__":
    for za, zf, tag in ((False, False, "baseline"),
                        (True, False, "armature=0"),
                        (False, True, "frictionloss=0"),
                        (True, True, "both=0")):
        dv, tf = replay(za, zf)
        print(f"{tag:15s}: first-6-steps dof_vel diff max {dv:.3f} | "
              f"replay {'fell %.2fs' % tf if tf else '4s stable'}")
