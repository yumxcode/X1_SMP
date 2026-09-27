"""MuJoCo counterpart of the Isaac airborne step probe (TASK_20260928_009).

Lifts the fixed-asset X1 to z=+1m (contact-free), applies the same q_tar
steps (delta 0.8 on left_shoulder_pitch / right_hip_pitch /
right_knee_pitch) from the v3-dataset reset pose, 15 control steps.
Compares peak |qd| and final q-q0 with Isaac's numbers:
  Isaac: shoulder 6.94 / hip 6.43 / knee 6.67 rad/s, final deltas ~0.8.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

import sim2sim_validate as SV

pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v3b_it1800.pt"))
sim = SV.Sim2Sim(pol)
m, d = sim.m, sim.d

# v3-dataset reset pose (same sampling as the harness)
sim.reset(0)
q0 = d.qpos[sim.qadr].copy()
root0 = d.qpos[:3].copy()
quat0 = d.qpos[3:7].copy()
root0[2] = 1.58  # +1m airborne
DELTA = 0.8
PROBE = {"left_shoulder_pitch": 3, "right_hip_pitch": 23, "right_knee_pitch": 26}
q_tar = q0.copy()
q_tar[3] += DELTA
q_tar[23] -= DELTA
q_tar[26] += DELTA

ctrl = 1.0 / 30.0
steps = int(round(ctrl / m.opt.timestep))
q_log, qd_log = [], []
sim.mj.mj_resetData(m, d)
d.qpos[:3] = root0
d.qpos[3:7] = quat0
d.qpos[sim.qadr] = q0
sim.mj.mj_forward(m, d)
for t in range(15):
    for _ in range(steps):
        q = d.qpos[sim.qadr]
        tau = np.clip(sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                      -sim.eff, sim.eff)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    q_log.append(d.qpos[sim.qadr].copy())
    qd_log.append(d.qvel[sim.vadr].copy())
q = np.array(q_log); qd = np.array(qd_log)
isaac = {"left_shoulder_pitch": 6.94, "right_hip_pitch": 6.43,
         "right_knee_pitch": 6.67}
isaac_fin = {"left_shoulder_pitch": 0.799, "right_hip_pitch": -0.745,
             "right_knee_pitch": 0.780}
print(f"{'joint':22s} {'MJ peak|qd|':>11s} {'Isaac peak':>11s} "
      f"{'MJ final':>9s} {'Isaac fin':>9s}")
for nm, idx in PROBE.items():
    print(f"{nm:22s} {np.abs(qd[:, idx]).max():11.2f} {isaac[nm]:11.2f} "
          f"{q[-1, idx]-q0[idx]:+9.3f} {isaac_fin[nm]:+9.3f}")
print("\n(q0 differs from Isaac's random v3 frame; comparison is "
      "order-of-magnitude on the drive response)")
