"""POSITIVE CONTROL for the sim2sim harness: replay the retargeted
reference clip as q_tar (no policy). If the harness/PD/contact model is
sound, the robot tracks and stays up -> proves the FAIL verdicts come
from the policy, not the harness."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import pickle

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

import sim2sim_validate as SV

pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v2_policy_it1900.pt"))  # unused
sim = SV.Sim2Sim(pol)
m, d = sim.m, sim.d

clip = pickle.load(open(REPO / "data/motions/x1_v2/x1_run2_subject1_seg2.pkl", "rb"))
F = np.array(clip["frames"])
fps_clip = clip["fps"] * clip.get("time_scale", 1.0)

ctrl = 1.0 / 30.0
steps = int(round(ctrl / m.opt.timestep))
dur = 10.0
n_ctrl = int(dur / ctrl)
t0 = 50

sim.reset(0)
f = F[t0]
d.qpos[:3] = f[0:3]
em = np.asarray(f[3:6])
ang = np.linalg.norm(em)
if ang < 1e-8:
    d.qpos[3:7] = [1, 0, 0, 0]
else:
    d.qpos[3:7] = [np.cos(ang / 2), *(em / ang * np.sin(ang / 2))]
d.qpos[sim.qadr] = f[6:35]
f0, f1 = F[t0 - 1], F[t0 + 1]
dtc = 1.0 / clip["fps"]
d.qvel[:3] = (f1[0:3] - f0[0:3]) / (2 * dtc)
R = Rot.from_rotvec(em).as_matrix()
d.qvel[3:6] = R.T @ ((f1[3:6] - f0[3:6]) / (2 * dtc))
d.qvel[sim.vadr] = (f1[6:35] - f0[6:35]) / (2 * dtc)
mujoco.mj_forward(m, d)

log = dict(t=[], root_z=[], root_x=[], root_y=[], lf_z=[], rf_z=[],
           lh_x=[], rh_x=[], lf_x=[], rf_x=[], pitch=[], roll=[],
           q=[], torque=[], bad_contact=[])
src_t = float(t0)
for it in range(n_ctrl):
    src_t += fps_clip * ctrl
    fi = min(int(round(src_t)), len(F) - 2)
    a = F[fi][6:35]
    q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
    for _ in range(steps):
        q = d.qpos[sim.qadr]
        tau = np.clip(sim.kp * (q_tar - q), -sim.eff, sim.eff)
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    Rb = d.xmat[sim.torso_bid].reshape(3, 3)
    up = Rb[:, 2]
    log["t"].append(it * ctrl)
    log["root_z"].append(d.qpos[2])
    log["root_x"].append(d.qpos[0])
    log["root_y"].append(d.qpos[1])
    log["lf_z"].append(d.site("x_lfoot").xpos[2])
    log["rf_z"].append(d.site("x_rfoot").xpos[2])
    log["lh_x"].append(d.site("x_lhand").xpos[0])
    log["rh_x"].append(d.site("x_rhand").xpos[0])
    log["lf_x"].append(d.site("x_lfoot").xpos[0])
    log["rf_x"].append(d.site("x_rfoot").xpos[0])
    log["pitch"].append(np.arctan2(-up[0], up[2]))
    log["roll"].append(np.arctan2(up[1], up[2]))
    log["q"].append(d.qpos[sim.qadr].copy())
    log["torque"].append(np.abs(np.clip(sim.kp * (q_tar - d.qpos[sim.qadr]),
                                        -sim.eff, sim.eff)))
    log["bad_contact"].append(any(d.geom_xpos[g][2] < 0.03
                                  for g in sim.fall_gids))
    if d.qpos[2] < 0.2 and it > 30:
        print(f"FELL at t={it * ctrl:.2f}")
        break

r = SV.analyze(log, dur, sim.eff)
print("POSITIVE CONTROL (reference replay as q_tar):")
print("  S1 no_fall:", r["S1_no_fall"])
print("  S2 gait   :", {k: (round(v, 2) if isinstance(v, float) else v)
                       for k, v in r["S2_gait"].items()})
print("  S3 morph  :", {k: (round(v, 2) if isinstance(v, float) else v)
                       for k, v in r["S3_morphology"].items()})
print("  S4 actuat:", {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in r["S4_actuation"].items()})
print("  PASS:", r["PASS"])
