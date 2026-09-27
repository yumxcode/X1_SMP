"""Positive controls for the sim2sim harness (review item 5).

A) STANDING hold: q_tar = standing pose, zero root velocity. A sound
   harness/PD/contact model must hold this indefinitely. If it falls,
   the harness (not the engine gap) is broken.
B) Slow-clip open-loop replay: slowest retargeted clip (ts=2.78) as
   q_tar via the Isaac-aligned implicit servo. Running references have
   no open-loop self-stability, but the SLOWEST clip is the least
   demanding; survival time there bounds what any policy could do.
"""
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
from aligned_actuator_test import retrofit


def standing_hold(sim, seconds=10.0):
    m, d = sim.m, sim.d
    ctrl = 1.0 / 30.0
    steps = int(round(ctrl / m.opt.timestep))
    # standing pose from the env init_pose (legs)
    home = np.zeros(29)
    home[17:23] = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
    home[23:29] = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]
    sim.mj.mj_resetData(m, d)
    d.qpos[:3] = [0, 0, 0.6141]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[sim.qadr] = home
    mujoco.mj_forward(m, d)
    t_fall = None
    t = 0.0
    while t < seconds:
        d.ctrl[:29] = home
        for _ in range(steps):
            mujoco.mj_step(m, d)
        t += ctrl
        if d.qpos[2] < 0.30:
            t_fall = t
            break
    return t_fall, d.qpos[2]


def slow_clip_replay(sim, pkl_path, seconds=10.0):
    m, d = sim.m, sim.d
    clip = pickle.load(open(pkl_path, "rb"))
    F = np.array(clip["frames"])
    fps_clip = clip["fps"] * clip.get("time_scale", 1.0)
    ctrl = 1.0 / 30.0
    steps = int(round(ctrl / m.opt.timestep))
    t0 = 100
    sim.mj.mj_resetData(m, d)
    f = F[t0]
    d.qpos[:3] = f[0:3]
    em = np.asarray(f[3:6])
    ang = np.linalg.norm(em)
    d.qpos[3:7] = [1, 0, 0, 0] if ang < 1e-8 else \
        [np.cos(ang / 2), *(em / ang * np.sin(ang / 2))]
    d.qpos[sim.qadr] = f[6:35]
    d.qvel[:] = 0  # quasi-static start: no initial-velocity mismatch
    mujoco.mj_forward(m, d)
    src_t = float(t0)
    t_fall = None
    t = 0.0
    while t < seconds:
        src_t += fps_clip * ctrl
        fi = min(int(round(src_t)), len(F) - 2)
        d.ctrl[:29] = np.clip(F[fi][6:35], -sim.a_bound, sim.a_bound)
        for _ in range(steps):
            mujoco.mj_step(m, d)
        t += ctrl
        if d.qpos[2] < 0.30:
            t_fall = t
            break
    return t_fall, d.qpos[2]


if __name__ == "__main__":
    import json
    pol = SV.Policy(str(REPO / "output/remote_ckpt/smp_v2_policy_final.pt"))
    sim = SV.Sim2Sim(pol)
    retrofit(sim)  # Isaac-aligned implicit servo

    tf_s, z_s = standing_hold(sim)
    print(f"[A] STANDING HOLD (aligned servo): "
          f"{'STABLE 10s' if tf_s is None else f'FELL at {tf_s:.2f}s'} "
          f"(final z={z_s:.3f})")

    clip = REPO / "data/motions/x1_v2/x1_run2_subject4_seg0.pkl"
    tf_c, z_c = slow_clip_replay(sim, clip)
    print(f"[B] SLOWEST-CLIP REPLAY (ts=2.78, quasi-static start, aligned "
          f"servo): {'STABLE 10s' if tf_c is None else f'FELL at {tf_c:.2f}s'} "
          f"(final z={z_c:.3f})")

    Path(REPO / "output/positive_control_v2.json").write_text(json.dumps({
        "standing_hold_fall_s": tf_s, "standing_final_z": float(z_s),
        "slow_clip_fall_s": tf_c, "slow_clip_final_z": float(z_c),
        "slow_clip": str(clip),
    }, indent=2))
    print("saved output/positive_control_v2.json")
