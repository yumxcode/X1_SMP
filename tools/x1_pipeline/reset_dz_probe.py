"""Reset-penetration probe: does the v3 policy survive MuJoCo when reset
with soles actually PENETRATING (hard-contact regime) instead of hovering
in PhysX's 2cm contact-offset band?

Shifts the reset root z down by dz (mm) and runs the standard sim2sim
episode protocol (policy in the loop, NOT open-loop replay).
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import mujoco

import sim2sim_validate as SV


def run(ckpt, dz_mm, seeds=(0, 1, 2, 3), length=10.0):
    pol = SV.Policy(ckpt)
    sim = SV.Sim2Sim(pol)
    sim.env_yaml_rel = "data/envs/smp_x1_env_v3.yaml"
    # reload motions for v3 dataset
    import pickle, re
    env_yaml = REPO / "data/envs/smp_x1_env_v3.yaml"
    ds_line = [l for l in env_yaml.read_text().splitlines()
               if l.strip().startswith("motion_file:")][0]
    ds = REPO / re.search(r'"([^"]+)"', ds_line).group(1)
    motions = []
    for line in ds.read_text().splitlines():
        mm = re.search(r'file:\s*"([^"]+)"', line)
        if mm:
            motions.append(pickle.load(open(REPO / mm.group(1), "rb")))
    sim.motions = motions
    m, d = sim.m, sim.d
    ctrl = 1.0 / 30.0
    steps = int(round(ctrl / m.opt.timestep))
    out = []
    for seed in seeds:
        sim.reset(seed)
        d.qpos[2] -= dz_mm / 1000.0
        mujoco.mj_forward(m, d)
        log = dict(t=[], root_z=[], root_x=[], root_y=[], lf_z=[], rf_z=[],
                   lh_x=[], rh_x=[], lf_x=[], rf_x=[], pitch=[], roll=[],
                   q=[], torque=[], bad_contact=[])
        fell = None
        for it in range(int(length / ctrl)):
            o = sim.obs()
            a = pol.forward(o)
            q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
            for _ in range(steps):
                q = d.qpos[sim.qadr]
                tau = np.clip(sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                              -sim.eff, sim.eff)
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
            q = d.qpos[sim.qadr]
            log["torque"].append(np.abs(np.clip(
                sim.kp * (q_tar - q) - sim.kd * d.qvel[sim.vadr],
                -sim.eff, sim.eff)))
            log["bad_contact"].append(any(d.geom_xpos[g][2] < 0.03
                                          for g in sim.fall_gids))
            if fell is None and d.qpos[2] < 0.30:
                fell = it * ctrl
                break
        r = SV.analyze(log, length, sim.eff)
        out.append((seed, fell, r))
        print(f"  dz={dz_mm:+.0f}mm seed {seed}: "
              f"{'10s stable' if fell is None else f'fell {fell:.2f}s'} "
              f"PASS={r['PASS']}")
    return out


if __name__ == "__main__":
    ckpt = str(REPO / "output/remote_ckpt/smp_v3_it2200.pt")
    for dz in (0.0, -2.0, -4.0, -8.0):
        print(f"== dz {dz:+.0f} mm ==")
        run(ckpt, dz)
