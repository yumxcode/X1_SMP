"""Positive control: open-loop demo tracking in the MuJoCo sim2sim harness
with the v4 LOW-GAIN assets (kp explicit clip, kd implicit — harness
semantics, no policy involved).

Purpose: decouple 'v4 kp too weak to hold the robot' (asset design error)
from 'policy not yet trained' (training matter). We replay the SAME demo
motions used for resets as q_tar at 30 Hz and measure survival + root_z
tracking vs the original high-gain x1.xml.

Usage: python tools/x1_pipeline/v4_openloop_probe.py [--asset x1_v4.xml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))

import sim2sim_validate as SV


def probe(asset_xml, sim_xml, motion_idx=0, length_s=10.0):
    class Dummy:
        def forward(self, obs):
            return np.zeros(29)
    sim = SV.Sim2Sim(Dummy(), sim_xml=sim_xml, asset_xml=asset_xml)
    mo = sim.motions[motion_idx]
    F = np.asarray(mo["frames"])
    fps = mo["fps"]
    n_ctrl = int(length_s * 30)
    dt_ctrl = 1.0 / 30.0
    steps = int(round(dt_ctrl / sim.m.opt.timestep))

    # start from the demo's first frame (running start of the clip)
    sim.reset(seed=100 + motion_idx)
    i0 = 1
    log_z, log_ref, track_err = [], [], []
    fell_t = None
    for it in range(n_ctrl):
        # demo reference at this control tick (loop if clip ends)
        ti = (i0 + it * (fps / 30.0))
        ti_int = int(ti) % (len(F) - 1)
        f = F[ti_int]
        q_tar = np.asarray(f[6:35])
        q_tar = np.clip(q_tar, -sim.a_bound, sim.a_bound)
        for _ in range(steps):
            q = sim.d.qpos[sim.qadr]
            tau = np.clip(sim.kp * (q_tar - q), -sim.eff, sim.eff)
            sim.d.ctrl[:] = tau
            sim.mj.mj_step(sim.m, sim.d)
        log_z.append(float(sim.d.qpos[2]))
        log_ref.append(float(F[ti_int][2]))
        track_err.append(float(np.abs(sim.d.qpos[sim.qadr] - q_tar).mean()))
        if sim.d.qpos[2] < 0.30 and fell_t is None:
            fell_t = (it + 1) * dt_ctrl
            break
    z = np.array(log_z)
    tr = np.array(track_err)
    return dict(asset=Path(asset_xml).name, fell_t=fell_t,
                z_med=float(np.median(z)), z_min=float(z.min()),
                track_err_med=float(np.median(tr)),
                track_err_p90=float(np.quantile(tr, 0.9)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="data/assets/x1/x1_v4.xml")
    ap.add_argument("--sim", default="data/assets/x1/x1_sim_v4.xml")
    args = ap.parse_args()
    print(f"=== open-loop demo tracking: {args.asset} ===")
    for mi in [0, 1, 2]:
        r = probe(args.asset, args.sim, motion_idx=mi)
        print(f"motion{mi}: "
              f"{'FELL %.2fs' % r['fell_t'] if r['fell_t'] else 'no-fall 10s'}"
              f" | root_z med {r['z_med']:.3f} min {r['z_min']:.3f}"
              f" | dof err med {r['track_err_med']:.3f}"
              f" p90 {r['track_err_p90']:.3f} rad")
