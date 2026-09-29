"""v4 actuator-utilization diagnostic at MuJoCo failure time.

Measures per-joint PD deflection (|q_tar - q|) and torque saturation for
ankle/knee/hip during the (failing) sim2sim rollout. Discriminates
'gains physically too weak' (saturation, huge ankle deflection) from
'immature policy commanding infeasible targets' (large deflection with
torque headroom, concentrated at knee/hip).

Run: python tools/x1_pipeline/v4_actuator_diag.py [--ckpt ...]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import sim2sim_validate as SV

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(
        REPO / "output/remote_ckpt/smpv4e_it700.pt"))
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()

    pol = SV.Policy(args.ckpt)
    SV.Sim2Sim.ENV_YAML_REL = "data/envs/smp_x1_env_v3.yaml"
    sim = SV.Sim2Sim(pol, sim_xml=str(REPO / "data/assets/x1/x1_sim_v4.xml"),
                     asset_xml=str(REPO / "data/assets/x1/x1_v4.xml"))

    ankle_p, ankle_r, knee, hip_p = [21, 27], [22, 28], [20, 26], [17, 23]
    idx = ankle_p + ankle_r + knee + hip_p
    names = ["L_ankP", "R_ankP", "L_ankR", "R_ankR",
             "L_knee", "R_knee", "L_hipP", "R_hipP"]

    for ep in range(args.episodes):
        sim.reset(1 + ep)
        ctrl_period = 1.0 / 30.0
        steps = int(round(ctrl_period / sim.m.opt.timestep))
        defl, tau, sat, fell = [], [], [], None
        for it in range(300):
            o = sim.obs()
            a = pol.forward(o)
            q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
            for _ in range(steps):
                q = sim.d.qpos[sim.qadr]
                t = np.clip(sim.kp * (q_tar - q), -sim.eff, sim.eff)
                sim.d.ctrl[:] = t
                sim.mj.mj_step(sim.m, sim.d)
            q = sim.d.qpos[sim.qadr]
            defl.append([abs(q_tar[i] - q[i]) for i in idx])
            tau.append([abs(t[i]) for i in idx])
            sat.append([abs(t[i]) >= 0.95 * sim.eff[i] for i in idx])
            if sim.d.qpos[2] < 0.30 and fell is None:
                fell = (it + 1) * ctrl_period
                break
        defl, tau, sat = np.array(defl), np.array(tau), np.array(sat)
        print(f"--- ep{ep}: fell {fell if fell else 'no'} ---")
        print("  median deflection (rad):",
              {n: round(float(np.median(defl[:, j])), 3)
               for j, n in enumerate(names)})
        print("  p90 deflection    (rad):",
              {n: round(float(np.quantile(defl[:, j], 0.9)), 3)
               for j, n in enumerate(names)})
        print("  tau/eff p90            :",
              {n: round(float(np.quantile(tau[:, j] / sim.eff[idx[j]], 0.9)), 2)
               for j, n in enumerate(names)})
        print("  sat frac               :",
              {n: round(float(sat[:, j].mean()), 2)
               for j, n in enumerate(names)})
