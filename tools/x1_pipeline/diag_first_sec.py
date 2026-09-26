"""Deep-dive diagnostic: first seconds of MuJoCo rollout, per control step.

Logs: root_z, pitch, action stats (max abs, norm), torque saturation per
joint group, PD tracking gap. Goal: classify the collapse mechanism —
insane actions (mapping bug) vs saturated torques (actuation gap) vs
sane-but-falls (contact/ankle dynamics).
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import sim2sim_validate as SV


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        "output/remote_ckpt/smp_v2_policy_it1900.pt"
    pol = SV.Policy(ckpt)
    sim = SV.Sim2Sim(pol)
    log = dict(t=[], root_z=[], pitch=[], q=[], torque=[])
    sim.reset(seed=0)
    # instrumented rollout (mirror run_episode but stop after 1.5 s)
    ctrl = 1.0 / 30.0
    steps = int(round(ctrl / sim.m.opt.timestep))
    R0 = np.eye(3)
    import mujoco
    n_ctrl = int(1.5 / ctrl)
    for it in range(n_ctrl):
        o = sim.obs()
        a = sim.policy.forward(o)
        q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
        # action stats
        for _ in range(steps):
            q = sim.d.qpos[sim.qadr]
            tau = np.clip(sim.kp * (q_tar - q), -sim.eff, sim.eff)
            sim.d.ctrl[:] = tau
            sim.mj.mj_step(sim.m, sim.d)
        sim.mj.mj_forward(sim.m, sim.d)
        R = sim.d.xmat[sim.torso_bid].reshape(3, 3)
        pitch = np.degrees(np.arctan2(R[2, 0], R[0, 0]))
        q = sim.d.qpos[sim.qadr]
        tau = np.clip(sim.kp * (q_tar - q), -sim.eff, sim.eff)
        sat = (np.abs(tau) >= 0.95 * sim.eff)
        lag = np.abs(q_tar - q)
        from retarget_g1_x1 import X1_DOF_ORDER
        print(f"t={it*ctrl:5.2f} z={sim.d.qpos[2]:.3f} pitch={pitch:7.1f} "
              f"|a|max={np.abs(a).max():6.3f} sat={sat.sum():2d} "
              f"tau/eff max={np.max(np.abs(tau)/sim.eff):.2f} "
              f"pd_gap_max={lag.max():6.3f} "
              f"sat_joints={[X1_DOF_ORDER[i] for i in np.where(sat)[0]][:4]}")
        if sim.d.qpos[2] < 0.30:
            print("  -> fallen")
            break


if __name__ == "__main__":
    main()
