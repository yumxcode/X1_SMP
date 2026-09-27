"""Aligned actuator retrofit for the MuJoCo sim2sim harness.

Isaac side (verified from repo source):
  * control_mode "pos" -> PhysX DOF POSITION DRIVE (implicit), torque
    internally tau = kp*(tar-q) - kd*qd, force limited by motor effort.
  * action bounds (char_env._build_action_bounds_pos, zero_center_action
    ABSENT in smp_x1_env.yaml -> False): mid = 0.5*(lo+hi),
    half = 1.4 * max(|hi-mid|, |lo-mid|)  -> bounds [mid-half, mid+half].
  * action IS the absolute joint-position target (clipped to bounds).

MuJoCo equivalent (this retrofit, applied in-memory):
  * per-joint position servo actuator: gainprm=[kp], biasprm=[0,-kp,-kv]
    (MuJoCo integrates actuator + damping implicitly with Euler — matches
    PhysX implicit drive far better than an explicit tau=kp*(tar-q)).
  * actuator ctrlrange = Isaac action bounds; forcerange = [-eff, eff].
  * run loop writes d.ctrl = action directly.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import sim2sim_validate as SV


def retrofit(sim):
    """Convert the torque motors into Isaac-aligned position servos."""
    import mujoco
    m = sim.m
    from build_x1_assets import parse_urdf_limits
    from retarget_g1_x1 import X1_DOF_ORDER
    lim = parse_urdf_limits()

    kp = sim.kp
    kd = sim.kd
    # Isaac action bounds (zero_center_action=False semantics)
    a_lo, a_hi = [], []
    for j in X1_DOF_ORDER:
        lo, hi = lim[j]["low"], lim[j]["high"]
        mid = 0.5 * (hi + lo)
        half = 1.4 * max(abs(hi - mid), abs(lo - mid))
        a_lo.append(mid - half)
        a_hi.append(mid + half)
    a_lo = np.array(a_lo)
    a_hi = np.array(a_hi)

    for k, dof_adr in enumerate(sim.vadr):
        act_id = k  # actuators are declared in X1_DOF_ORDER in x1_sim.xml
        # verify mapping: actuator joint == this dof
        aj = m.actuator_trnid[act_id, 0]
        assert m.jnt_dofadr[aj] == dof_adr, \
            f"actuator {act_id} joint mismatch at dof {dof_adr}"
        m.actuator_gaintype[act_id] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[act_id, :3] = [kp[k], 0, 0]
        m.actuator_biastype[act_id] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_biasprm[act_id, :3] = [0.0, -kp[k], -kd[k]]
        m.actuator_ctrlrange[act_id] = [a_lo[k], a_hi[k]]
        m.actuator_ctrllimited[act_id] = 1
        m.actuator_forcerange[act_id] = [-sim.eff[k], sim.eff[k]]
        m.actuator_forcelimited[act_id] = 1
        m.actuator_gear[act_id] = 1.0
    # dof_damping already = kd (kept implicit); passive stiffness is 0
    sim.a_bound = np.maximum(np.abs(a_lo), np.abs(a_hi))  # clip only
    sim.pos_mode = True
    return sim


def run_aligned(ckpt, seeds=(0, 1, 2, 3), length=10.0):
    pol = SV.Policy(ckpt)
    sim = SV.Sim2Sim(pol)
    retrofit(sim)
    import mujoco
    m, d = sim.m, sim.d
    ctrl = 1.0 / 30.0
    steps = int(round(ctrl / m.opt.timestep))
    out = []
    for seed in seeds:
        sim.reset(seed)
        log = dict(t=[], root_z=[], root_x=[], root_y=[], lf_z=[], rf_z=[],
                   lh_x=[], rh_x=[], lf_x=[], rf_x=[], pitch=[], roll=[],
                   q=[], torque=[], bad_contact=[])
        for it in range(int(length / ctrl)):
            o = sim.obs()
            a = pol.forward(o)
            d.ctrl[:] = np.clip(a, -sim.a_bound, sim.a_bound)
            for _ in range(steps):
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
            tar = d.ctrl[:len(q)]
            log["torque"].append(np.abs(np.clip(
                sim.kp * (tar - q) - sim.kd * d.qvel[sim.vadr],
                -sim.eff, sim.eff)))
            log["bad_contact"].append(any(d.geom_xpos[g][2] < 0.03
                                          for g in sim.fall_gids))
        r = SV.analyze(log, length, sim.eff)
        z = np.array(log["root_z"])
        ft = next((t for t, zz in zip(log["t"], z) if zz < 0.30), None)
        out.append((seed, ft, r))
        print(f"seed {seed}: fall_t={ft} PASS={r['PASS']} "
              f"S1={r['S1_no_fall']['pass_']} S2={r['S2_gait']['pass_']} "
              f"S3={r['S3_morphology']['pass_']} S4={r['S4_actuation']['pass_']}")
    n_pass = sum(1 for _, _, r in out if r["PASS"])
    print(f"ALIGNED-ACTUATOR RESULT: {n_pass}/{len(out)} PASS")
    return out


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else \
        str(REPO / "output/remote_ckpt/smp_v2_policy_final.pt")
    run_aligned(ckpt)
