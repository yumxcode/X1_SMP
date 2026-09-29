"""TRUE static-stance positive control for the X1 sim2sim harness (v4 era).

Experiment E-v4-STATIC-01
Hypothesis H1: with a PERFECT static init (qpos = home, qvel = 0, root_z
  calibrated so both soles rest on the ground) a hold-home controller using
  the SAME actuation semantics as run_episode (explicit clip(kp*(qtar-q)) +
  implicit dof_damping kd) keeps the robot standing >= 10 s under the v4
  low-gain assets.
Alternative H0: the robot falls even from a perfect static home under both
  the v4 and the v3 (high-gain) assets -> the harness/asset design cannot
  statically stand at all and no policy maturity can pass sim2sim; the
  positive-control failure then indicts the validation environment.

This differs from debug_hold.py (which resets to a RUNNING demo frame with
finite-diff velocities and then switches targets to home — a transient
catch, not a static stance test).

Usage: python tools/x1_pipeline/v4_static_stance_probe.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))

import sim2sim_validate as SV  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402


def build_model(sim_xml, asset_xml, kp_scale=None, ankle_kp=None, ankle_kd=None):
    """Load sim model + gains exactly like Sim2Sim.__init__ (implicit kd,
    zero frictionloss, 1/120 timestep)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(REPO / sim_xml))
    m.opt.timestep = 1.0 / 120.0
    d = mujoco.MjData(m)
    qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])
    vadr = np.array([m.joint(m.joint(j).id).dofadr[0] for j in X1_DOF_ORDER])
    mi = mujoco.MjModel.from_xml_path(str(REPO / asset_xml))
    kp = np.array([mi.jnt_stiffness[mi.joint(j).id] for j in X1_DOF_ORDER], float)
    kd = np.array([mi.dof_damping[mi.joint(mi.joint(j).id).dofadr[0]]
                   for j in X1_DOF_ORDER], float)
    if kp_scale is not None:
        kp = kp * kp_scale
    if ankle_kp is not None or ankle_kd is not None:
        for i, j in enumerate(X1_DOF_ORDER):
            if "ankle" in j:
                if ankle_kp is not None:
                    kp[i] = ankle_kp
                if ankle_kd is not None:
                    kd[i] = ankle_kd
    from build_x1_assets import parse_urdf_limits
    lim = parse_urdf_limits()
    eff = np.array([lim[j]["effort"] for j in X1_DOF_ORDER])
    m.dof_damping[vadr] = kd          # implicit kd (run_episode semantics)
    m.dof_frictionloss[vadr] = 0.0    # parity audit T013
    return m, d, qadr, vadr, kp, kd, eff


def static_init(m, d, qadr, home, sole_gids, target_clear=0.001):
    """Perfect static init: upright torso, home joints, zero velocity,
    root_z iterated until min sole-geom bottom z == target_clear."""
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0                      # identity quat (upright)
    d.qpos[qadr] = home
    d.qvel[:] = 0.0
    lo, hi = 0.3, 1.5
    for _ in range(60):                  # bisect on root_z
        z = 0.5 * (lo + hi)
        d.qpos[2] = z
        SV_mujoco_step = None
        import mujoco
        mujoco.mj_forward(m, d)
        minz = min(d.geom_xpos[g][2] - m.geom_size[g][1]
                   for g in sole_gids)
        if minz < target_clear:
            lo = z
        else:
            hi = z
    d.qpos[2] = 0.5 * (lo + hi)
    import mujoco
    mujoco.mj_forward(m, d)
    minz = min(d.geom_xpos[g][2] - m.geom_size[g][1] for g in sole_gids)
    return d.qpos[2], minz


def hold_run(m, d, qadr, vadr, kp, eff, home, length_s=10.0):
    """Hold-home with run_episode semantics: 30 Hz ctrl, 4x120 Hz steps,
    tau = clip(kp*(qtar-q)) (kd implicit in dof_damping)."""
    ctrl_period = 1.0 / 30.0
    steps = int(round(ctrl_period / m.opt.timestep))
    n_ctrl = int(length_s / ctrl_period)
    import mujoco
    ankle_idx = [i for i, j in enumerate(X1_DOF_ORDER) if "ankle" in j]
    knee_idx = [i for i, j in enumerate(X1_DOF_ORDER) if "knee" in j]
    fell_t = None
    log_z, log_defl_a, log_defl_k, log_tau_ratio = [], [], [], []
    for it in range(n_ctrl):
        for _ in range(steps):
            q = d.qpos[qadr]
            tau = np.clip(kp * (home - q), -eff, eff)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        log_z.append(float(d.qpos[2]))
        q = d.qpos[qadr]
        log_defl_a.append(float(np.abs(q[ankle_idx] - home[ankle_idx]).max()))
        log_defl_k.append(float(np.abs(q[knee_idx] - home[knee_idx]).max()))
        tau_now = np.clip(kp * (home - q), -eff, eff)
        log_tau_ratio.append(float((np.abs(tau_now) / eff).max()))
        if d.qpos[2] < 0.30 and fell_t is None:
            fell_t = (it + 1) * ctrl_period
            break
    z = np.array(log_z)
    return dict(
        fell_t=fell_t,
        root_z0=float(z[0]), root_z_end=float(z[-1]),
        root_z_med=float(np.median(z)), root_z_min=float(z.min()),
        ankle_defl_med=float(np.median(log_defl_a)),
        ankle_defl_max=float(np.max(log_defl_a)),
        knee_defl_med=float(np.median(log_defl_k)),
        knee_defl_max=float(np.max(log_defl_k)),
        tau_ratio_max=float(np.max(log_tau_ratio)),
    )


def main():
    home = np.zeros(29)
    home[17:23] = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
    home[23:29] = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]

    cfgs = [
        ("v4_stock", dict(sim_xml="data/assets/x1/x1_sim_v4.xml",
                          asset_xml="data/assets/x1/x1_v4.xml")),
        ("v3_highgain", dict(sim_xml="data/assets/x1/x1_sim.xml",
                             asset_xml="data/assets/x1/x1.xml")),
        ("v4_ankle100_2", dict(sim_xml="data/assets/x1/x1_sim_v4.xml",
                               asset_xml="data/assets/x1/x1_v4.xml",
                               ankle_kp=100.0, ankle_kd=2.0)),
        ("v4_ankle150_3", dict(sim_xml="data/assets/x1/x1_sim_v4.xml",
                               asset_xml="data/assets/x1/x1_v4.xml",
                               ankle_kp=150.0, ankle_kd=3.0)),
        ("v4_kpx1.5", dict(sim_xml="data/assets/x1/x1_sim_v4.xml",
                           asset_xml="data/assets/x1/x1_v4.xml",
                           kp_scale=1.5)),
    ]
    results = {}
    for name, cfg in cfgs:
        kw = {k: v for k, v in cfg.items()}
        m, d, qadr, vadr, kp, kd, eff = build_model(**kw)
        sole_gids = [g for g in range(m.ngeom)
                     if (m.geom(g).name or "").endswith("_sole")]
        z0, minz = static_init(m, d, qadr, home, sole_gids)
        r = hold_run(m, d, qadr, vadr, kp, eff, home)
        r["init_root_z"] = round(z0, 4)
        r["init_min_sole_z"] = round(minz, 5)
        results[name] = r
        print(f"[{name}] init_z={z0:.4f} sole_min={minz*1000:.1f}mm | "
              f"{'FELL %.2fs' % r['fell_t'] if r['fell_t'] else 'UP 10s'}"
              f" | z_med {r['root_z_med']:.3f} (init {r['root_z0']:.3f})"
              f" | ankle defl med/max {r['ankle_defl_med']:.3f}"
              f"/{r['ankle_defl_max']:.3f} rad"
              f" | knee defl med/max {r['knee_defl_med']:.3f}"
              f"/{r['knee_defl_max']:.3f} rad"
              f" | tau_ratio_max {r['tau_ratio_max']:.2f}")

    out = REPO / "output/v4_static_stance_probe.json"
    out.write_text(json.dumps(
        dict(experiment="E-v4-STATIC-01",
             note="perfect static init + hold-home, run_episode semantics",
             results=results), indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
