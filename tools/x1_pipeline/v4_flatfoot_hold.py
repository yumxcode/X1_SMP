"""E-v4-STATIC-03: flat-foot stance positive control.

E-v4-STATIC-01/02 established: home joint array puts both soles 24 mm
non-level (line contact) -> hold-home falls at ~1.7 s under ALL gain sets;
COM is inside the polygon, torque far from saturation. So the home array is
not a stance pose and that positive control was invalid as an environment
indictment.

This probe solves a FLAT-FOOT stance pose (ankle_pitch/roll per side solved
so both sole boxes are level), then holds it for 10 s with run_episode
semantics under v4 and v3 gains. Distinguishes:
  H1 (pass): flat-foot hold stands >= 10 s -> harness statically sound,
     sim2sim failures are dynamic-side (policy/engine), maturity line OK.
  H0 (fail): even flat-foot static hold falls -> harness-level defect
     (contact/integrator/assets), must be fixed before any policy claim.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))

import mujoco  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402
from build_x1_assets import parse_urdf_limits  # noqa: E402


def idx_of(name):
    return X1_DOF_ORDER.index(name)


LA_P, LA_R = idx_of("left_ankle_pitch_joint"), idx_of("left_ankle_roll_joint")
RA_P, RA_R = idx_of("right_ankle_pitch_joint"), idx_of("right_ankle_roll_joint")


def load(sim_xml, asset_xml):
    m = mujoco.MjModel.from_xml_path(str(REPO / sim_xml))
    m.opt.timestep = 1.0 / 120.0
    d = mujoco.MjData(m)
    qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])
    vadr = np.array([m.joint(m.joint(j).id).dofadr[0] for j in X1_DOF_ORDER])
    mi = mujoco.MjModel.from_xml_path(str(REPO / asset_xml))
    kp = np.array([mi.jnt_stiffness[mi.joint(j).id] for j in X1_DOF_ORDER], float)
    kd = np.array([mi.dof_damping[mi.joint(mi.joint(j).id).dofadr[0]]
                   for j in X1_DOF_ORDER], float)
    lim = parse_urdf_limits()
    eff = np.array([lim[j]["effort"] for j in X1_DOF_ORDER])
    m.dof_damping[vadr] = kd
    m.dof_frictionloss[vadr] = 0.0
    sole_gids = [g for g in range(m.ngeom)
                 if (m.geom(g).name or "").endswith("_sole")]
    return m, d, qadr, vadr, kp, kd, eff, sole_gids


def corners(m, d, g):
    hx, hy, hz = m.geom_size[g]
    local = np.array([[sx * hx, sy * hy, sz * hz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def set_state(m, d, qadr, q):
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    d.qpos[qadr] = q
    d.qvel[:] = 0.0


def settle_root_z(m, d, sole_gids, target=0.001):
    def minz():
        return min(corners(m, d, g)[:, 2].min() for g in sole_gids)
    lo, hi = 0.3, 1.5
    for _ in range(60):
        z = 0.5 * (lo + hi)
        d.qpos[2] = z
        mujoco.mj_forward(m, d)
        lo, hi = (z, hi) if minz() < target else (lo, z)
    d.qpos[2] = 0.5 * (lo + hi)
    mujoco.mj_forward(m, d)


def solve_flat(m, d, qadr, sole_gids, home, iters=40):
    """Gauss-Seidel: per ankle pitch/roll, minimize sole corner z spread."""
    q = home.copy()
    for _ in range(iters):
        for ap, ar, g in ((LA_P, LA_R, sole_gids[0]),
                          (RA_P, RA_R, sole_gids[-1])):
            best = None
            for dp in (-0.02, -0.01, 0.0, 0.01, 0.02):
                for dr in (-0.02, -0.01, 0.0, 0.01, 0.02):
                    q2 = q.copy()
                    q2[ap] += dp
                    q2[ar] += dr
                    set_state(m, d, qadr, q2)
                    settle_root_z(m, d, sole_gids)
                    c = corners(m, d, g)
                    cost = np.ptp(c[:, 2])
                    if best is None or cost < best[0]:
                        best = (cost, dp, dr)
            q[ap] += best[1]
            q[ar] += best[2]
        set_state(m, d, qadr, q)
        settle_root_z(m, d, sole_gids)
        spreads = [np.ptp(corners(m, d, g)[:, 2]) for g in sole_gids]
        if max(spreads) < 1e-4:
            break
    set_state(m, d, qadr, q)
    settle_root_z(m, d, sole_gids)
    spreads = [np.ptp(corners(m, d, g)[:, 2]) for g in sole_gids]
    return q, spreads


def hold(m, d, qadr, kp, eff, qtar, length_s=10.0):
    steps = int(round((1 / 30.0) / m.opt.timestep))
    n_ctrl = int(length_s * 30)
    ankle_idx = [i for i, j in enumerate(X1_DOF_ORDER) if "ankle" in j]
    fell_t = None
    zs, defl = [], []
    for it in range(n_ctrl):
        for _ in range(steps):
            q = d.qpos[qadr]
            d.ctrl[:] = np.clip(kp * (qtar - q), -eff, eff)
            mujoco.mj_step(m, d)
        zs.append(float(d.qpos[2]))
        defl.append(float(np.abs(d.qpos[qadr][ankle_idx] - qtar[ankle_idx]).max()))
        if d.qpos[2] < 0.30:
            fell_t = (it + 1) / 30.0
            break
    return dict(fell_t=fell_t, z_med=float(np.median(zs)),
                z_min=float(np.min(zs)), ankle_defl_max=float(np.max(defl)))


def main():
    home = np.zeros(29)
    home[17:23] = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
    home[23:29] = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]

    out = {"experiment": "E-v4-STATIC-03"}
    for label, sx, ax in (("v4", "data/assets/x1/x1_sim_v4.xml",
                           "data/assets/x1/x1_v4.xml"),
                          ("v3", "data/assets/x1/x1_sim.xml",
                           "data/assets/x1/x1.xml")):
        m, d, qadr, vadr, kp, kd, eff, soles = load(sx, ax)
        set_state(m, d, qadr, home)
        settle_root_z(m, d, soles)
        pre = [np.ptp(corners(m, d, g)[:, 2]) for g in soles]
        q_flat, spreads = solve_flat(m, d, qadr, soles, home)
        com = d.subtree_com[0].copy()
        print(f"[{label}] pre-spread {np.round(pre*1000,1)} mm -> "
              f"flat-spread {np.round(spreads,5)} m | root_z {d.qpos[2]:.4f} "
              f"| ankle L p/r {q_flat[LA_P]:+.4f}/{q_flat[LA_R]:+.4f} "
              f"R {q_flat[RA_P]:+.4f}/{q_flat[RA_R]:+.4f} "
              f"| COM x {com[0]:+.4f}")
        r = hold(m, d, qadr, kp, eff, q_flat)
        print(f"[{label}] FLAT-FOOT HOLD: "
              f"{'FELL %.2fs' % r['fell_t'] if r['fell_t'] else 'UP 10s'}"
              f" | z_med {r['z_med']:.3f} z_min {r['z_min']:.3f}"
              f" | ankle defl max {r['ankle_defl_max']:.3f} rad")
        r.update(pre_spread_mm=[float(v * 1000) for v in pre],
                 flat_spread_m=[float(v) for v in spreads],
                 q_flat=[float(v) for v in q_flat], com=[float(v) for v in com])
        out[label] = r
    (REPO / "output/v4_flatfoot_hold.json").write_text(json.dumps(out, indent=1))
    print("saved output/v4_flatfoot_hold.json")


if __name__ == "__main__":
    main()
