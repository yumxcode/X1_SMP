"""E-v4-STATIC-02: static equilibrium geometry audit of the home pose.

For a hold-home stance to be statically stable the whole-body COM xy
projection must sit inside the support polygon (convex hull of both sole
boxes). This probe measures:
  - COM xy (world) at the perfect static init
  - support polygon extent (per-foot sole box corners, world xy)
  - COM height, total mass
  - per-foot sole corner heights (flatness: are both soles level?)

Run under both v4 and v3 sim models.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))

import mujoco  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402


def sole_corners(m, d, g):
    """World coords of the 4 bottom corners of a sole box geom."""
    pos = d.geom_xpos[g].copy()
    R = d.geom_xmat[g].reshape(3, 3).copy()
    hx, hy, hz = m.geom_size[g]
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            local = np.array([sx * hx, sy * hy, -hz])
            corners.append(pos + R @ local)
    return np.array(corners)


def audit(sim_xml, label):
    m = mujoco.MjModel.from_xml_path(str(REPO / sim_xml))
    d = mujoco.MjData(m)
    qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])

    home = np.zeros(29)
    home[17:23] = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
    home[23:29] = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]

    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    d.qpos[qadr] = home
    d.qvel[:] = 0.0

    sole_gids = [g for g in range(m.ngeom)
                 if (m.geom(g).name or "").endswith("_sole")]
    lo, hi = 0.3, 1.5
    for _ in range(60):
        z = 0.5 * (lo + hi)
        d.qpos[2] = z
        mujoco.mj_forward(m, d)
        minz = min(d.geom_xpos[g][2] - m.geom_size[g][1] for g in sole_gids)
        if minz < 0.001:
            lo = z
        else:
            hi = z
    d.qpos[2] = 0.5 * (lo + hi)
    mujoco.mj_forward(m, d)

    com = d.subtree_com[0]  # whole-body COM (world)
    corners_l, corners_r, all_c = None, None, []
    for g in sole_gids:
        nm = m.geom(g).name
        c = sole_corners(m, d, g)
        all_c.append(c)
        if nm.startswith("l") or "left" in nm:
            corners_l = c
        else:
            corners_r = c
    all_c = np.vstack(all_c)

    print(f"=== {label} ({sim_xml}) ===")
    print(f"total mass: {m.body_mass.sum():.2f} kg")
    print(f"COM world: x={com[0]:+.4f} y={com[1]:+.4f} z={com[2]:.4f}")
    print(f"support polygon xy: x [{all_c[:,0].min():+.4f}, {all_c[:,0].max():+.4f}]"
          f"  y [{all_c[:,1].min():+.4f}, {all_c[:,1].max():+.4f}]")
    cx = 0.5 * (all_c[:, 0].min() + all_c[:, 0].max())
    cy = 0.5 * (all_c[:, 1].min() + all_c[:, 1].max())
    print(f"polygon center: x={cx:+.4f} y={cy:+.4f}")
    print(f"COM offset from polygon center: dx={com[0]-cx:+.4f} dy={com[1]-cy:+.4f} m")
    for nm, c in (("left", corners_l), ("right", corners_r)):
        if c is None:
            continue
        print(f"{nm} sole corners z: {np.round(c[:,2], 4)} (flat if equal)"
              f" | xy centroid: ({c[:,0].mean():+.4f}, {c[:,1].mean():+.4f})")
        print(f"{nm} COM-foot dx={com[0]-c[:,0].mean():+.4f}"
              f" dy={com[1]-c[:,1].mean():+.4f}")
    # inside-polygon check (2D point-in-convex-hull via half-planes)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(all_c[:, :2])
    inside = True
    margins = []
    for eq in hull.equations:
        m_ = -(eq[0] * com[0] + eq[1] * com[1] + eq[2])
        margins.append(m_)
        if m_ < 0:
            inside = False
    print(f"COM inside support hull: {inside}; min margin {min(margins):+.4f} m")
    print()
    return dict(com=com.tolist(), inside=inside,
                min_margin=float(min(margins)), mass=float(m.body_mass.sum()))


if __name__ == "__main__":
    audit("data/assets/x1/x1_sim_v4.xml", "v4")
    audit("data/assets/x1/x1_sim.xml", "v3")
