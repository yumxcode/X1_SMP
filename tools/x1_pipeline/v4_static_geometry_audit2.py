"""E-v4-STATIC-02b: correct support-polygon audit using actual MESH vertices.

The sole geoms are MuJoCo mesh geoms (type 6), so the box-corner math in
v4_static_geometry_audit.py was invalid. Redo with mesh vertex clouds:
  - per-foot sole mesh vertices in world coords at the perfect static home
  - support polygon = convex hull of both clouds (projected to xy)
  - COM xy vs polygon, margins
  - per-foot flatness: vertex z spread when robot is set level
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))

import mujoco  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402


def geom_world_verts(m, d, g):
    mid = m.geom_dataid[g]
    verts = np.array(m.mesh_vert[mid]) * m.geom_size[g]  # scale
    pos = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    return (verts @ R.T) + pos


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
    # calibrate root_z so the LOWEST sole-mesh vertex sits at +1 mm
    def min_vertex_z():
        return min(geom_world_verts(m, d, g)[:, 2].min() for g in sole_gids)
    lo, hi = 0.3, 1.5
    for _ in range(60):
        z = 0.5 * (lo + hi)
        d.qpos[2] = z
        mujoco.mj_forward(m, d)
        if min_vertex_z() < 0.001:
            lo = z
        else:
            hi = z
    d.qpos[2] = 0.5 * (lo + hi)
    mujoco.mj_forward(m, d)

    com = d.subtree_com[0].copy()
    clouds = {m.geom(g).name: geom_world_verts(m, d, g) for g in sole_gids}
    all_v = np.vstack(list(clouds.values()))

    hull = ConvexHull(all_v[:, :2])
    margins = []
    inside = True
    for eq in hull.equations:
        mg = -(eq[0] * com[0] + eq[1] * com[1] + eq[2])
        margins.append(mg)
        if mg < 0:
            inside = False

    print(f"=== {label} ({sim_xml}) ===")
    print(f"root_z(init) = {d.qpos[2]:.4f}, min vertex z = {min_vertex_z()*1000:.1f} mm")
    print(f"mass {m.body_mass.sum():.2f} kg | COM x={com[0]:+.4f} y={com[1]:+.4f} z={com[2]:.4f}")
    print(f"polygon x [{all_v[:,0].min():+.4f},{all_v[:,0].max():+.4f}] "
          f"y [{all_v[:,1].min():+.4f},{all_v[:,1].max():+.4f}] "
          f"(x span {all_v[:,0].ptp():.4f}, y span {all_v[:,1].ptp():.4f})")
    for nm, c in clouds.items():
        print(f"  {nm}: n={len(c)} x[{c[:,0].min():+.4f},{c[:,0].max():+.4f}] "
              f"y[{c[:,1].min():+.4f},{c[:,1].max():+.4f}] z[{c[:,2].min():.4f},{c[:,2].max():.4f}]"
              f" flatness spread {c[:,2].ptp()*1000:.1f} mm")
    print(f"COM inside hull: {inside} | min margin {min(margins):+.4f} m")
    return dict(label=label, root_z=float(d.qpos[2]), com=com.tolist(),
                inside=bool(inside), min_margin=float(min(margins)))


if __name__ == "__main__":
    audit("data/assets/x1/x1_sim_v4.xml", "v4")
    audit("data/assets/x1/x1_sim.xml", "v3")
