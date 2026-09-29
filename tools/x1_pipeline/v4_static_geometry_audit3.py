"""E-v4-STATIC-02c: support polygon from BOX corners with full geom_xmat.

Sole geoms are boxes (mjGEOM_BOX=6 in mujoco 3.1.6 enum) with size
(hx,hy,hz) in the GEOM local frame; the foot body carries a 90-deg y
rotation so the box long axis (local z) maps to world +x. Correct world
corners: geom_xpos + geom_xmat @ (±hx,±hy,±hz).
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))

import mujoco  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402


def box_world_corners(m, d, g):
    hx, hy, hz = m.geom_size[g]
    local = np.array([[sx * hx, sy * hy, sz * hz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    R = d.geom_xmat[g].reshape(3, 3)
    return local @ R.T + d.geom_xpos[g]


def audit(sim_xml, label, home=None):
    m = mujoco.MjModel.from_xml_path(str(REPO / sim_xml))
    d = mujoco.MjData(m)
    qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])

    if home is None:
        home = np.zeros(29)
        home[17:23] = [0.48891, 0.06213, -0.33853, 0.63204, -0.27224, 0.0]
        home[23:29] = [-0.48891, -0.06213, 0.33853, 0.63204, -0.27224, 0.0]

    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    d.qpos[qadr] = home
    d.qvel[:] = 0.0

    sole_gids = [g for g in range(m.ngeom)
                 if (m.geom(g).name or "").endswith("_sole")]

    def min_corner_z():
        return min(box_world_corners(m, d, g)[:, 2].min() for g in sole_gids)

    lo, hi = 0.3, 1.5
    for _ in range(60):
        z = 0.5 * (lo + hi)
        d.qpos[2] = z
        mujoco.mj_forward(m, d)
        if min_corner_z() < 0.001:
            lo = z
        else:
            hi = z
    d.qpos[2] = 0.5 * (lo + hi)
    mujoco.mj_forward(m, d)

    com = d.subtree_com[0].copy()
    clouds = {m.geom(g).name: box_world_corners(m, d, g) for g in sole_gids}
    all_v = np.vstack(list(clouds.values()))

    hull = ConvexHull(all_v[:, :2])
    margins = [-(eq[0] * com[0] + eq[1] * com[1] + eq[2]) for eq in hull.equations]
    inside = all(mm >= 0 for mm in margins)

    print(f"=== {label} ({Path(sim_xml).name}) ===")
    print(f"root_z={d.qpos[2]:.4f} | mass {m.body_mass.sum():.2f} kg | "
          f"COM x={com[0]:+.4f} y={com[1]:+.4f} z={com[2]:.4f}")
    print(f"polygon x [{all_v[:,0].min():+.4f},{all_v[:,0].max():+.4f}] "
          f"y [{all_v[:,1].min():+.4f},{all_v[:,1].max():+.4f}]")
    for nm, c in clouds.items():
        zs = np.sort(np.unique(np.round(c[:, 2], 4)))
        print(f"  {nm}: corner z values {zs} | "
              f"z spread {(c[:,2].max()-c[:,2].min())*1000:.1f} mm")
    cx = 0.5 * (all_v[:, 0].min() + all_v[:, 0].max())
    print(f"polygon x-center {cx:+.4f} | COM-polyCenter dx={com[0]-cx:+.4f} m")
    print(f"COM inside hull: {inside} | min margin {min(margins):+.4f} m\n")
    return dict(label=label, com=[float(v) for v in com],
                inside=bool(inside), min_margin=float(min(margins)),
                poly_x=[float(all_v[:, 0].min()), float(all_v[:, 0].max())])


if __name__ == "__main__":
    import json
    out = []
    out.append(audit("data/assets/x1/x1_sim_v4.xml", "v4 home"))
    out.append(audit("data/assets/x1/x1_sim.xml", "v3 home"))
    # demo first-frame audit: reset like sim2sim_validate (motion 0 frame 1)
    import pickle, re
    env_yaml = REPO / "data/envs/smp_x1_env.yaml"
    ds_line = [l for l in env_yaml.read_text().splitlines()
               if l.strip().startswith("motion_file:")][0]
    ds = REPO / re.search(r'"([^"]+)"', ds_line).group(1)
    mo_file = None
    for line in ds.read_text().splitlines():
        m_ = re.search(r'file:\s*"([^"]+)"', line)
        if m_:
            mo_file = REPO / m_.group(1)
            break
    mo = pickle.load(open(mo_file, "rb"))
    F = np.asarray(mo["frames"])
    f = F[1]
    home2 = np.asarray(f[6:35])
    print(f"demo audit: {mo_file.name} frame 1 root=({f[0]:.3f},{f[1]:.3f},{f[2]:.3f})")
    out.append(audit("data/assets/x1/x1_sim_v4.xml", "v4 demo-frame", home=home2))
    (REPO / "output/v4_static_geometry_audit.json").write_text(
        json.dumps(out, indent=1))
