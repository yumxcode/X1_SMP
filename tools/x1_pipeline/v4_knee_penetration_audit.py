"""E-v4-KNEE-01: does the v3 demo dataset itself put the knee collision
capsules below ground?

The retarget v3 quality gate checks SOLE penetration only (R8 final-frame
zero-penetration uses sole corners). The knee collision capsule spans the
whole shank (fromto len 0.305 m, r=0.048 m). At running knee flexion the
shank tilts and the capsule lower surface can dip below the sole.

Method: offline FK over every demo frame (root pose + joints from the pkl),
compute per-frame min z of both knee capsules and of both sole boxes.
Metrics: knee-capsule min z distribution, % frames below ground,
% frames where a knee capsule is >= 10 mm below the sole's min.
"""
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402

m = mujoco.MjModel.from_xml_path(str(REPO / "data/assets/x1/x1_sim_v4.xml"))
d = mujoco.MjData(m)
qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])

knee_gids = [g for g in range(m.ngeom)
             if "knee" in (m.geom(g).name or "")]
sole_gids = [g for g in range(m.ngeom)
             if (m.geom(g).name or "").endswith("_sole")]


def capsule_minz(g):
    """Min world-z over a capsule surface: min over axis segment minus r.

    geom axis in world: capsule from fromto is compiled with local z along
    the segment; use geom_xmat third column.
    """
    pos = d.geom_xpos[g]
    R = d.geom_xmat[g].reshape(3, 3)
    r, half = m.geom_size[g][0], m.geom_size[g][1]
    ax = R @ np.array([0.0, 0.0, 1.0])
    a = pos - ax * half
    b = pos + ax * half
    seg_min_z = min(a[2], b[2])
    return seg_min_z - r


def sole_minz(g):
    hx, hy, hz = m.geom_size[g]
    local = np.array([[sx * hx, sy * hy, -hz]
                      for sx in (-1, 1) for sy in (-1, 1)])
    R = d.geom_xmat[g].reshape(3, 3)
    return (local @ R.T + d.geom_xpos[g])[:, 2].min()


def audit(envf):
    p = REPO / envf
    if not p.exists():
        print(f"(skip missing {envf})")
        return
    ds_line = [l for l in p.read_text().splitlines()
               if l.strip().startswith("motion_file:")][0]
    ds = REPO / re.search(r'"([^"]+)"', ds_line).group(1)
    files = [REPO / mm.group(1) for mm in
             (re.search(r'file:\s*"([^"]+)"', l)
              for l in ds.read_text().splitlines()) if mm]
    print(f"### {envf} -> {ds.name}: {len(files)} clips")
    worst = []
    for f in files:
        mo = pickle.load(open(f, "rb"))
        F = np.asarray(mo["frames"])
        knee_pen, sole_pen = [], []
        for i in range(0, len(F), 2):
            fr = F[i]
            d.qpos[:] = 0
            d.qpos[:3] = fr[0:3]
            em = np.asarray(fr[3:6])
            ang = np.linalg.norm(em)
            if ang < 1e-8:
                d.qpos[3:7] = [1, 0, 0, 0]
            else:
                ax = em / ang
                d.qpos[3:7] = [np.cos(ang / 2), *(ax * np.sin(ang / 2))]
            d.qpos[qadr] = fr[6:35]
            mujoco.mj_forward(m, d)
            kz = min(capsule_minz(g) for g in knee_gids)
            sz = min(sole_minz(g) for g in sole_gids)
            knee_pen.append(kz)
            sole_pen.append(sz)
        knee_pen = np.array(knee_pen)
        sole_pen = np.array(sole_pen)
        n = len(knee_pen)
        print(f"  {f.name:34s} n={n:4d} | knee z: min {knee_pen.min()*1e3:7.1f}mm "
              f"med {np.median(knee_pen)*1e3:6.1f}mm | knee<0: {(knee_pen<0).mean()*100:5.1f}% "
              f"| knee<sole-10mm: {((knee_pen) < (sole_pen - 0.01)).mean()*100:5.1f}% "
              f"| sole z min {sole_pen.min()*1e3:6.1f}mm")
        worst.append((f.name, float(knee_pen.min()),
                      float((knee_pen < 0).mean())))
    return worst


if __name__ == "__main__":
    audit("data/envs/smp_x1_env.yaml")
    audit("data/envs/smp_x1_env_v3.yaml")
