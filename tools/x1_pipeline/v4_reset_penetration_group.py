"""E-v4-RESET-01: does an initial knee-capsule penetration at reset predict
the instant MuJoCo sim2sim fall?

Mechanism under test (from E-v4-KNEE-01): 25-68% of v3 demo frames put a
knee collision capsule 0-5.2 mm below ground (quality gate R8 checked soles
only). sim2sim resets sample those frames; MuJoCo resolves the initial
penetration with a hard push, polluting the episode from step 0. Isaac
(PhysX TGS) treats the same small penetrations differently, which is
consistent with the Isaac-falls-3-4s vs MuJoCo-falls-0.5s same-generation gap.

Design: reset to SPECIFIC frames (deterministic coverage of clean vs
penetrating), log per-episode initial knee min-z / initial contacts, run the
policy 5 s, record fall time. Compare the two groups.

Usage:
  .venv/bin/python tools/x1_pipeline/v4_reset_penetration_group.py
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))

import sim2sim_validate as SV  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402


def knee_minz(sim):
    m, d = sim.m, sim.d
    knee_gids = [g for g in range(m.ngeom) if "knee" in (m.geom(g).name or "")]
    zs = []
    for g in knee_gids:
        pos = d.geom_xpos[g]
        R = d.geom_xmat[g].reshape(3, 3)
        r, half = m.geom_size[g][0], m.geom_size[g][1]
        ax = R @ np.array([0.0, 0.0, 1.0])
        a = pos - ax * half
        b = pos + ax * half
        zs.append(min(a[2], b[2]) - r)
    return min(zs)


def reset_at(sim, mi, fi):
    """Deterministic reset to motions[mi] frame fi (same math as reset())."""
    d, mj = sim.d, sim.mj
    mj.mj_resetData(sim.m, d)
    F = np.asarray(sim.motions[mi]["frames"])
    fps = sim.motions[mi]["fps"]
    dt = 1.0 / fps
    f, f0, f1 = F[fi], F[fi - 1], F[fi + 1]
    d.qpos[:3] = f[0:3]
    em = np.asarray(f[3:6])
    ang = np.linalg.norm(em)
    if ang < 1e-8:
        d.qpos[3:7] = [1, 0, 0, 0]
    else:
        ax = em / ang
        d.qpos[3:7] = [np.cos(ang / 2), *(ax * np.sin(ang / 2))]
    d.qpos[sim.qadr] = f[6:35]
    d.qvel[:3] = (np.asarray(f1[0:3]) - np.asarray(f0[0:3])) / (2 * dt)
    R = np.zeros(9)
    mj.mju_quat2Mat(R, d.qpos[3:7])
    R = R.reshape(3, 3)
    w_world = (np.asarray(f1[3:6]) - np.asarray(f0[3:6])) / (2 * dt)
    d.qvel[3:6] = R.T @ w_world
    d.qvel[sim.vadr] = (np.asarray(f1[6:35]) - np.asarray(f0[6:35])) / (2 * dt)
    mj.mj_forward(sim.m, d)


def run(sim, length_s=5.0):
    steps = int(round((1 / 30.0) / sim.m.opt.timestep))
    n_ctrl = int(length_s * 30)
    for it in range(n_ctrl):
        o = sim.obs()
        a = sim.policy.forward(o)
        q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
        for _ in range(steps):
            q = sim.d.qpos[sim.qadr]
            tau = np.clip(sim.kp * (q_tar - q), -sim.eff, sim.eff)
            sim.d.ctrl[:] = tau
            sim.mj.mj_step(sim.m, sim.d)
        if sim.d.qpos[2] < 0.30:
            return (it + 1) / 30.0
    return None


def main():
    ckpt = REPO / "output/remote_ckpt/smpv4f_it2100.pt"
    pol = SV.Policy(str(ckpt))
    SV.Sim2Sim.ENV_YAML_REL = "data/envs/smp_x1_env_v4.yaml"
    sim = SV.Sim2Sim(pol, sim_xml="data/assets/x1/x1_sim_v4.xml",
                     asset_xml="data/assets/x1/x1_v4.xml")

    rows = []
    n_motions = len(sim.motions)
    for mi in range(n_motions):
        F = np.asarray(sim.motions[mi]["frames"])
        # pick frames across the clip; skip edges (need fi-1, fi+1)
        cand = np.linspace(2, len(F) - 3, 4).astype(int)
        for fi in cand:
            reset_at(sim, mi, int(fi))
            kz = knee_minz(sim)
            init_contacts = []
            for c in range(sim.d.ncon):
                g1 = sim.m.geom(sim.d.contact[c].geom1).name or ""
                g2 = sim.m.geom(sim.d.contact[c].geom2).name or ""
                init_contacts.append(f"{g1}|{g2}")
            knee_contact = any("knee" in s for s in init_contacts)
            fell_t = run(sim, 5.0)
            rows.append(dict(mi=mi, fi=int(fi), knee_minz_mm=round(kz * 1000, 2),
                             ncon0=sim.d.ncon if False else len(init_contacts),
                             knee_contact=bool(knee_contact),
                             fell_t=fell_t))
            print(f"m{mi:02d} f{fi:04d} knee_z {kz*1000:+7.2f}mm "
                  f"ncon0 {len(init_contacts)} knee_contact {knee_contact} "
                  f"-> {'FELL %.2fs' % fell_t if fell_t else 'UP 5s'}")

    pen = [r for r in rows if r["knee_contact"] or r["knee_minz_mm"] < 0]
    clean = [r for r in rows if not (r["knee_contact"] or r["knee_minz_mm"] < 0)]
    def stat(g):
        ft = [r["fell_t"] if r["fell_t"] is not None else 5.0 for r in g]
        return (len(g), float(np.median(ft)), float(np.mean(ft)),
                sum(1 for r in g if r["fell_t"] is None))
    out = dict(experiment="E-v4-RESET-01", ckpt=ckpt.name, rows=rows)
    for name, g in (("penetrating", pen), ("clean", clean)):
        n, med, mean, up = stat(g)
        print(f"\n[{name}] n={n} | fell_t med {med:.2f}s mean {mean:.2f}s | up5s {up}/{n}")
        out[f"{name}_summary"] = dict(n=n, median=med, mean=mean, up5s=up)
    (REPO / "output/v4_reset_penetration_group.json").write_text(json.dumps(out, indent=1))
    print("saved output/v4_reset_penetration_group.json")


if __name__ == "__main__":
    main()
