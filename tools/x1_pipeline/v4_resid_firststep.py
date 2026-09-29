"""E-v4-RESID-01: first-step engine residual under v3-high vs v4-low gains.

Background: FINAL_REPORT §4.2 measured a 3.3 rad/s (pos-mode) -> 2.4 rad/s
(pdx2) first-control-step dof_vel divergence between Isaac and the MuJoCo
harness on identical reset states. v4 lowered kp/kd (commit 05f8b905)
specifically to shrink the residual TORQUE (108 -> <=6 N*m), but the v4
residual itself was never re-measured.

Method: replay the existing Isaac pdx2 dump (isaac_traj_pdx2.pt, v3-high
gains, explicit-kp + implicit-kd semantics identical to the harness).
For each dumped step i: set MuJoCo state from dump frame i, apply the SAME
tau = clip(kp*(q_tar[i]-q)) with dumped kp (and implicit kd = dumped kd),
step one CONTROL step (4x120Hz), compare dof_vel vs dump frame i+1.

Gain sensitivity: repeat the MuJoCo side with v4 gains (kp_v4, kd_v4) on
the SAME dumped state and q_tar (Isaac counterpart for v4 gains is not
dumped - that comparison isolates how the MUJOCO response changes with
gains; the engine-pair residual for v4 is then bounded/estimated from the
tau ratio).

Outputs: per-step rms/max dof_vel diff (v3 gains, engine pair), MuJoCo-only
response change v3->v4 gains, per-joint breakdown of the worst steps.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))

import mujoco  # noqa: E402
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402
from build_x1_assets import parse_urdf_limits  # noqa: E402


def build(sim_xml, asset_xml, kd_override=None):
    m = mujoco.MjModel.from_xml_path(str(REPO / sim_xml))
    m.opt.timestep = 1.0 / 120.0
    d = mujoco.MjData(m)
    qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])
    vadr = np.array([m.joint(m.joint(j).id).dofadr[0] for j in X1_DOF_ORDER])
    mi = mujoco.MjModel.from_xml_path(str(REPO / asset_xml))
    kp = np.array([mi.jnt_stiffness[mi.joint(j).id] for j in X1_DOF_ORDER], float)
    kd = np.array([mi.dof_damping[mi.joint(mi.joint(j).id).dofadr[0]]
                   for j in X1_DOF_ORDER], float)
    if kd_override is not None:
        kd = kd_override
    lim = parse_urdf_limits()
    eff = np.array([lim[j]["effort"] for j in X1_DOF_ORDER])
    m.dof_damping[vadr] = kd
    m.dof_frictionloss[vadr] = 0.0
    return m, d, qadr, vadr, kp, kd, eff


def set_state(m, d, qadr, vadr, root_pos, root_quat, root_vel, root_ang, q, qd):
    d.qpos[:3] = root_pos
    # dump quats are XYZW (torch_util); MuJoCo wants WXYZ
    x, y, z, w = root_quat
    d.qpos[3:7] = [w, x, y, z]
    d.qpos[qadr] = q
    d.qvel[:3] = root_vel
    d.qvel[3:6] = root_ang  # local frame both sides
    d.qvel[vadr] = qd
    mujoco.mj_forward(m, d)


def main():
    dump = torch.load(REPO / "output/remote_ckpt/isaac_traj_pdx2.pt",
                      map_location="cpu", weights_only=False)
    kp_d = dump["kp"].numpy()
    kd_d = dump["kd"].numpy()
    tlim_d = dump["tlim"].numpy()

    variants = {}
    m, d, qadr, vadr, kp, kd, eff = build(
        "data/assets/x1/x1_sim.xml", "data/assets/x1/x1.xml")
    # v3 replay uses the DUMPED gains (what Isaac actually used)
    variants["v3_dumped_gains"] = (m, d, qadr, vadr, kp_d, kd_d, tlim_d)
    m4, d4, qadr4, vadr4, kp4, kd4, eff4 = build(
        "data/assets/x1/x1_sim_v4.xml", "data/assets/x1/x1_v4.xml")
    variants["v4_assets_gains"] = (m4, d4, qadr4, vadr4, kp4, kd4, eff4)

    n_steps = min(30, dump["dof_pos"].shape[0] - 1)
    results = {}
    for name, (m, d, qadr, vadr, kp, kd, eff) in variants.items():
        rows = []
        for i in range(n_steps):
            set_state(m, d, qadr, vadr,
                      dump["root_pos"][i].numpy(), dump["root_quat"][i].numpy(),
                      dump["root_vel"][i].numpy(), dump["root_ang_vel"][i].numpy(),
                      dump["dof_pos"][i].numpy(), dump["dof_vel"][i].numpy())
            q_tar = dump["q_tar"][i].numpy()
            steps = int(round((1 / 30.0) / m.opt.timestep))
            for _ in range(steps):
                q = d.qpos[qadr]
                tau = np.clip(kp * (q_tar - q), -eff, eff)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
            diff = d.qvel[vadr] - dump["dof_vel"][i + 1].numpy()
            rows.append(dict(i=i, rms=float(np.sqrt((diff ** 2).mean())),
                             max=float(np.abs(diff).max()),
                             worst_joint=X1_DOF_ORDER[int(np.argmax(np.abs(diff)))],
                             max_tau=float(np.abs(np.clip(
                                 kp * (q_tar - d.qpos[qadr]), -eff, eff)).max())))
        results[name] = rows
        first5 = rows[:5]
        print(f"[{name}] first-step rms {first5[0]['rms']:.2f} "
              f"max {first5[0]['max']:.2f} ({first5[0]['worst_joint']}) | "
              f"mean rms over {n_steps} steps "
              f"{np.mean([r['rms'] for r in rows]):.2f} | "
              f"max overall {np.max([r['max'] for r in rows]):.2f}")
        for r in first5:
            print(f"   step {r['i']}: rms {r['rms']:6.2f} max {r['max']:6.2f} "
                  f"joint {r['worst_joint']} |tau|max {r['max_tau']:.1f}")

    out = dict(experiment="E-v4-RESID-01",
               dump="isaac_traj_pdx2.pt (v3-high-gain pdx2 semantics)",
               results=results)
    (REPO / "output/v4_resid_firststep.json").write_text(json.dumps(out, indent=1))
    print("saved output/v4_resid_firststep.json")


if __name__ == "__main__":
    main()
