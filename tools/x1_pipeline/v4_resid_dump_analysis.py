"""E-v4-RESID-02: v4 low-gain first-step engine residual (Isaac vs MuJoCo).

PRE-REGISTERED verdict rules in
reports/studio/smp_worker/2026-09-29_1921_SW-PREREG-001_e-v4-resid-02-thresholds.md
(commit 94454223, before any dump output was viewed):
  first-step max|dof_vel diff| < 1.0  -> H2 weakened, continue-training line
  1.0 - 2.0 grey                      -> needs ckpt-axis second dump
  >= 2.0                              -> H2 strengthened, route decision
Also stratify by tau saturation (|tau|/tlim >= 95%) per joint.

Inputs: output/remote_ckpt/isaac_traj_v4.pt (TASK_20260929_163, v4 assets
x1_v4.xml, pdx2 semantics, no DR, ckpt smpv4_eval.pt abs16928).
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


def main():
    dump = torch.load(REPO / "output/remote_ckpt/isaac_traj_v4.pt",
                      map_location="cpu", weights_only=False)
    kp_d = dump["kp"].numpy()
    kd_d = dump["kd"].numpy()
    tlim_d = dump["tlim"].numpy()

    m = mujoco.MjModel.from_xml_path(str(REPO / "data/assets/x1/x1_sim_v4.xml"))
    m.opt.timestep = 1.0 / 120.0
    d = mujoco.MjData(m)
    qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])
    vadr = np.array([m.joint(m.joint(j).id).dofadr[0] for j in X1_DOF_ORDER])
    m.dof_damping[vadr] = kd_d        # implicit kd, harness semantics
    m.dof_frictionloss[vadr] = 0.0

    n_steps = min(30, dump["dof_pos"].shape[0] - 1)
    steps_per_ctrl = int(round((1 / 30.0) / m.opt.timestep))
    rows = []
    for i in range(n_steps):
        d.qpos[:] = 0
        d.qpos[:3] = dump["root_pos"][i].numpy()
        x, y, z, w = dump["root_quat"][i].numpy()
        d.qpos[3:7] = [w, x, y, z]
        d.qpos[qadr] = dump["dof_pos"][i].numpy()
        d.qvel[:3] = dump["root_vel"][i].numpy()
        d.qvel[3:6] = dump["root_ang_vel"][i].numpy()
        d.qvel[vadr] = dump["dof_vel"][i].numpy()
        mujoco.mj_forward(m, d)
        q_tar = dump["q_tar"][i].numpy()
        sat_frac = None
        for _ in range(steps_per_ctrl):
            q = d.qpos[qadr]
            tau = np.clip(kp_d * (q_tar - q), -tlim_d, tlim_d)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        diff = d.qvel[vadr] - dump["dof_vel"][i + 1].numpy()
        # per-joint saturation ratio at step start (first substep)
        tau_start = np.clip(kp_d * (q_tar - dump["dof_pos"][i].numpy()),
                            -tlim_d, tlim_d)
        sat_ratio = np.abs(tau_start) / tlim_d
        rows.append(dict(
            i=i,
            rms=float(np.sqrt((diff ** 2).mean())),
            max=float(np.abs(diff).max()),
            worst_joint=X1_DOF_ORDER[int(np.argmax(np.abs(diff)))],
            worst_sat=float(sat_ratio[int(np.argmax(np.abs(diff)))]),
            n_sat_joints=int((sat_ratio >= 0.95).sum()),
            max_sat_frac=float(sat_ratio.max()),
            tau_max=float(np.abs(tau_start).max()),
        ))

    first = rows[0]
    print("=== E-v4-RESID-02 (v4 low-gain, unsaturated regime) ===")
    print(f"ckpt-axis: smpv4_eval.pt abs16928 (reward ~0.185)")
    print(f"FIRST STEP: max |dv| {first['max']:.3f} rad/s | rms {first['rms']:.3f}"
          f" | worst joint {first['worst_joint']}"
          f" (its tau sat {first['worst_sat']:.2f})")
    print(f"  tau saturation at step start: {first['n_sat_joints']} joints >=95%,"
          f" max sat frac {first['max_sat_frac']:.2f}, |tau|max {first['tau_max']:.0f} Nm")
    for r in rows[:6]:
        print(f"  step {r['i']}: rms {r['rms']:6.3f} max {r['max']:6.3f} "
              f"joint {r['worst_joint']} (sat {r['worst_sat']:.2f}) "
              f"n_sat {r['n_sat_joints']} |tau|max {r['tau_max']:.0f}")
    mean_rms = float(np.mean([r["rms"] for r in rows]))
    overall_max = float(np.max([r["max"] for r in rows]))
    print(f"OVER 30 ctrl steps: mean rms {mean_rms:.3f} | overall max {overall_max:.3f}")

    verdict = ("<1.0" if first["max"] < 1.0
               else (">=2.0" if first["max"] >= 2.0 else "GREY 1.0-2.0"))
    print(f"PRE-REGISTERED RULE -> first-step max {first['max']:.3f} rad/s = {verdict}")

    out = dict(experiment="E-v4-RESID-02",
               prereg="SW-PREREG-001 (commit 94454223)",
               dump="isaac_traj_v4.pt (TASK_20260929_163, md5 a66b6a32fff41aa5)",
               first_step=first, rows=rows,
               mean_rms=mean_rms, overall_max=overall_max,
               verdict_band=verdict)
    (REPO / "output/v4_resid_dump_analysis.json").write_text(json.dumps(out, indent=1))
    print("saved output/v4_resid_dump_analysis.json")


if __name__ == "__main__":
    main()
