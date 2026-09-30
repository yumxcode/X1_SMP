"""SW-PREREG-005 analysis: full-X1 Isaac Lab entry probe (H-X1-IL) verdict.

Input: output/remote_ckpt/x1_il_entry_F1_implicit_v4.pt
(remote task, dumped by scripts_remote/probe_x1_il_entry.py).

MuJoCo harness replay (same as SW-R004/R005 analyses): x1_sim_v4.xml @120Hz,
explicit kp torque + implicit dof_damping=kd, tlim clip; replay the SAME
open-loop q_tar sequence from the anchor dump.

Pre-registered verdict (SW-PREREG-005, commit 15151d1a):
  F1 first-step max < 1.0        -> H_X1_IL_GREEN (unlock training pipeline)
  1.0 <= x < 2.0                 -> grey (per-joint + zero-error stratify)
  >= 2.0                         -> H_X1_IL_REFUTED (MuJoCo-native sole
                                    primary, preregistered chain)
Grey joint rules: elbow/wrist-dominated -> actuator-integration residual,
fixable; hip/knee/ankle+contact-leg dominated -> contact solver family
difference, treat as refuted. Zero-error probe > 2 -> bare-dynamics contact
divergence, treat as refuted.
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

DUMP = "output/remote_ckpt/x1_il_entry_F1_implicit_v4.pt"
N_STEPS = 30


def replay(m, d, qadr, vadr, dump, kp_d, tlim_d, q_tar_seq, n_steps):
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
        q_tar = q_tar_seq[i]
        for _ in range(steps_per_ctrl):
            q = d.qpos[qadr]
            tau = np.clip(kp_d * (q_tar - q), -tlim_d, tlim_d)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        rows.append(d.qvel[vadr] - dump["dof_vel"][i + 1].numpy())
    return rows


def family_means(per_joint):
    fams = {"elbow": [], "wrist": [], "hip_pitch": [], "leg_contact": []}
    for j, v in per_joint.items():
        if "elbow" in j:
            fams["elbow"].append(v)
        elif "wrist" in j:
            fams["wrist"].append(v)
        elif "hip_pitch" in j:
            fams["hip_pitch"].append(v)
        elif ("hip" in j or "knee" in j or "ankle" in j):
            fams["leg_contact"].append(v)
    return {k: float(np.mean(v)) if v else None for k, v in fams.items()}


def main():
    m = mujoco.MjModel.from_xml_path(str(REPO / "data/assets/x1/x1_sim_v4.xml"))
    m.opt.timestep = 1.0 / 120.0
    d = mujoco.MjData(m)
    qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])
    vadr = np.array([m.joint(m.joint(j).id).dofadr[0] for j in X1_DOF_ORDER])

    dump = torch.load(REPO / DUMP, map_location="cpu", weights_only=False)
    meta = dump.get("meta", {})
    print(f"[x1a] meta: isaaclab {meta.get('isaaclab')} | state0 pos dev "
          f"{meta.get('state0_pos_dev'):.2e} vel dev "
          f"{meta.get('state0_vel_dev'):.2e} | kp dev "
          f"{meta.get('kp_dev'):.2e} kd dev {meta.get('kd_dev'):.2e}")

    kp_d = dump["kp"].numpy()
    kd_d = dump["kd"].numpy()
    tlim_d = dump["tlim"].numpy()
    q_tar_seq = dump["q_tar_seq"].numpy()
    m.dof_damping[vadr] = kd_d
    m.dof_frictionloss[vadr] = 0.0

    traj = dump["traj"]
    rows = replay(m, d, qadr, vadr, traj, kp_d, tlim_d, q_tar_seq,
                  min(N_STEPS, traj["dof_pos"].shape[0] - 1))
    diffs = np.stack(rows)
    per_joint = np.abs(diffs).mean(axis=0)
    first_max = float(np.abs(diffs[0]).max())
    mean_rms = float(np.sqrt((diffs ** 2).sum(axis=1).mean()))
    worst_i = int(np.argmax(np.abs(diffs[0])))

    probe = dump.get("probe")
    probe_first = None
    if probe is not None and "q0" in probe:
        q0 = probe["q0"].numpy().astype(np.float64)
        prows = replay(m, d, qadr, vadr, probe, kp_d, tlim_d,
                       np.broadcast_to(q0, (N_STEPS + 2, len(q0))),
                       min(8, probe["dof_pos"].shape[0] - 1))
        probe_first = float(np.abs(np.stack(prows)[0]).max())

    fams = family_means({X1_DOF_ORDER[j]: float(per_joint[j])
                         for j in range(len(X1_DOF_ORDER))})
    print(f"[x1a] F1 first-step max {first_max:.4f} | mean rms {mean_rms:.3f}"
          f" | worst {X1_DOF_ORDER[worst_i]} | probe {probe_first}")
    print(f"[x1a] families: {json.dumps(fams)}")

    # ---- pre-registered verdict ----
    if first_max < 1.0:
        verdict = "H_X1_IL_GREEN"
        detail = f"first-step {first_max:.3f} < 1.0"
    elif first_max >= 2.0:
        verdict = "H_X1_IL_REFUTED"
        detail = (f"first-step {first_max:.3f} >= 2.0 -> MuJoCo-native sole "
                  "primary per preregistered chain")
    else:
        contact_fam = fams["leg_contact"] or 0.0
        arm_fam = max(f for f in (fams["elbow"], fams["wrist"]) if f is not None)
        if probe_first is not None and probe_first > 2.0:
            verdict = "H_X1_IL_REFUTED(grey)"
            detail = (f"grey {first_max:.3f} + zero-error probe "
                      f"{probe_first:.3f} > 2 -> bare contact divergence")
        elif contact_fam > 2.0 * arm_fam:
            verdict = "H_X1_IL_REFUTED(grey)"
            detail = (f"grey {first_max:.3f} + leg/contact family "
                      f"{contact_fam:.2f} >> arm {arm_fam:.2f}")
        else:
            verdict = "H_X1_IL_GREY_ARM"
            detail = (f"grey {first_max:.3f} arm-dominated "
                      f"(arm {arm_fam:.2f} vs leg {contact_fam:.2f}) -> "
                      "actuator-integration residual, fixable path")
    print(f"[x1a] PRE-REGISTERED RULE -> {verdict} ({detail})")

    out = dict(experiment="H-X1-IL-full-entry-probe",
               prereg="SW-PREREG-005 (commit 15151d1a)",
               first_step_max=first_max, mean_rms=mean_rms,
               probe_first=probe_first, worst_joint=X1_DOF_ORDER[worst_i],
               per_joint={X1_DOF_ORDER[j]: float(per_joint[j])
                          for j in range(len(X1_DOF_ORDER))},
               families=fams, verdict=verdict, verdict_detail=detail,
               context=dict(sw_r006_single_joint_S1=0.0937, S2=0.2586,
                            sw_r004_isaac_gym=26.057,
                            sw_r005_newton=19.120))
    (REPO / "output/x1_il_entry_analysis.json").write_text(json.dumps(out, indent=1))
    print("[x1a] saved output/x1_il_entry_analysis.json")


if __name__ == "__main__":
    main()
