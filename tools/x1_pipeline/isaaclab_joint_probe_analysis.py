"""SW-PREREG-004 analysis: Isaac Lab single-joint probe verdict.

Inputs: output/remote_ckpt/isaaclab_joint_probe_{tag}.pt for
  S1_lowkd_implicit, S2_v4kd_implicit, S3_lowkd_explicit
(dumped by scripts_remote/probe_isaaclab_joint.py, remote task).

MuJoCo mirror (harness pdx2-mirror semantics): identical rod inertial
(mass 0.833, com -0.245 z, diaginertia 0.01667/0.01667/0.000028),
hinge axis x, armature 0.01, gravity -9.81, dt 1/120 Euler.
  implicit configs: per-substep tau = kp*(tar - q), dof_damping = kd
  explicit config:  per-substep tau = kp*(tar - q) - kd*qd, damping = 0

Metric per config: first control step |qd_MJ - qd_IL|, max over the 3
q0 offsets; zero-error variant (target = q0) same.

Pre-registered verdict (SW-PREREG-004, commit 1212f42a):
  S1 & S2 both < 1.0            -> H-IL_GREEN (unlock asset migration)
  any of S1/S2 >= 2.0           -> H-IL_REFUTED (MuJoCo-native route sole primary)
  grey (1.0-2.0)                -> joint with S3: if S3 > 2x worse than
                                   S1/S2 and S1/S2 < 1.5 -> conditional
                                   continue (needs full-X1 probe); else refuted
"""
import json
from pathlib import Path

import numpy as np
import torch

import mujoco

REPO = Path(__file__).resolve().parents[2]

TAGS = ["S1_lowkd_implicit", "S2_v4kd_implicit", "S3_lowkd_explicit"]
Q0_OFFSETS = [0.2, -0.35, 0.6]
N_CTRL = 8
SUBSTEPS = 4
DT = 1.0 / 120.0

MJCF = """
<mujoco>
  <option timestep="{dt}" gravity="0 0 -9.81" integrator="Euler"/>
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="1 0 0" armature="0.01"
             damping="{damping}"/>
      <inertial pos="0 0 -0.245" mass="0.833"
                diaginertia="0.01667 0.01667 0.000028"/>
      <geom type="box" size="0.015 0.015 0.245" pos="0 0 -0.245"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def mj_run(q0, target, kp, kd, damping):
    m = mujoco.MjModel.from_xml_string(MJCF.format(dt=DT, damping=damping))
    d = mujoco.MjData(m)
    d.qpos[0] = q0
    d.qvel[0] = 0.0
    mujoco.mj_forward(m, d)
    qd_seq = []
    for _ in range(N_CTRL):
        for _sub in range(SUBSTEPS):
            q, qd = d.qpos[0], d.qvel[0]
            tau = kp * (target - q)
            if damping == 0.0:  # explicit-kd config
                tau = tau - kd * qd
            d.qfrc_applied[0] = tau
            mujoco.mj_step(m, d)
        qd_seq.append(float(d.qvel[0]))
    return qd_seq


def main():
    dumps, failed = {}, []
    for tag in TAGS:
        p = REPO / "output/remote_ckpt" / f"isaaclab_joint_probe_{tag}.pt"
        if not p.exists():
            failed.append(tag)
            continue
        dumps[tag] = torch.load(p, map_location="cpu", weights_only=False)
    print(f"[il] loaded {len(dumps)}/{len(TAGS)}; missing: {failed}")

    results = {}
    for tag in sorted(dumps):
        dump = dumps[tag]
        meta = dump.get("meta", {})
        kp, kd = float(meta.get("kp", 40.0)), float(meta.get("kd", 0.5))
        mode = meta.get("mode", "implicit")
        damping = kd if mode == "implicit" else 0.0
        masses = dump.get("masses", [])
        arm_mass = masses[-1] if masses else None
        first_step, zero_first = [], None
        per_offset = {}
        for q0 in Q0_OFFSETS:
            il = np.asarray(dump["qd_step"][str(q0)], dtype=np.float64)
            mj = np.asarray(mj_run(q0, 0.0, kp, kd, damping))
            diff0 = abs(mj[0] - il[0])
            first_step.append(diff0)
            per_offset[str(q0)] = dict(
                il_qd=[round(float(x), 4) for x in il[:4]],
                mj_qd=[round(float(x), 4) for x in mj[:4]],
                first_step_diff=diff0,
                max_all_steps=float(np.max(np.abs(mj - il))))
        il_z = np.asarray(dump["qd_zero"], dtype=np.float64)
        mj_z = np.asarray(mj_run(Q0_OFFSETS[0], Q0_OFFSETS[0], kp, kd, damping))
        zero_first = abs(mj_z[0] - il_z[0])
        results[tag] = dict(
            meta=meta, arm_mass=arm_mass,
            first_step_max=max(first_step),
            zero_error_first=zero_first,
            per_offset=per_offset)
        print(f"[il] {tag:18s} first-step max {max(first_step):7.4f} | "
              f"zero-err {zero_first:7.4f} | arm mass {arm_mass}")

    # ---- pre-registered verdict ----
    verdict, detail = "inconclusive", ""
    s1 = results.get("S1_lowkd_implicit")
    s2 = results.get("S2_v4kd_implicit")
    s3 = results.get("S3_lowkd_explicit")
    if s1 is not None and s2 is not None:
        v1, v2 = s1["first_step_max"], s2["first_step_max"]
        worst = max(v1, v2)
        if v1 < 1.0 and v2 < 1.0:
            verdict = "H-IL_GREEN"
            detail = f"S1 {v1:.3f} / S2 {v2:.3f} both < 1.0"
        elif v1 >= 2.0 or v2 >= 2.0:
            verdict = "H-IL_REFUTED"
            detail = f"S1 {v1:.3f} / S2 {v2:.3f} (>=2.0 present)"
        else:
            if s3 is not None and s3["first_step_max"] > 2.0 * worst \
                    and worst < 1.5:
                verdict = "H-IL_GREY_CONDITIONAL"
                detail = (f"S1 {v1:.3f} / S2 {v2:.3f} grey-low, S3 "
                          f"{s3['first_step_max']:.3f} >2x worse -> kd "
                          "semantics fixable, needs full-X1 probe")
            else:
                verdict = "H-IL_REFUTED"
                detail = (f"S1 {v1:.3f} / S2 {v2:.3f} grey without S3 "
                          "rescue -> treated as refuted per prereg")
        if s3 is not None:
            detail += f"; S3 {s3['first_step_max']:.3f} (kd axis)"
    else:
        detail = "S1/S2 missing"
    print(f"[il] PRE-REGISTERED RULE -> {verdict} ({detail})")

    out = dict(experiment="IDEA-010r-isaaclab-joint-probe",
               prereg="SW-PREREG-004 (commit 1212f42a)", failed=failed,
               results=results, verdict=verdict, verdict_detail=detail,
               context=dict(sw_r004_isaac_gym_anchor_first=26.057,
                            sw_r005_newton_B_first=19.120,
                            note="single-joint gravity+dynamics probe, "
                                 "no contact, no policy"))
    (REPO / "output/isaaclab_joint_probe_analysis.json").write_text(
        json.dumps(out, indent=1))
    print("[il] saved output/isaaclab_joint_probe_analysis.json")


if __name__ == "__main__":
    main()
