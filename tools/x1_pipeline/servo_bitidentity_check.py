"""Bit-identity smoke (SW native route step 2b): servo training variant vs
harness-style manual replay, same engine, same 30-step q_tar (F1 sequence).

A) x1_train_servo.xml: ctrl=q_tar once per control step, 4x mj_step
B) x1_sim_v4.xml (motor): per substep tau=clip(kp*(tar-q)) -> ctrl, 4x
   mj_step, dof_damping=kd, frictionloss=0 (harness replay semantics)

Expect max |dof_vel diff| ~1e-12 (measured 4.87e-13 over 30
steps; two earlier false alarms: <position> timeconst=0.02
ctrl low-pass filter -> <general dyntype=none> exact mirror;
and a dt mismatch in this comparison script) (same engine, same arithmetic; servo
forcerange == np.clip; damping both implicit).
"""
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402


def load_state(m, d, qadr, vadr, traj):
    d.qpos[:] = 0
    d.qpos[:3] = traj["root_pos"][0].numpy()
    x, y, z, w = traj["root_quat"][0].numpy()
    d.qpos[3:7] = [w, x, y, z]
    d.qpos[qadr] = traj["dof_pos"][0].numpy().astype(np.float64)
    d.qvel[:3] = traj["root_vel"][0].numpy()
    d.qvel[3:6] = traj["root_ang_vel"][0].numpy()
    d.qvel[vadr] = traj["dof_vel"][0].numpy().astype(np.float64)
    mujoco.mj_forward(m, d)


def main():
    ref = torch.load(REPO / "output/remote_ckpt/x1_il_entry_F1_implicit_v4.pt",
                     map_location="cpu", weights_only=False)
    kp = ref["kp"].numpy().astype(np.float64)
    kd = ref["kd"].numpy().astype(np.float64)
    tlim = ref["tlim"].numpy().astype(np.float64)
    q_tar = ref["q_tar_seq"].numpy().astype(np.float64)

    results = {}
    for tag, xml, manual in (("servo", "data/assets/x1/x1_train_servo.xml", False),
                             ("harness", "data/assets/x1/x1_sim_v4.xml", True)):
        m = mujoco.MjModel.from_xml_path(str(REPO / xml))
        m.opt.timestep = 1.0 / 120.0  # x1_sim_v4.xml default dt=0.005!
        d = mujoco.MjData(m)
        qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])
        vadr = np.array([m.joint(m.joint(j).id).dofadr[0] for j in X1_DOF_ORDER])
        if manual:
            m.dof_damping[vadr] = kd
            m.dof_frictionloss[vadr] = 0.0
        load_state(m, d, qadr, vadr, ref["traj"])
        seq = []
        for t in range(30):
            tar = q_tar[t]
            for _ in range(4):
                if manual:
                    q = d.qpos[qadr]
                    d.ctrl[:] = np.clip(kp * (tar - q), -tlim, tlim)
                else:
                    d.ctrl[:] = tar
                mujoco.mj_step(m, d)
            seq.append(d.qvel[vadr].copy())
        results[tag] = np.stack(seq)

    dv = results["servo"] - results["harness"]
    print(f"30-step servo vs harness: max |dof_vel diff| {np.abs(dv).max():.3e}"
          f" | per-step max: {np.round(np.abs(dv).max(axis=1)[:6], 12).tolist()}")
    print(f"VERDICT: {'BIT-IDENTICAL (<1e-9)' if np.abs(dv).max() < 1e-9 else 'DIVERGENT'}")
    out = REPO / "output/servo_harness_bitidentity.json"
    import json
    out.write_text(json.dumps(dict(
        max_abs_dv=float(np.abs(dv).max()),
        bit_identical=bool(np.abs(dv).max() < 1e-9),
        per_step_max=[float(x) for x in np.abs(dv).max(axis=1)]), indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
