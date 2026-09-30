"""SW-PREREG-003 analysis: Newton entry probe (IDEA-012r) verdict.

Inputs: output/remote_ckpt/newton_probe_{tag}.pt for
  A_explicit, B_implicit, C_pos, D_impratio1
(dumped by scripts_remote/probe_newton_entry.py, remote task).

Per config: MuJoCo harness replay of the SAME open-loop q_tar sequence
(from the r4 GPU anchor dump) -> first-step max |dof_vel diff|, mean rms
(truncated at first done), 29-joint mean |dv|, zero-error probe replay.
Plus direct Newton-vs-Isaac first-step diff for context.

Pre-registered verdict (SW-PREREG-003, commit b0dbbfc1):
  B_implicit first-step max < 1.0      -> Newton route GREEN
  B_implicit 1.0 <= x < 5.0            -> grey (warp parity investigation)
  B_implicit >= 5.0                    -> Newton downgraded
Side rules: A vs B >2x -> explicit-kd artifact confirmed; A~=B -> kd axis
not decisive at v4 gains. D vs B -> impratio contribution. Zero-error
probe: >2 -> constraint/gravity path also differs; <0.5 while rollout is
large -> command-tracking path.
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

TAGS = ["A_explicit", "B_implicit", "C_pos", "D_impratio1"]
ANCHOR = "output/remote_ckpt/isaac_traj_v4_solver_tgs4_0_gpu.pt"
N_STEPS = 30


def replay(m, d, qadr, vadr, dump, kp_d, tlim_d, q_tar_seq, n_steps,
           start_index=0):
    """Replay control steps in MuJoCo; return per-step diffs."""
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
        q_tar = q_tar_seq[start_index + i]
        for _ in range(steps_per_ctrl):
            q = d.qpos[qadr]
            tau = np.clip(kp_d * (q_tar - q), -tlim_d, tlim_d)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        rows.append(d.qvel[vadr] - dump["dof_vel"][i + 1].numpy())
    return rows


def main():
    m = mujoco.MjModel.from_xml_path(str(REPO / "data/assets/x1/x1_sim_v4.xml"))
    m.opt.timestep = 1.0 / 120.0
    d = mujoco.MjData(m)
    qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])
    vadr = np.array([m.joint(m.joint(j).id).dofadr[0] for j in X1_DOF_ORDER])

    ref = torch.load(REPO / ANCHOR, map_location="cpu", weights_only=False)
    isaac_dv1 = ref["dof_vel"][1].numpy()

    dumps, failed = {}, []
    for tag in TAGS:
        p = REPO / "output/remote_ckpt" / f"newton_probe_{tag}.pt"
        if not p.exists():
            failed.append(tag)
            continue
        dumps[tag] = torch.load(p, map_location="cpu", weights_only=False)
    print(f"[np] loaded {len(dumps)}/{len(TAGS)}; missing: {failed}")

    kp_d = ref["kp"].numpy()
    kd_d = ref["kd"].numpy()
    tlim_d = ref["tlim"].numpy()
    q_tar_seq = ref["q_tar"].numpy()
    m.dof_damping[vadr] = kd_d       # implicit kd harness semantics
    m.dof_frictionloss[vadr] = 0.0

    results = {}
    for tag in sorted(dumps):
        dump = dumps[tag]
        chk = dump.get("state0_check", {})
        ok_state0 = (chk.get("dof_max_diff", 1.0) < 1e-5
                     and chk.get("root_max_diff", 1.0) < 1e-5)
        rows = replay(m, d, qadr, vadr, dump, kp_d, tlim_d, q_tar_seq,
                      min(N_STEPS, dump["dof_pos"].shape[0] - 1))
        done_flags = dump.get("done")
        n_valid, reset_at = len(rows), None
        if done_flags is not None:
            dn = done_flags.numpy().reshape(-1)
            nz = np.nonzero(dn)[0]
            if len(nz):
                reset_at = int(nz[0])
                n_valid = min(n_valid, reset_at)
        diffs = np.stack(rows[:n_valid])
        per_joint = np.abs(diffs).mean(axis=0)
        first_max = float(np.abs(diffs[0]).max())
        mean_rms = float(np.sqrt((diffs ** 2).sum(axis=1).mean()))
        # direct Newton-vs-Isaac first-step (context)
        direct_first = float(np.max(np.abs(
            dump["dof_vel"][1].numpy() - isaac_dv1)))
        # zero-error probe replay
        probe = dump.get("probe")
        probe_first = None
        if probe is not None and "q0" in probe:
            q0 = probe["q0"].numpy().astype(np.float64)
            prows = replay(m, d, qadr, vadr, probe, kp_d, tlim_d,
                           np.broadcast_to(q0, (N_STEPS + 2, len(q0))),
                           min(8, probe["dof_pos"].shape[0] - 1))
            probe_first = float(np.abs(np.stack(prows)[0]).max())
        results[tag] = dict(
            config=dump.get("probe_config", {}), state0_ok=ok_state0,
            state0_check=chk, n_valid=n_valid, reset_at=reset_at,
            first_step_max=first_max, mean_rms=mean_rms,
            direct_vs_isaac_first=direct_first,
            probe_first_step_max=probe_first,
            worst_joint=X1_DOF_ORDER[int(np.argmax(np.abs(diffs[0])))],
            per_joint={X1_DOF_ORDER[j]: float(per_joint[j])
                       for j in range(len(X1_DOF_ORDER))},
        )
        print(f"[np] {tag:12s} first {first_max:7.3f} | rms {mean_rms:6.3f} "
              f"| vs-isaac {direct_first:7.3f} | probe {probe_first} "
              f"| worst {results[tag]['worst_joint']} | n_valid {n_valid} "
              f"| state0 {'OK' if ok_state0 else 'FAIL'}")

    # ---- pre-registered verdict ----
    verdict, detail = "inconclusive", ""
    b = results.get("B_implicit")
    if b is not None and b["state0_ok"]:
        v = b["first_step_max"]
        if v < 1.0:
            verdict = "NEWTON_GREEN"
        elif v < 5.0:
            verdict = "GREY_warp_parity"
        else:
            verdict = "NEWTON_DOWNGRADED"
        detail = f"B_implicit first-step max {v:.3f} rad/s"
        a = results.get("A_explicit")
        if a is not None and a["state0_ok"]:
            r = a["first_step_max"] / max(v, 1e-9)
            detail += f"; A/B ratio {r:.2f} ({'kd artifact confirmed' if r > 2 else 'kd axis not decisive at v4 gains'})"
        dd = results.get("D_impratio1")
        if dd is not None and dd["state0_ok"]:
            detail += f"; D(impratio1) first {dd['first_step_max']:.3f} (delta {dd['first_step_max'] - v:+.3f})"
    else:
        detail = "B_implicit missing or state0 check failed"
    print(f"[np] PRE-REGISTERED RULE -> {verdict} ({detail})")

    out = dict(experiment="IDEA-012r-Newton-entry-probe",
               prereg="SW-PREREG-003 (commit b0dbbfc1)", failed=failed,
               results=results, verdict=verdict, verdict_detail=detail,
               isaac_context=dict(
                   note="SW-R004 r4 anchor (TGS 4/0 GPU) same protocol:",
                   first_step_max=26.057, probe_first=20.024))
    (REPO / "output/newton_probe_analysis.json").write_text(json.dumps(out, indent=1))
    print("[np] saved output/newton_probe_analysis.json")


if __name__ == "__main__":
    main()
