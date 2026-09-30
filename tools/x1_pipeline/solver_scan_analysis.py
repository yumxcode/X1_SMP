"""E-ISAAC-SOLVER-01 (SW-PREREG-002): analyze the PhysX solver-family scan.

Inputs: output/remote_ckpt/isaac_traj_v4_solver_{tag}.pt for
  tgs4_0_gpu (anchor), tgs4_0_cpu, tgs32_0_cpu, tgs4_8_cpu,
  pgs4_0_cpu, pgs32_0_cpu, pgs4_8_cpu
(dumped by scripts_remote/dump_traj_v4_solver.py, TASK_20260930_031).

Per config: MuJoCo replay of 30 control steps (identical algorithm to
v4_resid_dump_analysis.py) -> first-step max |dof_vel diff|, 30-step mean
rms, 29-joint mean |dv| table, zero-error probe replay diff.

Verdict bands pre-registered in SW-PREREG-002 (commit a703fbb4, before task
creation):
  any valid CPU config first-step max < 2.0            -> H2' supported
  all valid CPU configs >= 6.0                         -> H2'' supported
  else grey (joint per-joint pattern + zero-error probe)
Anchor validity: tgs4_0_gpu first-step max within 25.44 +/-30% (TASK_20260929_165).
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

TAGS = ["tgs4_0_gpu", "tgs4_0_cpu", "tgs32_0_cpu", "tgs4_8_cpu",
        "pgs4_0_cpu", "pgs32_0_cpu", "pgs4_8_cpu"]
CPU_TAGS = [t for t in TAGS if t.endswith("_cpu")]
ANCHOR = "tgs4_0_gpu"
ANCHOR_REF = 25.44          # TASK_20260929_165 first-step max, same ckpt
ANCHOR_TOL = 0.30           # +/-30% validity band (17.81 .. 33.07)
N_STEPS = 30


def replay_rollout(m, d, qadr, vadr, dump, kp_d, tlim_d, q_tar_override=None):
    """Replay control steps in MuJoCo; return list of per-step diff rows."""
    steps_per_ctrl = int(round((1 / 30.0) / m.opt.timestep))
    rows = []
    for i in range(min(N_STEPS, dump["dof_pos"].shape[0] - 1)):
        d.qpos[:] = 0
        d.qpos[:3] = dump["root_pos"][i].numpy()
        x, y, z, w = dump["root_quat"][i].numpy()
        d.qpos[3:7] = [w, x, y, z]
        d.qpos[qadr] = dump["dof_pos"][i].numpy()
        d.qvel[:3] = dump["root_vel"][i].numpy()
        d.qvel[3:6] = dump["root_ang_vel"][i].numpy()
        d.qvel[vadr] = dump["dof_vel"][i].numpy()
        mujoco.mj_forward(m, d)
        q_tar = (q_tar_override if q_tar_override is not None
                 else dump["q_tar"][i].numpy())
        for _ in range(steps_per_ctrl):
            q = d.qpos[qadr]
            tau = np.clip(kp_d * (q_tar - q), -tlim_d, tlim_d)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        diff = d.qvel[vadr] - dump["dof_vel"][i + 1].numpy()
        rows.append(diff)
    return rows


def main():
    m = mujoco.MjModel.from_xml_path(str(REPO / "data/assets/x1/x1_sim_v4.xml"))
    m.opt.timestep = 1.0 / 120.0
    d = mujoco.MjData(m)
    qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])
    vadr = np.array([m.joint(m.joint(j).id).dofadr[0] for j in X1_DOF_ORDER])

    dumps, failed = {}, []
    for tag in TAGS:
        p = REPO / "output/remote_ckpt" / f"isaac_traj_v4_solver_{tag}.pt"
        if not p.exists():
            failed.append(tag)
            continue
        try:
            dumps[tag] = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[scan] load FAILED {tag}: {exc}")
            failed.append(tag)
    print(f"[scan] loaded {len(dumps)}/{len(TAGS)} configs; missing/failed: {failed}")

    # ---- obs[0] cross-config consistency (validity self-check) ----
    obs0 = {t: v["obs"][0].numpy().astype(np.float64) for t, v in dumps.items()}
    valid = set(dumps)
    if ANCHOR in obs0:
        for t, o in obs0.items():
            if t == ANCHOR:
                continue
            dev = float(np.max(np.abs(o - obs0[ANCHOR])))
            ok = dev < 1e-4
            print(f"[scan] obs0[{t}] vs anchor: max diff {dev:.2e} "
                  f"{'OK' if ok else 'RESET-MISMATCH'}")
            if not ok:
                valid.discard(t)
    for t in dumps:
        arr = dumps[t]["dof_vel"].numpy()
        if not np.isfinite(arr).all():
            print(f"[scan] {t}: NON-FINITE dof_vel -> excluded")
            valid.discard(t)
    print(f"[scan] valid configs for verdict: {sorted(valid)}")

    # ---- per-config replay metrics ----
    results = {}
    for tag in sorted(valid):
        dump = dumps[tag]
        kp_d = dump["kp"].numpy()
        kd_d = dump["kd"].numpy()
        tlim_d = dump["tlim"].numpy()
        m.dof_damping[vadr] = kd_d
        m.dof_frictionloss[vadr] = 0.0
        rows = replay_rollout(m, d, qadr, vadr, dump, kp_d, tlim_d)
        # r2 lesson: early-termination resets break state continuity; the
        # diff at step i needs state i+1, so truncate after the first done.
        done_flags = dump.get("done")
        n_valid = len(rows)
        reset_at = None
        if done_flags is not None:
            dn = done_flags.numpy().reshape(-1)
            nz = np.nonzero(dn)[0]
            if len(nz):
                reset_at = int(nz[0])
                n_valid = min(n_valid, reset_at)
        diffs = np.stack(rows[:n_valid])          # [n_valid, 29]
        per_joint = np.abs(diffs).mean(axis=0)     # [29]
        first_max = float(np.abs(diffs[0]).max())
        mean_rms = float(np.sqrt((diffs ** 2).sum(axis=1).mean()))
        overall_max = float(np.abs(diffs).max())
        # zero-error probe replay
        probe = dump.get("probe")
        probe_first = probe_mean = None
        if probe is not None and "q0" in probe:
            q0 = probe["q0"].numpy().astype(np.float64)
            prows = replay_rollout(m, d, qadr, vadr, probe, kp_d, tlim_d,
                                   q_tar_override=q0)
            pdiffs = np.stack(prows)
            probe_first = float(np.abs(pdiffs[0]).max())
            probe_mean = float(np.sqrt((pdiffs ** 2).sum(axis=1).mean()))
        results[tag] = dict(
            solver_config=dump.get("solver_config", {}),
            obs0_hash=dump.get("obs0_hash", ""),
            n_valid=n_valid, reset_at_ctrl_step=reset_at,
            first_step_max=first_max, mean_rms=mean_rms,
            overall_max=overall_max,
            probe_first_step_max=probe_first, probe_mean_rms=probe_mean,
            per_joint={X1_DOF_ORDER[j]: float(per_joint[j])
                       for j in range(len(X1_DOF_ORDER))},
            worst_joint=X1_DOF_ORDER[int(np.argmax(np.abs(diffs[0])))],
        )
        print(f"[scan] {tag:12s} first {first_max:7.3f} | rms {mean_rms:6.3f} "
              f"| max {overall_max:7.3f} | worst {results[tag]['worst_joint']}"
              f" | n_valid {n_valid}{' (reset@' + str(reset_at) + ')' if reset_at is not None else ''}"
              f" | probe first {probe_first if probe_first is not None else float('nan'):7.3f}")

    # ---- anchor validity ----
    anchor_ok = None
    if ANCHOR in results:
        a = results[ANCHOR]["first_step_max"]
        anchor_ok = abs(a - ANCHOR_REF) / ANCHOR_REF <= ANCHOR_TOL
        print(f"[scan] anchor {ANCHOR}: first {a:.3f} vs ref {ANCHOR_REF} "
              f"(+/-30%): {'OK' if anchor_ok else 'OUT-OF-BAND'}")

    # ---- pre-registered verdict (SW-PREREG-002) ----
    cpu_valid = [t for t in CPU_TAGS if t in results]
    verdict = "inconclusive"
    detail = ""
    if cpu_valid:
        best = min(results[t]["first_step_max"] for t in cpu_valid)
        worst = max(results[t]["first_step_max"] for t in cpu_valid)
        if best < 2.0:
            verdict = "H2'_supported"
            detail = f"best cpu config first-step max {best:.3f} < 2.0"
        elif worst >= 6.0 and len(cpu_valid) == len(CPU_TAGS):
            verdict = "H2''_supported"
            detail = f"all {len(cpu_valid)} cpu configs >= 6.0 (max {worst:.3f})"
        else:
            verdict = "grey"
            detail = f"cpu range [{best:.3f}, {worst:.3f}]"
        if anchor_ok is False:
            verdict += "+ANCHOR_INCONSISTENT"
            detail += "; anchor out of band -> downgrade per prereg"
    print(f"[scan] PRE-REGISTERED RULE -> {verdict} ({detail})")

    out = dict(experiment="E-ISAAC-SOLVER-01",
               prereg="SW-PREREG-002 (commit a703fbb4)",
               task="TASK_20260930_031", failed=failed,
               valid=sorted(valid), anchor_ok=anchor_ok,
               results=results, verdict=verdict, verdict_detail=detail,
               prereg_bands="any_cpu<2.0->H2' / all_cpu>=6.0->H2'' / grey")
    (REPO / "output/solver_scan_analysis.json").write_text(json.dumps(out, indent=1))
    print("[scan] saved output/solver_scan_analysis.json")


if __name__ == "__main__":
    main()
