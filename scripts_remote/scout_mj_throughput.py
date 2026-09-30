"""MuJoCo-native throughput scout (remote image): version, cores, rollout.

Checks mujoco availability/version on the training image; if mujoco>=3.2
(mujoco.rollout exists), benchmarks threaded batch rollouts on the exact
sim2sim model (x1_sim_v4.xml @dt=1/120) at several nenv sizes. Saves
output/mj_throughput_remote.json.
"""
import json
import multiprocessing
import os
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

info = dict(cpus=multiprocessing.cpu_count())
try:
    import mujoco
    info["mujoco_version"] = mujoco.__version__
except Exception as exc:  # noqa: BLE001
    info["mujoco_import_error"] = str(exc)
    json.dump(info, open("output/mj_throughput_remote.json", "w"), indent=1)
    print("[mj-scout] mujoco unavailable:", exc)
    raise SystemExit(0)

import numpy as np

has_rollout = hasattr(mujoco, "rollout")
info["has_rollout"] = has_rollout
print(f"[mj-scout] mujoco {mujoco.__version__} | cpus {info['cpus']} | "
      f"rollout {has_rollout}", flush=True)

# serial baseline on the remote CPU too (comparability)
m = mujoco.MjModel.from_xml_path("data/assets/x1/x1_sim_v4.xml")
m.opt.timestep = 1 / 120.0
d = mujoco.MjData(m)
for _ in range(100):
    mujoco.mj_step(m, d)
N = 10000
t0 = time.perf_counter()
for _ in range(N):
    mujoco.mj_step(m, d)
sps = N / (time.perf_counter() - t0)
info["serial_steps_per_s"] = sps
info["serial_envs_equiv_30hz"] = sps / 120.0
print(f"[mj-scout] serial: {sps:,.0f} steps/s = {sps/120.0:,.0f} envs-equiv",
      flush=True)

if has_rollout:
    from mujoco import rollout as mj_rollout

    results = {}
    for nenv in (64, 256, 1024, 2048):
        nstep = 480  # 4s of sim @120Hz per rollout call
        try:
            r = mj_rollout.Rollout(nthread=min(nenv, info["cpus"]))
            init_state = np.zeros((nenv, m.nq + m.nv + m.na))
            init_state[:, 2] = 0.6  # root height
            ctrl = np.zeros((nenv, nstep, m.nu))
            state = np.zeros((nenv, nstep, m.nq + m.nv + m.na))
            # warmup once
            r.rollout(model=m, data=[mujoco.MjData(m) for _ in range(1)],
                      initial_state=init_state[:1], control=ctrl[:1, :8])
            t0 = time.perf_counter()
            r.rollout(model=m, data=None, initial_state=init_state,
                      control=ctrl, state=state)
            dt = time.perf_counter() - t0
            steps = nenv * nstep
            results[nenv] = dict(
                wall_s=dt, total_steps=steps,
                steps_per_s=steps / dt,
                envs_equiv_30hz=steps / dt / 120.0)
            print(f"[mj-scout] rollout nenv={nenv}: {steps/dt:,.0f} steps/s "
                  f"= {steps/dt/120.0:,.0f} envs-equiv (wall {dt:.2f}s)",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            results[nenv] = {"error": str(exc)}
            print(f"[mj-scout] rollout nenv={nenv} FAILED: {exc}", flush=True)
    info["rollout"] = results

os.makedirs("output", exist_ok=True)
json.dump(info, open("output/mj_throughput_remote.json", "w"), indent=1)
print("[mj-scout] saved output/mj_throughput_remote.json", flush=True)
