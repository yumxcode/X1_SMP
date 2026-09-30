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

# r1 lesson: mujoco.rollout is a SUBMODULE - hasattr(mujoco, "rollout")
# is False until imported; must try explicit import.
try:
    from mujoco import rollout as mj_rollout
    has_rollout = True
except ImportError:
    mj_rollout = None
    has_rollout = False
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

results = {}
info["rollout_class"] = None
try:
    import inspect
    from mujoco import rollout as _rmod
    if hasattr(_rmod, "Rollout"):
        info["rollout_class"] = str(inspect.signature(_rmod.Rollout.rollout))[:500]
except Exception as exc:  # noqa: BLE001
    info["rollout_class_error"] = str(exc)

# Multiprocess benchmark (engine-grade path): each worker owns an MjData and
# runs serial control-step rollouts (servo semantics: ctrl=q_tar + 4x step).
import multiprocessing as mp

_XML = "data/assets/x1/x1_train_servo.xml"

def _worker(args):
    nenv, nctrl = args
    import mujoco
    import numpy as np
    m = mujoco.MjModel.from_xml_path(_XML)
    m.opt.timestep = 1.0 / 120.0
    d = mujoco.MjData(m)
    ctrl = np.zeros(m.nu)
    for _ in range(100):
        mujoco.mj_step(m, d)  # warmup
    import time
    t0 = time.perf_counter()
    steps = 0
    for _ in range(nctrl):
        for _ in range(4):
            d.ctrl[:] = ctrl
            mujoco.mj_step(m, d)
        steps += 4
    return steps / (time.perf_counter() - t0)

if __name__ == "__main__" or True:
    ctx = mp.get_context("fork")
    for nproc in (8, 16, 32, 64):
        nctrl = 3000
        with ctx.Pool(nproc) as pool:
            t0 = time.perf_counter()
            rates = pool.map(_worker, [(1, nctrl)] * nproc)
            wall = time.perf_counter() - t0
        total = sum(rates)
        results[f"mp_{nproc}"] = dict(
            total_substeps_per_s=total,
            envs_equiv_30hz=total / 120.0,
            wall_s=wall)
        print(f"[mj-scout] multiprocess nproc={nproc}: {total:,.0f} substeps/s "
              f"= {total/120.0:,.0f} envs-equiv (per-proc mean "
              f"{total/nproc:,.0f})", flush=True)
    info["multiprocess"] = results

os.makedirs("output", exist_ok=True)
json.dump(info, open("output/mj_throughput_remote.json", "w"), indent=1)
print("[mj-scout] saved output/mj_throughput_remote.json", flush=True)
