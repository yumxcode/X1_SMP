"""MuJoCo-native throughput scout (local, serial baseline).

Serial mj_step rate on x1_sim_v4.xml @dt=1/120 (the exact sim2sim model),
plus the 30Hz-control-loop variant (PD torque recompute every 4 substeps,
training-like). Establishes the lower bound for the native engine budget;
parallelism (mujoco.rollout on 3.2+ or multiprocess) is evaluated on the
remote training image next.
"""
import multiprocessing
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools/x1_pipeline"))
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402


def main():
    m = mujoco.MjModel.from_xml_path(str(REPO / "data/assets/x1/x1_sim_v4.xml"))
    m.opt.timestep = 1 / 120.0
    d = mujoco.MjData(m)
    qadr = np.array([m.joint(j).qposadr[0] for j in X1_DOF_ORDER])
    d.qpos[qadr] = 0.3
    mujoco.mj_forward(m, d)
    for _ in range(100):
        mujoco.mj_step(m, d)

    N = 20000
    t0 = time.perf_counter()
    for _ in range(N):
        mujoco.mj_step(m, d)
    sps = N / (time.perf_counter() - t0)
    print(f"serial mj_step: {sps:,.0f} steps/s | "
          f"envs-equivalent (30Hz ctrl, 4 substeps): {sps * 1 / 120:,.0f}")

    Nc = 2000
    t0 = time.perf_counter()
    for _ in range(Nc):
        for _ in range(4):
            q = d.qpos[qadr]
            d.ctrl[:] = np.clip(100.0 * (0.1 - q), -50, 50)
            mujoco.mj_step(m, d)
    ctrl_rate = Nc * 4 / (time.perf_counter() - t0)
    print(f"with 120Hz PD recompute: {ctrl_rate:,.0f} substeps/s = "
          f"{ctrl_rate * 1 / 120:,.0f} envs-equivalent")
    print("cpu count:", multiprocessing.cpu_count())


if __name__ == "__main__":
    main()
