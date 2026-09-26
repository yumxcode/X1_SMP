"""Render a MuJoCo sim2sim policy rollout to MP4 (fall evidence / demos)."""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

import sim2sim_validate as SV


def main():
    ckpt = sys.argv[1]
    out = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    length = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
    import mujoco
    import imageio

    pol = SV.Policy(ckpt)
    sim = SV.Sim2Sim(pol)
    sim.reset(seed)
    m, d = sim.m, sim.d
    OPT = mujoco.MjvOption()
    for g in range(6):
        OPT.geomgroup[g] = 1
    r = mujoco.Renderer(m, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ctrl = 1.0 / 30.0
    steps = int(round(ctrl / m.opt.timestep))
    writer = imageio.get_writer(out, fps=30, macro_block_size=8)
    for it in range(int(length / ctrl)):
        o = sim.obs()
        a = pol.forward(o)
        q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
        for _ in range(steps):
            q = d.qpos[sim.qadr]
            tau = np.clip(sim.kp * (q_tar - q), -sim.eff, sim.eff)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        cam.lookat[:] = d.qpos[:3]
        cam.lookat[2] += 0.1
        cam.distance = 2.6
        yaw = np.degrees(np.arctan2(
            d.qvel[0], d.qvel[1])) if np.linalg.norm(d.qvel[:2]) > 0.1 else 0
        cam.azimuth = yaw + 180.0
        cam.elevation = -18.0
        r.update_scene(d, camera=cam, scene_option=OPT)
        writer.append_data(np.asarray(r.render())[..., :3])
    writer.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
