"""Render a MuJoCo sim2sim policy rollout to MP4 with the REAL X1 meshes.

Unlike render_sim2sim.py (which renders data/assets/x1_sim.xml — primitive
capsules + sole boxes), this poses the ORIGINAL X1_29DOF mjcf (URDF meshes,
what the user's eyeball validation uses) from the sim2sim harness state:
same physics (x1_sim.xml), mesh-only visuals.

Usage: python render_sim2sim_mesh.py <ckpt> <out.mp4> [seed] [len_s] [--env <env.yaml>]
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))

X1_SRC_MJCF = REPO / "X1_29DOF/mjcf/xyber_x1_flat.xml"
from retarget_v3 import X1_DOF_ORDER  # noqa: E402


def main():
    args = sys.argv[1:]
    ckpt = args[0]
    out = args[1]
    seed = int(args[2]) if len(args) > 2 else 0
    length = float(args[3]) if len(args) > 3 else 4.0
    env = "data/envs/smp_x1_env_v3.yaml"
    if "--env" in args:
        env = args[args.index("--env") + 1]

    import mujoco
    import imageio
    import sim2sim_validate as SV

    SV.Sim2Sim.ENV_YAML_REL = env
    pol = SV.Policy(ckpt)
    sim = SV.Sim2Sim(pol)          # physics: fixed-asset x1_sim.xml
    sim.reset(seed)
    m_sim, d_sim = sim.m, sim.d

    # visual model: original mjcf with URDF meshes, posed by joint name
    m_mesh = mujoco.MjModel.from_xml_path(str(X1_SRC_MJCF))
    # FIX: the mjcf's floor plane only covers |y|<=3 m while sim2sim resets
    # sit at arbitrary world coords (e.g. y~4) — extend the visual floor so
    # the robot isn't rendered hovering over a void (misdiagnosed as physics
    # failure by the video reviewer on 2026-09-28)
    for g in range(m_mesh.ngeom):
        if (m_mesh.geom(g).name or '') == 'floor':
            m_mesh.geom_size[g][1] = 200.0
    d_mesh = mujoco.MjData(m_mesh)
    xadr = np.array([m_mesh.joint(j).qposadr[0] for j in X1_DOF_ORDER])

    OPT = mujoco.MjvOption()
    for g in range(6):
        OPT.geomgroup[g] = 1
    r = mujoco.Renderer(m_mesh, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    ctrl = 1.0 / 30.0
    steps = int(round(ctrl / m_sim.opt.timestep))
    writer = imageio.get_writer(out, fps=30, macro_block_size=8)
    for it in range(int(length / ctrl)):
        o = sim.obs()
        a = pol.forward(o)
        q_tar = np.clip(a, -sim.a_bound, sim.a_bound)
        for _ in range(steps):
            q = d_sim.qpos[sim.qadr]
            tau = np.clip(sim.kp * (q_tar - q), -sim.eff, sim.eff)
            d_sim.ctrl[:] = tau
            mujoco.mj_step(m_sim, d_sim)
        mujoco.mj_forward(m_sim, d_sim)

        # pose the mesh model identically
        d_mesh.qpos[:] = 0
        d_mesh.qpos[:3] = d_sim.qpos[:3]
        d_mesh.qpos[3:7] = d_sim.qpos[3:7]
        d_mesh.qpos[xadr] = d_sim.qpos[sim.qadr]
        mujoco.mj_forward(m_mesh, d_mesh)

        cam.lookat[:] = d_mesh.qpos[:3]
        cam.lookat[2] += 0.1
        cam.distance = 2.6
        v = d_sim.qvel[:2]
        yaw = np.degrees(np.arctan2(v[0], v[1])) if np.linalg.norm(v) > 0.1 else 0
        cam.azimuth = yaw + 180.0
        cam.elevation = -18.0
        r.update_scene(d_mesh, camera=cam, scene_option=OPT)
        writer.append_data(np.asarray(r.render())[..., :3])
    writer.close()
    print(f"saved {out} (mesh-rendered, seed {seed}, {length:.1f}s)")


if __name__ == "__main__":
    main()
