"""Migration scout: check IsaacLab MJCF/URDF spawner availability + X1 direct-MJCF load smoke (IDEA-010r GREEN branch step 1)."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True, offscreen_render=False)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402

print("[scout] isaaclab", getattr(__import__("isaaclab"), "__version__", "?"), flush=True)

try:
    from isaaclab.sim.spawners.from_files import UrdfFileCfg  # noqa: F401
    print("[scout] UrdfFileCfg: OK", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[scout] UrdfFileCfg: FAIL {exc}", flush=True)

try:
    from isaaclab.sim.spawners.from_files import MjcfFileCfg  # noqa: F401
    print("[scout] MjcfFileCfg: OK", flush=True)
except Exception as exc:  # noqa: BLE001
    print(f"[scout] MjcfFileCfg: FAIL {exc}", flush=True)

# try direct X1 MJCF load if MjcfFileCfg exists
try:
    from isaaclab.sim.spawners.from_files import MjcfFileCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        device="cuda:0", dt=1.0 / 120.0, gravity=(0.0, 0.0, -9.81)))
    art_cfg = ArticulationCfg(
        prim_path="/World/x1",
        spawn=MjcfFileCfg(asset_path="data/assets/x1/x1_v4.xml",
                          fix_base=False),
        actuators={"joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
    x1 = Articulation(art_cfg)
    sim.reset()
    n_joints = x1.num_joints
    names = [x1.joint_names[i] for i in range(min(n_joints, 8))]
    print(f"[scout] X1 MJCF loaded: joints {n_joints} first {names}",
          flush=True)
    dof_lim = x1.root_physx_view.get_dof_limits()
    print(f"[scout] dof_limits shape {tuple(dof_lim.shape)}", flush=True)
except Exception as exc:  # noqa: BLE001
    import traceback
    print("[scout] X1 MJCF direct load: FAIL", flush=True)
    traceback.print_exc()

simulation_app.close()
