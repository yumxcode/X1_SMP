"""IDEA-010r / SW-PREREG-004: Isaac Lab (PhysX 5) single-joint entry probe.

Standalone isaaclab script (no mimickit engine, no USD pipeline): spawns a
1-DOF pendulum from a runtime-written URDF via UrdfFileCfg, parameters
representative of the X1 arm chain (kp=40, kd in {0.5, 2.0}, armature 0.01,
rod 0.833 kg @ 0.245 m -> I_joint ~= 0.0667 + armature), gravity -9.81.

Config matrix (per SW-PREREG-004, frozen):
  S1_lowkd_implicit  ImplicitActuator kp=40 kd=0.5   (MAIN)
  S2_v4kd_implicit   ImplicitActuator kp=40 kd=2.0   (MAIN)
  S3_lowkd_explicit  per-substep explicit kp*(tar-q)-kd*qd (kd axis control)

Per config: 3 q0 offsets {0.2, -0.35, 0.6} rad step response (target 0) +
zero-error variant (target = q0), 8 control steps x 4 substeps @ dt=1/120.
Records joint velocity at each control-step boundary (forced copies).

Driver loops configs as fresh subprocesses (one sim per process).
Outputs: output/isaaclab_joint_probe_{tag}.pt
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SELF = os.path.abspath(__file__)
os.chdir(ROOT)

CONFIGS = [
    ("S1_lowkd_implicit", "implicit", 40.0, 0.5),
    ("S2_v4kd_implicit", "implicit", 40.0, 2.0),
    ("S3_lowkd_explicit", "explicit", 40.0, 0.5),
]
Q0_OFFSETS = [0.2, -0.35, 0.6]
N_CTRL = 8
SUBSTEPS = 4
DT = 1.0 / 120.0

URDF_TEXT = """<?xml version="1.0"?>
<robot name="x1_pend">
  <link name="base">
    <inertial><origin xyz="0 0 0"/><mass value="0.001"/>
      <inertia ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/></inertial>
  </link>
  <link name="arm">
    <inertial><origin xyz="0 0 -0.245"/><mass value="0.833"/>
      <inertia ixx="0.01667" ixy="0" ixz="0" iyy="0.01667" iyz="0" izz="0.000028"/>
    </inertial>
    <visual><origin xyz="0 0 -0.245"/><geometry><box size="0.03 0.03 0.49"/></geometry></visual>
    <collision><origin xyz="0 0 -0.245"/><geometry><box size="0.03 0.03 0.49"/></geometry></collision>
  </link>
  <joint name="hinge" type="revolute">
    <parent link="base"/><child link="arm"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="1000" velocity="100"/>
    <dynamics damping="0.0" friction="0.0"/>
  </joint>
</robot>
"""


def run_child(tag, mode, kp, kd):
    env = dict(os.environ)
    env.update({"X1_IL_CHILD": "1", "X1_IL_TAG": tag, "X1_IL_MODE": mode,
                "X1_IL_KP": str(kp), "X1_IL_KD": str(kd)})
    print(f"[il-probe] === {tag}: mode={mode} kp={kp} kd={kd} ===", flush=True)
    proc = subprocess.run([sys.executable, SELF], env=env,
                          capture_output=True, text=True, timeout=1500)
    out = proc.stdout + proc.stderr
    for ln in out.splitlines():
        if ("[il-probe]" in ln or "Traceback" in ln or "Error" in ln
                or "ImportError" in ln or "ModuleNotFound" in ln):
            print(ln, flush=True)
    out_path = os.path.join(ROOT, "output", f"isaaclab_joint_probe_{tag}.pt")
    status = "OK" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
    md5 = (hashlib.md5(open(out_path, "rb").read()).hexdigest()[:12]
           if os.path.exists(out_path) else "missing")
    print(f"[il-probe] {tag}: {status} | dump md5 {md5}", flush=True)
    if proc.returncode != 0:
        print(f"[il-probe] child {tag} failed; full tail:", flush=True)
        for ln in out.splitlines()[-25:]:
            print(f"  {ln}", flush=True)
    return proc.returncode == 0


if os.environ.get("X1_IL_CHILD", "") != "1":
    print("[il-probe] driver start", flush=True)
    ok_all = True
    for tag, mode, kp, kd in CONFIGS:
        ok_all &= run_child(tag, mode, kp, kd)
    sys.exit(0 if ok_all else 1)

# ================= child =================
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True, offscreen_render=False)
simulation_app = app_launcher.app

import torch  # noqa: E402
import numpy as np  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim.spawners.from_files import UrdfFileCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402

TAG = os.environ["X1_IL_TAG"]
MODE = os.environ["X1_IL_MODE"]
KP = float(os.environ["X1_IL_KP"])
KD = float(os.environ["X1_IL_KD"])
DEVICE = os.environ.get("X1_IL_DEVICE", "cuda:0")

import isaaclab  # noqa: E402
try:
    il_ver = isaaclab.__version__
except Exception:
    il_ver = "?"
print(f"[il-probe] isaaclab {il_ver} | mode={MODE} kp={KP} kd={KD} "
      f"device={DEVICE}", flush=True)

urdf_path = os.path.join(ROOT, "output", "x1_pendulum_probe.urdf")
os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
open(urdf_path, "w").write(URDF_TEXT)

sim_cfg = sim_utils.SimulationCfg(
    device=DEVICE, dt=DT, gravity=(0.0, 0.0, -9.81))  # r1: use_gpu_pipeline
# kwarg removed in IsaacLab 3.x/6.x API
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view(eye=(2.0, 2.0, 1.0), target=(0.0, 0.0, 0.0))

act_cfg = ImplicitActuatorCfg(
    joint_names_expr=[".*"], stiffness=KP, damping=KD, armature=0.01,
    effort_limit=1000.0, velocity_limit=100.0)

art_cfg = ArticulationCfg(
    prim_path="/World/pend",
    spawn=UrdfFileCfg(asset_path=urdf_path, fix_base=True,
                      merge_fixed_joints=False),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.4)),
    actuators={"hinge": act_cfg},
)
pend = Articulation(art_cfg)
sim.reset()

# inertia sanity print (must match MuJoCo mirror; asserted in analysis)
mass = torch.as_tensor(np.asarray(pend.root_physx_view.get_masses()),
                       dtype=torch.float32).cpu()
print(f"[il-probe] masses {mass.tolist()}", flush=True)

N_ENV = 1


def run_seq(q0, target, torque_mode):
    """Run N_CTRL control steps from q0/0; return qd at ctrl boundaries."""
    qd_seq = []
    joint_idx = torch.tensor([0], device=DEVICE)
    # set initial state (fresh PhysX state; no physics stepped yet)
    joint_pos = torch.tensor([[q0]], dtype=torch.float32, device=DEVICE)
    joint_vel = torch.zeros(1, 1, dtype=torch.float32, device=DEVICE)
    pend.write_joint_state_to_sim(joint_pos, joint_vel)
    pend.update(DT)
    if torque_mode == "implicit":
        pend.set_joint_position_target(torch.tensor(
            [[target]], dtype=torch.float32, device=DEVICE))
        pend.write_data_to_sim()
    for _ in range(N_CTRL):
        for _sub in range(SUBSTEPS):
            if torque_mode == "explicit":
                # harness-parity: recompute explicit torque each substep
                # from FRESH PhysX state (update() refreshes data tensors)
                pend.update(DT)
                pos = pend.data.joint_pos[:, joint_idx].clone()
                vel = pend.data.joint_vel[:, joint_idx].clone()
                tau = KP * (target - pos) - KD * vel
                pend.set_joint_effort_target(tau)
                pend.write_data_to_sim()
            sim.step()
        pend.update(DT * SUBSTEPS)
        qd = pend.data.joint_vel[:, joint_idx].detach().cpu().numpy().copy()
        qd_seq.append(float(qd.flatten()[0]))
    return qd_seq


results = dict(step={}, zero={}, config=dict(tag=TAG, mode=MODE, kp=KP,
                                             kd=KD, armature=0.01,
                                             device=DEVICE, isaaclab=il_ver,
                                             dt=DT, substeps=SUBSTEPS,
                                             q0_offsets=Q0_OFFSETS))
for q0 in Q0_OFFSETS:
    results["step"][f"{q0}"] = run_seq(q0, 0.0, MODE)
results["zero"] = run_seq(Q0_OFFSETS[0], Q0_OFFSETS[0], MODE)

out = dict(qd_step=results["step"], qd_zero=results["zero"],
           meta=results["config"], masses=mass.tolist())
out_path = os.path.join(ROOT, "output", f"isaaclab_joint_probe_{TAG}.pt")
torch.save(out, out_path)
print(f"[il-probe] saved {out_path} md5 "
      f"{hashlib.md5(open(out_path,'rb').read()).hexdigest()[:12]}", flush=True)
print(f"[il-probe] {TAG} step q0=0.2 qd[:4] "
      f"{[round(x, 4) for x in results['step']['0.2'][:4]]}", flush=True)
# r1/r2 lesson: AppLauncher swallows exceptions and close() hard-exits;
# do the dump-existence check BEFORE close(), exit non-zero on missing
if not os.path.exists(out_path):
    print("[il-probe] FATAL: dump not produced", flush=True)
    sys.exit(1)
simulation_app.close()
