"""H-X1-IL / SW-PREREG-005: full-X1 Isaac Lab entry probe.

MjcfFileCfg direct-load of data/assets/x1/x1_v4.xml (29 joints),
ImplicitActuator with gains read from the MJCF (verified against the r4
anchor dump kp/kd, fail-fast), ground plane (friction 1.0, engine parity).

Protocol (same family as SW-R004/R005 probes):
  - state injection from anchor dump frame 0 (motion0/t0): root pose/vel
    + 29 dof pos/vel; state0 readback self-check < 1e-5 (pos & vel)
  - open-loop q_tar replay from the anchor dump (30 ctrl steps,
    4 substeps @ dt=1/120), joint order X1_DOF_ORDER <-> articulation
    order mapping printed and applied
  - zero-error probe: q_tar pinned at q0, 8 ctrl steps
  - forced copies everywhere (r3 lesson); success judged by dump existence
    (AppLauncher swallows exceptions)

Outputs: output/x1_il_entry_F1_implicit_v4.pt
"""
import hashlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)

TAG = "F1_implicit_v4"
N_ROLLOUT = 30
N_PROBE = 8
SUBSTEPS = 4
DT = 1.0 / 120.0
ANCHOR_DUMP = "output/remote_ckpt/isaac_traj_v4_solver_tgs4_0_gpu.pt"
X1_XML = "data/assets/x1/x1_v4.xml"
# per-joint gain groups computed locally from x1_v4.xml + anchor dump
# (kp,kd from MJCF joint stiffness/damping == anchor kp/kd; tlim = effort
# limits; armature from MJCF). r1 lesson: MjcfFileCfg does NOT map joint
# stiffness/damping into the PhysX drive (readback all-zero) -> supply
# explicit per-group ImplicitActuatorCfg. Runtime-verified vs dump below.
GAIN_GROUPS = [
    (20.0, 1.0, 10.0, 0.005), (40.0, 2.0, 20.0, 0.01),
    (50.0, 1.0, 80.0, 0.02), (120.0, 3.0, 150.0, 0.02),
    (120.0, 4.0, 150.0, 0.02), (150.0, 4.0, 180.0, 0.02),
    (150.0, 5.0, 180.0, 0.02), (150.0, 8.0, 180.0, 0.02),
]

sys.path.insert(0, os.path.join(ROOT, "tools/x1_pipeline"))
from retarget_g1_x1 import X1_DOF_ORDER  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher(headless=True, offscreen_render=False)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim.spawners.from_files import (  # noqa: E402
    GroundPlaneCfg, MjcfFileCfg, spawn_ground_plane)
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402

print(f"[x1-il] isaaclab {getattr(isaaclab, '__version__', '?')}", flush=True)
for p in (X1_XML, ANCHOR_DUMP):
    fp = os.path.join(ROOT, p)
    print(f"[x1-il] md5 {p} "
          f"{hashlib.md5(open(fp,'rb').read()).hexdigest()[:12]}", flush=True)

ref = torch.load(os.path.join(ROOT, ANCHOR_DUMP), map_location="cpu",
                 weights_only=False)
kp_dof = ref["kp"].numpy().astype(np.float64)     # X1_DOF_ORDER
kd_dof = ref["kd"].numpy().astype(np.float64)
tlim_dof = ref["tlim"].numpy().astype(np.float64)
q_tar_seq = ref["q_tar"].numpy().astype(np.float64)  # [40, 29] X1_DOF_ORDER

DEVICE = "cuda:0"
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    device=DEVICE, dt=DT, gravity=(0.0, 0.0, -9.81)))

physics_material = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
plane_cfg = GroundPlaneCfg(physics_material=physics_material)
spawn_ground_plane(prim_path="/World/ground", cfg=plane_cfg)

# Build per-group joint name lists by matching the anchor dump gains
# (names are unique across X1_DOF_ORDER and the MJCF articulation order,
# so explicit name lists are order-independent).
_name_groups = {g: [] for g in GAIN_GROUPS}
for _i, _n in enumerate(X1_DOF_ORDER):
    _key = (float(kp_dof[_i]), float(kd_dof[_i]), float(tlim_dof[_i]), 0.0)
    # find armature group member by kp/kd/tlim triple
    _match = [g for g in GAIN_GROUPS
              if g[0] == _key[0] and g[1] == _key[1] and g[2] == _key[2]]
    assert len(_match) == 1, f"gain triple not unique for {_n}: {_match}"
    _name_groups[_match[0]].append(_n)

actuators = {}
for _gi, _g in enumerate(GAIN_GROUPS):
    actuators[f"g{_gi}"] = ImplicitActuatorCfg(
        joint_names_expr=list(_name_groups[_g]),
        stiffness=_g[0], damping=_g[1], effort_limit=_g[2], armature=_g[3])

art_cfg = ArticulationCfg(
    prim_path="/World/x1",
    spawn=MjcfFileCfg(asset_path=X1_XML, fix_base=False),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.6)),
    actuators=actuators,
)
x1 = Articulation(art_cfg)
sim.reset()

art_names = list(x1.joint_names)
assert len(art_names) == 29, f"expected 29 joints, got {len(art_names)}"
missing = [n for n in X1_DOF_ORDER if n not in art_names]
assert not missing, f"joints missing in articulation: {missing}"
# perm[i] = index in art vector of X1_DOF_ORDER[i]
perm = np.array([art_names.index(n) for n in X1_DOF_ORDER])
print("[x1-il] joint order map (X1_DOF_ORDER idx -> art idx) "
      f"first 10: {perm[:10].tolist()}", flush=True)

# ---- gains verification (fail-fast per prereg) ----
try:
    kp_art = np.asarray(x1.root_physx_view.get_dof_stiffnesses(),
                        dtype=np.float64).reshape(-1)[:29]
    kd_art = np.asarray(x1.root_physx_view.get_dof_dampings(),
                        dtype=np.float64).reshape(-1)[:29]
except Exception as exc:  # noqa: BLE001
    print(f"[x1-il] FATAL: gain readback failed: {exc}", flush=True)
    sys.exit(1)
kp_read = kp_art[perm]
kd_read = kd_art[perm]
kp_dev = float(np.max(np.abs(kp_read - kp_dof)))
kd_dev = float(np.max(np.abs(kd_read - kd_dof)))
print(f"[x1-il] gains vs anchor dump: kp max dev {kp_dev:.3e} "
      f"kd max dev {kd_dev:.3e} (fail-fast tolerance 1e-3 relative)",
      flush=True)
print(f"[x1-il] kp_read[:6] {np.round(kp_read[:6], 2)} | "
      f"kd_read[:6] {np.round(kd_read[:6], 2)}", flush=True)
if kp_dev > 1e-3 * max(1.0, float(np.max(kp_dof))) or \
        kd_dev > 1e-3 * max(1.0, float(np.max(kd_dof))):
    print("[x1-il] FATAL: gains mismatch -> config invalid, abort",
          flush=True)
    sys.exit(1)


def inject_state(frame=0):
    rp = ref["root_pos"][frame].numpy().astype(np.float64)
    rq = ref["root_quat"][frame].numpy().astype(np.float64)  # xyzw
    rv = ref["root_vel"][frame].numpy().astype(np.float64)
    ra = ref["root_ang_vel"][frame].numpy().astype(np.float64)
    dp = ref["dof_pos"][frame].numpy().astype(np.float64)
    dv = ref["dof_vel"][frame].numpy().astype(np.float64)
    pose = torch.tensor([[rp[0], rp[1], rp[2],
                          rq[3], rq[0], rq[1], rq[2]]],  # wxyz
                        dtype=torch.float32, device=DEVICE)
    vel = torch.tensor([np.concatenate([rv, ra])],
                       dtype=torch.float32, device=DEVICE)
    x1.write_root_pose_to_sim(pose)
    x1.write_root_velocity_to_sim(vel)
    x1.write_joint_state_to_sim(
        torch.tensor(dp[perm].reshape(1, -1), dtype=torch.float32,
                     device=DEVICE),
        torch.tensor(dv[perm].reshape(1, -1), dtype=torch.float32,
                     device=DEVICE))
    x1.update(DT)
    return rp, rq, rv, ra, dp, dv


def read_state():
    rp = x1.data.root_pos_w[0].detach().cpu().numpy().copy()
    rq = x1.data.root_quat_w[0].detach().cpu().numpy().copy()  # wxyz
    rv = x1.data.root_lin_vel_w[0].detach().cpu().numpy().copy()
    ra = x1.data.root_ang_vel_w[0].detach().cpu().numpy().copy()
    dp = x1.data.joint_pos[0].detach().cpu().numpy().copy()[perm]
    dv = x1.data.joint_vel[0].detach().cpu().numpy().copy()[perm]
    rq_xyzw = np.array([rq[1], rq[2], rq[3], rq[0]])  # wxyz -> xyzw
    return dict(root_pos=rp, root_quat_w=rq, root_vel=rv, root_ang_vel=ra,
                root_quat=rq_xyzw, dof_pos=dp, dof_vel=dv)


def raw_joint_state():
    """Raw PhysX joint state (bypasses data-cache staleness), X1 order.
    IsaacLab 6.x: get_dof_positions()/get_dof_velocities() return single
    warp arrays (not tuples) -> wp.to_torch conversion."""
    import warp as _wp
    view = x1.root_physx_view
    pos = (_wp.to_torch(view.get_dof_positions()).detach().cpu()
           .numpy().astype(np.float64).reshape(-1)[:29])
    vel = (_wp.to_torch(view.get_dof_velocities()).detach().cpu()
           .numpy().astype(np.float64).reshape(-1)[:29])
    return pos[perm].copy(), vel[perm].copy()


inject_state(0)
dp_exp = ref["dof_pos"][0].numpy().astype(np.float64)
dv_exp = ref["dof_vel"][0].numpy().astype(np.float64)
s0 = read_state()
raw_pos, raw_vel = raw_joint_state()
print(f"[x1-il] state0 sources: expected dp[:6] "
      f"{np.round(dp_exp[:6], 4)} | data {np.round(s0['dof_pos'][:6], 4)} "
      f"| raw {np.round(raw_pos[:6], 4)}", flush=True)

# r5/r6 diagnosis: dump FULL vectors to derive the true view dof order
# locally (observed: values land in wrong joints; only idx0 coincidental).
if os.environ.get("X1_IL_DIAG", "") == "1":
    import warp as _wp
    _view = x1.root_physx_view
    art_full = (_wp.to_torch(_view.get_dof_positions()).cpu()
                .numpy().astype(np.float64).reshape(-1)[:29])
    print("[x1-il] DIAG expected dp (X1 order):", flush=True)
    print(f"[x1-il] {np.round(dp_exp, 4).tolist()}", flush=True)
    print("[x1-il] DIAG art view pos (no perm):", flush=True)
    print(f"[x1-il] {np.round(art_full, 4).tolist()}", flush=True)
    print(f"[x1-il] DIAG art_names: {art_names}", flush=True)
    raise RuntimeError("diag-full-vectors-dumped")

# r7 diagnosis: code the 4 ambiguous all-zero wrist joints (+left_hip_yaw)
# with unique values to pin their view slots (write in joint_names order).
if os.environ.get("X1_IL_WRIST_CODED", "") == "1":
    _probe_vals = {"left_wrist_pitch_joint": 0.11,
                   "right_wrist_pitch_joint": 0.22,
                   "left_wrist_roll_joint": 0.33,
                   "right_wrist_roll_joint": 0.44,
                   "left_hip_yaw_joint": 0.55}
    dp_art = np.zeros(29)
    for _i, _n in enumerate(X1_DOF_ORDER):
        dp_art[art_names.index(_n)] = dp_exp[_i]
    for _n, _v in _probe_vals.items():
        dp_art[art_names.index(_n)] = _v
    x1.write_joint_state_to_sim(
        torch.tensor(dp_art.reshape(1, -1), dtype=torch.float32,
                     device=DEVICE),
        torch.zeros(1, 29, dtype=torch.float32, device=DEVICE))
    x1.update(DT)
    import warp as _wp
    _view = x1.root_physx_view
    art_full = (_wp.to_torch(_view.get_dof_positions()).cpu()
                .numpy().astype(np.float64).reshape(-1)[:29])
    print("[x1-il] DIAG wrist-coded view pos (no perm):", flush=True)
    print(f"[x1-il] {np.round(art_full, 4).tolist()}", flush=True)
    raise RuntimeError("diag-wrist-coded-dumped")

pos_dev = float(np.max(np.abs(raw_pos - dp_exp)))
vel_dev = float(np.max(np.abs(raw_vel - dv_exp)))
print(f"[x1-il] state0 readback: dof_pos dev {pos_dev:.2e} "
      f"dof_vel dev {vel_dev:.2e} (must be < 1e-5)", flush=True)
assert pos_dev < 1e-5 and vel_dev < 1e-5, "state0 misaligned"

traj = dict(root_pos=[], root_quat=[], root_vel=[], root_ang_vel=[],
            dof_pos=[], dof_vel=[])
st = read_state()
st["dof_pos"] = raw_pos  # t=0 from verified raw physics state
st["dof_vel"] = raw_vel
for t in range(N_ROLLOUT):
    traj["root_pos"].append(st["root_pos"])
    traj["root_quat"].append(st["root_quat_w"])
    traj["root_vel"].append(st["root_vel"])
    traj["root_ang_vel"].append(st["root_ang_vel"])
    traj["dof_pos"].append(st["dof_pos"])
    traj["dof_vel"].append(st["dof_vel"])
    target = q_tar_seq[t]
    x1.set_joint_position_target(
        torch.tensor(target[perm].reshape(1, -1), dtype=torch.float32,
                     device=DEVICE))
    x1.write_data_to_sim()
    for _ in range(SUBSTEPS):
        sim.step()
    x1.update(DT * SUBSTEPS)
    st = read_state()

# ---- zero-error probe ----
inject_state(0)
q0 = ref["dof_pos"][0].numpy().astype(np.float64)
probe = dict(root_pos=[], root_quat=[], root_vel=[], root_ang_vel=[],
             dof_pos=[], dof_vel=[], q0=q0)
st = read_state()
tq0 = torch.tensor(q0[perm].reshape(1, -1), dtype=torch.float32,
                   device=DEVICE)
x1.set_joint_position_target(tq0)
x1.write_data_to_sim()
for t in range(N_PROBE):
    probe["root_pos"].append(st["root_pos"])
    probe["root_quat"].append(st["root_quat_w"])
    probe["root_vel"].append(st["root_vel"])
    probe["root_ang_vel"].append(st["root_ang_vel"])
    probe["dof_pos"].append(st["dof_pos"])
    probe["dof_vel"].append(st["dof_vel"])
    for _ in range(SUBSTEPS):
        sim.step()
    x1.update(DT * SUBSTEPS)
    st = read_state()

out = dict(
    traj={k: torch.tensor(np.stack(v)) for k, v in traj.items()},
    probe={k: torch.tensor(np.stack(v)) if isinstance(v, list) else
           torch.tensor(v) for k, v in probe.items()},
    meta=dict(tag=TAG, isaaclab=getattr(isaaclab, "__version__", "?"),
              device=DEVICE, dt=DT, substeps=SUBSTEPS, xml=X1_XML,
              anchor=ANCHOR_DUMP, perm=perm.tolist(),
              art_names=art_names, kp_read=kp_read.tolist(),
              kd_read=kd_read.tolist(), kp_dev=kp_dev, kd_dev=kd_dev,
              state0_pos_dev=pos_dev, state0_vel_dev=vel_dev),
    kp=ref["kp"].clone(), kd=ref["kd"].clone(), tlim=ref["tlim"].clone(),
    q_tar_seq=ref["q_tar"].clone(),
)
out_path = os.path.join(ROOT, "output", f"x1_il_entry_{TAG}.pt")
torch.save(out, out_path)
print(f"[x1-il] saved {out_path} md5 "
      f"{hashlib.md5(open(out_path,'rb').read()).hexdigest()[:12]}", flush=True)
dv1 = traj["dof_vel"][1].numpy()
i = int(np.argmax(np.abs(dv1 - ref["dof_vel"][1].numpy())))
print(f"[x1-il] step1 dof_vel[:6] {np.round(dv1[:6], 3)} | vs anchor "
      f"max {np.max(np.abs(dv1 - ref['dof_vel'][1].numpy())):.3f} at "
      f"{X1_DOF_ORDER[i]}", flush=True)

if not os.path.exists(out_path):
    sys.exit(1)
simulation_app.close()
