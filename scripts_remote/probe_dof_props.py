"""Probe Isaac dof_props: velocity limits + step-response clamp test.

Prints dof_props['velocity'] (PhysX maximumJointVelocity under DOF_MODE_POS
clamps joint speed at this value). Then from standing, commands a large
q_tar step on a few joints WITHOUT ground (spawn high) and records peak
|qd| per joint vs the commanded PD-response prediction.
"""
import os
import sys
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "--no-index", "--no-deps", "--find-links",
                       os.path.join(ROOT, "vendor_wheels"),
                       "gymnasium", "farama-notifications", "cloudpickle",
                       "diffusers", "huggingface-hub", "filelock",
                       "importlib-metadata", "zipp", "packaging",
                       "typing-extensions", "regex", "tqdm", "safetensors",
                       "requests", "matplotlib", "contourpy", "cycler",
                       "fonttools", "kiwisolver", "pyparsing",
                       "python-dateutil", "six", "tensorboardX", "protobuf"])

sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, ROOT)
import envs.env_builder as env_builder  # noqa: E402
import numpy as np  # noqa: E402


def to_np(x):
    import torch
    try:
        return torch.as_tensor(x).detach().cpu().numpy()
    except Exception:
        return np.asarray(x)
import torch  # noqa: E402

env = env_builder.build_env("data/envs/smp_x1_env_v3.yaml",
                            "data/engines/isaac_gym_engine.yaml",
                            1, "cuda:0", visualize=False, record_video=False)
char_id = env._get_char_id()
e = env._engine
gym = e._gym
env_ptr = e.get_env(0)

dof_props = gym.get_actor_dof_properties(env_ptr, char_id)
vel = dof_props["velocity"]
stiff = dof_props["stiffness"]
damp = dof_props["damping"]
names = gym.get_actor_dof_names(env_ptr, char_id)
print("[probe] joint velocity limits (PhysX maximumJointVelocity):", flush=True)
for n, v, k, dd in zip(names, vel, stiff, damp):
    print(f"  {n:32s} vel {v:8.3f}  kp {k:7.1f}  kd {dd:6.2f}", flush=True)

# step test: lift robot 1m (no ground contact), hold q0
env.reset()
rp = e.get_root_pos(char_id)
e.set_root_pos(char_id, rp.clone())
import math
q0 = to_np(e.get_dof_pos(char_id)[0]).copy()
out = dict(q=[], qd=[], q_tar=[])
q_tar = q0.copy()
DELTA = 0.8
q_tar[3] += DELTA      # left_shoulder_pitch
q_tar[23] -= DELTA     # right_hip_pitch
q_tar[26] += DELTA     # right_knee_pitch

for t in range(60):
    out["q"].append(to_np(e.get_dof_pos(char_id)[0]).copy())
    out["qd"].append(to_np(e.get_dof_vel(char_id)[0]).copy())
    e._dof_cmd_raw[0, :29] = torch.tensor(q_tar, device=e._dof_cmd_raw.device)
    e.step()

q = np.stack(out["q"]); qd = np.stack(out["qd"])
for idx, nm in ((3, "left_shoulder_pitch"), (23, "right_hip_pitch"),
                (26, "right_knee_pitch")):
    peak_qd = np.abs(qd[:, idx]).max()
    print(f"[probe] {nm:20s} delta {DELTA}: peak|qd| {peak_qd:.2f} rad/s, "
          f"final q-q0 {q[-1, idx]-q0[idx]:+.3f} rad", flush=True)
print("[probe] done", flush=True)
