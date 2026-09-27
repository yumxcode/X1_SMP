"""Isaac position-drive system identification: step-response probe.

From a standing reset, hold a constant q_tar = q0 + delta on a few joints,
step 60 control steps, record q/qd + dof force sensors. Locally we fit the
effective drive gains (kp_eff, kd_eff) that reproduce the response, and use
them in the MuJoCo sim2sim harness (replicating the dynamics the policy
actually trained under).
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

import engines.isaac_gym_engine as _ige  # noqa: E402
_ige.IsaacGymEngine._enable_dof_force_sensors = lambda self: True

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

# probe joints (indices in the X1 depth-first dof order:
# lumbar 0-2, left arm 3-9, right arm 10-16, left leg 17-22, right leg 23-28)
PROBE = {"right_ankle_roll_joint": 28, "left_ankle_pitch_joint": 21,
         "right_hip_pitch_joint": 23, "right_knee_pitch_joint": 26,
         "left_shoulder_pitch_joint": 3, "lumbar_yaw_joint": 0}
DELTA = 0.5

out = dict(dof_pos=[], dof_vel=[], dof_force=[], q_tar=[],
           probe=list(PROBE), delta=DELTA)

env.reset()
obs = env._get_curr_obs() if hasattr(env, "_get_curr_obs") else None
q0 = to_np(e.get_dof_pos(char_id)[0]).copy()
q_tar = q0.copy()
for k in PROBE.values():
    q_tar[k] += DELTA

kp, kd = e.get_obj_pd_gains(0, char_id)
out["kp_reported"] = np.asarray(kp, dtype=np.float64)
out["kd_reported"] = np.asarray(kd, dtype=np.float64)
out["q0"] = q0

# command via the engine's raw cmd buffer (1 env, 1 obj, 29 dofs)
for t in range(60):
    out["dof_pos"].append(to_np(e.get_dof_pos(char_id)[0]).copy())
    out["dof_vel"].append(to_np(e.get_dof_vel(char_id)[0]).copy())
    try:
        out["dof_force"].append(to_np(e.get_dof_forces(char_id)[0]).copy())
    except Exception:
        out["dof_force"].append(np.zeros(29))
    e._dof_cmd_raw[0, :29] = torch.tensor(q_tar, device=e._dof_cmd_raw.device)
    e.step()

torch.save({k: (torch.tensor(np.stack(v)) if isinstance(v, list)
                and isinstance(v[0], np.ndarray) else v)
            for k, v in out.items()},
           os.path.join(ROOT, "output", "isaac_drive_sysid.pt"))
print("[sysid] saved output/isaac_drive_sysid.pt "
      f"(60 steps, delta {DELTA}, probes {list(PROBE)})", flush=True)
