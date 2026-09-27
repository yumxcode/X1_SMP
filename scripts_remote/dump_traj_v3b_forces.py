"""Dump v3 policy rollout WITH dof force sensor + net contact forces.

Same as dump_traj_v3.py plus:
  * monkey-patches IsaacGymEngine._enable_dof_force_sensors -> True BEFORE
    env build (sensors must exist at actor creation)
  * records e.get_dof_forces(char_id) per step (PhysX dof force tensor:
    solver constraint force on each dof)
  * records e.get_contact_forces(char_id) per step (net per-body contact)
Purpose: MuJoCo replay-by-torque to discriminate drive-semantics vs
contact/integration as the sim2sim divergence source.
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

# enable dof force sensors BEFORE the env is built
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
import learning.agent_builder as agent_builder  # noqa: E402
from learning.base_agent import AgentMode  # noqa: E402

CKPT = os.environ.get("X1_DUMP_CKPT", "data/models/smp/smp_v3_it2200.pt")

env = env_builder.build_env("data/envs/smp_x1_env_v3.yaml",
                            "data/engines/isaac_gym_engine.yaml",
                            1, "cuda:0", visualize=False, record_video=False)
agent = agent_builder.build_agent("data/agents/smp_x1_agent_v3.yaml",
                                  env, "cuda:0")
agent.load(CKPT)
agent.eval()
agent.set_mode(AgentMode.TEST)
print(f"[dump-v3b] loaded {CKPT}", flush=True)

char_id = env._get_char_id()
e = env._engine
agent._curr_obs, agent._curr_info = agent._reset_envs()

kp, kd = e.get_obj_pd_gains(0, char_id)
kp = np.asarray(kp, dtype=np.float64)
kd = np.asarray(kd, dtype=np.float64)
tlim = np.asarray(to_np(e.get_obj_torque_limits(0, char_id)), dtype=np.float64)

traj = dict(obs=[], root_pos=[], root_quat=[], root_vel=[],
            root_ang_vel=[], dof_pos=[], dof_vel=[], action=[],
            q_tar=[], dof_force=[], contact_forces=[])
N = 120
for t in range(N):
    action, _ = agent._decide_action(agent._curr_obs, agent._curr_info)
    traj["obs"].append(to_np(agent._curr_obs[0]))
    traj["root_pos"].append(to_np(e.get_root_pos(char_id)[0]))
    traj["root_quat"].append(to_np(e.get_root_rot(char_id)[0]))
    traj["root_vel"].append(to_np(e.get_root_vel(char_id)[0]))
    traj["root_ang_vel"].append(to_np(e.get_root_ang_vel(char_id)[0]))
    traj["dof_pos"].append(to_np(e.get_dof_pos(char_id)[0]))
    traj["dof_vel"].append(to_np(e.get_dof_vel(char_id)[0]))
    traj["action"].append(to_np(action[0]))
    try:
        traj["dof_force"].append(to_np(e.get_dof_forces(char_id)[0]).copy())
    except Exception as ex:
        print(f"[dump-v3b] dof_forces failed: {ex}", flush=True)
        traj["dof_force"].append(np.zeros(29))
    try:
        traj["contact_forces"].append(
            to_np(e.get_contact_forces(char_id)[0]).copy())
    except Exception as ex:
        print(f"[dump-v3b] contact_forces failed: {ex}", flush=True)
        traj["contact_forces"].append(np.zeros((1, 3)))
    nobs, r, done, info = agent._step_env(action)
    traj["q_tar"].append(np.asarray(to_np(e._get_dof_cmd_buf()[0]),
                                    dtype=np.float64))
    agent._curr_obs, agent._curr_info = agent._reset_done_envs(done)

out = {k: torch.tensor(np.stack(v)) for k, v in traj.items() if v}
out["kp"] = torch.tensor(kp)
out["kd"] = torch.tensor(kd)
out["tlim"] = torch.tensor(tlim)
torch.save(out, os.path.join(ROOT, "output", "isaac_traj_v3b.pt"))
df = np.stack(traj["dof_force"])
print(f"[dump-v3b] saved output/isaac_traj_v3b.pt ({N} steps); "
      f"dof_force |sum| {np.abs(df).sum():.1f}", flush=True)
